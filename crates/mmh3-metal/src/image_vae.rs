//! MiniMax-H3-Image VAE decoding on Metal: a 2D ViT over 256-pixel tiles, several tiles at a time.
use crate::{
    AttentionPrecision, Buffer, Error, LinearPrecision, Result,
    model::Weights,
    ops::{Array, Packing, attends_in_half},
};
use mmh3_core::{
    image_vae::{LATENT_CHANNELS, LATENTS_MEAN, LATENTS_STD, rope_angles, tiles},
    picture::Picture,
    safetensors::{DType, SafeTensors},
    tensor::Tensor,
    vae::SPATIAL_RATIO,
};
use std::{collections::HashMap, path::Path};

/// Tokens a batch of tiles decodes together: enough rows to fill the matrix units, at about a
/// gigabyte of activations.
const BATCH_TOKENS: usize = 16384;

/// The decoder of `vae/minimax_h3_image_vae_fp16.safetensors`.
pub struct MetalImageDecoder {
    file: SafeTensors,
    prefix: String,
    weights: Weights,
    /// FP16 copies of the transformer's products for the matrix units, which then leave `weights`.
    half: HashMap<String, (Buffer, usize)>,
    dim: usize,
    layers: usize,
    registers: Vec<f32>,
    tiles_per_batch: Option<usize>,
}

impl MetalImageDecoder {
    pub fn load(path: &Path) -> Result<Self> {
        let file = SafeTensors::open(path).map_err(|error| Error::new(error.to_string()))?;
        Self::from_file(&file, "")
    }

    /// A decoder of the tensors of `file` whose names start with `prefix`.
    pub fn from_file(file: &SafeTensors, prefix: &str) -> Result<Self> {
        let shape = |name: &str| -> Result<Vec<usize>> {
            Ok(file
                .get(&format!("{prefix}{name}"))
                .ok_or_else(|| Error::new(format!("missing tensor {prefix}{name}")))?
                .shape
                .clone())
        };
        let embedder = shape("decoder.x_embedder.weight")?;
        let registers = shape("decoder.register_tokens")?;
        let dim = embedder[0];
        let layers = (0..)
            .take_while(|i| {
                file.get(&format!("{prefix}decoder.transformer_blocks.{i}.scale1"))
                    .is_some()
            })
            .count();
        if embedder != [dim, LATENT_CHANNELS]
            || dim == 0
            || !dim.is_multiple_of(64)
            || layers == 0
            || registers.len() != 3
            || registers[2] != dim
            || shape("decoder.proj_out.weight")? != [3 * SPATIAL_RATIO * SPATIAL_RATIO, dim]
        {
            return Err(Error::new("unsupported image decoder configuration".into()));
        }

        let weights = Self::upload(file, prefix, &HashMap::new())?;
        let registers = weights.host("decoder.register_tokens")?.data;
        Ok(Self {
            file: SafeTensors::open(file.path()).map_err(|error| Error::new(error.to_string()))?,
            prefix: prefix.to_owned(),
            weights,
            half: HashMap::new(),
            dim,
            layers,
            registers,
            tiles_per_batch: None,
        })
    }

    fn upload(
        file: &SafeTensors,
        prefix: &str,
        half: &HashMap<String, (Buffer, usize)>,
    ) -> Result<Weights> {
        Weights::load_selected(file, prefix, |name| {
            (name.starts_with("decoder.") || name.starts_with("post_quant_conv."))
                && name
                    .strip_suffix(".weight")
                    .is_none_or(|layer| !half.contains_key(layer))
        })
    }

    /// The products of a block and the output projection: every product with rows wide enough
    /// for the matrix units.
    fn products(&self) -> Vec<String> {
        (0..self.layers)
            .flat_map(|i| {
                ["attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2"]
                    .map(|layer| format!("decoder.transformer_blocks.{i}.{layer}"))
            })
            .chain(["decoder.proj_out".to_owned()])
            .collect()
    }

    /// Select the precisions of the products and the attention. Products at `Fp16` or `Int8` take
    /// FP16 weights on the matrix units, since the checkpoint holds no INT8 ones.
    pub fn set_precision(
        &mut self,
        linear: LinearPrecision,
        attention: AttentionPrecision,
    ) -> Result<()> {
        let half = matches!(linear, LinearPrecision::Fp16 | LinearPrecision::Int8);
        let device = self.weights.device.clone();
        if half && !device.supports_tensor_ops() {
            return Err(Error::new(
                "FP16 products of the image decoder require macOS 26 and Apple silicon".into(),
            ));
        }
        if half && self.half.is_empty() {
            for layer in self.products() {
                let info = self
                    .file
                    .get(&format!("{}{layer}.weight", self.prefix))
                    .ok_or_else(|| Error::new(format!("missing tensor {layer}.weight")))?;
                let buffer = match info.dtype {
                    DType::F16 => device.alloc(info.byte_count(), Some(self.file.data(info)))?,
                    DType::I8 => continue,
                    _ => self.weights.array(&format!("{layer}.weight"))?.to_half()?,
                };
                self.half.insert(layer, (buffer, info.shape[0]));
            }
            self.weights = Self::upload(&self.file, &self.prefix, &self.half)?;
        } else if !half && !self.half.is_empty() {
            self.half.clear();
            self.weights = Self::upload(&self.file, &self.prefix, &self.half)?;
        }
        self.weights.set_precision((linear, attention))
    }

    /// Tiles decoded together, which bounds the memory a decode takes; by default as many as fit
    /// in about 16 thousand tokens.
    pub fn set_tiles_per_batch(&mut self, tiles: usize) {
        self.tiles_per_batch = (tiles > 0).then_some(tiles);
    }

    fn linear(&self, x: &Array, name: &str) -> Result<Array> {
        let w = &self.weights;
        match self.half.get(name) {
            Some((weight, outputs)) => x.pack(Packing::Plain, LinearPrecision::Fp16)?.product_half(
                weight,
                *outputs,
                Some(&w.vector(&format!("{name}.bias"))?),
            ),
            None => w.linear(x, name),
        }
    }

    /// Decodes `tiles` tiles of `height × width` latent tokens. `latent_rows` holds each token's
    /// channels for every tile in turn, `[token][tile][channel]`, which is how the transformer
    /// keeps them: a token's rows of all the tiles side by side make one row of the attention,
    /// whose heads are then every tile's. Returns the pixels `[tile][3][height × 16][width × 16]`
    /// in the decoder's ImageNet-normalized space.
    fn decode_tiles(
        &self,
        latent_rows: &[f32],
        tiles: usize,
        height: usize,
        width: usize,
        angles: &Array,
    ) -> Result<Array> {
        let w = &self.weights;
        let (dim, patches) = (self.dim, height * width);
        let registers = self.registers.len() / dim;
        let rows = Array::from_f32(&w.device, patches * tiles, LATENT_CHANNELS, latent_rows)?;
        let rows = w.linear(&w.linear(&rows, "post_quant_conv")?, "decoder.x_embedder")?;
        let suffix: Vec<f32> = self
            .registers
            .chunks_exact(dim)
            .flat_map(|register| std::iter::repeat_n(register, tiles).flatten())
            .copied()
            .collect();
        let mut x = Array::concat(
            &[
                rows,
                Array::from_f32(&w.device, registers * tiles, dim, &suffix)?,
                Array::zeros(&w.device, tiles, dim)?,
            ],
            false,
        )?;
        let (tokens, heads) = (patches + registers + 1, dim / 64);
        let ones = Array::from_f32(&w.device, 1, 64, &[1.0; 64])?;
        let half = attends_in_half(&w.device, w.attention_precision, 64);

        for i in 0..self.layers {
            let p = format!("decoder.transformer_blocks.{i}");
            let qkv = self
                .linear(
                    &w.norm(&x, &format!("{p}.norm1"), 1e-5)?,
                    &format!("{p}.attn.to_qkv"),
                )?
                .reshape(tokens, tiles * 3 * dim)?;
            let attended = qkv
                .attention_inputs(
                    tiles * heads,
                    (&ones, &ones),
                    1e-5,
                    Some(angles),
                    half,
                    true,
                    false,
                )?
                .attend()?
                .reshape(tokens * tiles, dim)?;
            let attended = self.linear(&attended, &format!("{p}.attn.to_out"))?;
            x = x.add_scaled(&attended, &w.vector(&format!("{p}.scale1"))?)?;
            let normed = w.norm(&x, &format!("{p}.norm2"), 1e-5)?;
            let w1 = format!("{p}.ff.w1");
            let w2 = format!("{p}.ff.w2");
            let delta = if self.half.contains_key(&w2) {
                self.linear(&self.linear(&normed, &w1)?.swiglu()?, &w2)?
            } else if w.is_int8(&w1) {
                w.linear(&w.linear_swiglu(&normed, &w1)?, &w2)?
            } else {
                w.linear_gated(&w.linear(&normed, &w1)?, &w2)?
            };
            x = x.add_scaled(&delta, &w.vector(&format!("{p}.scale2"))?)?;
        }

        // The suffix tokens are projected too, at a few rows a tile, and the unpatching skips them.
        let projected = self.linear(&w.norm(&x, "decoder.norm_out", 1e-5)?, "decoder.proj_out")?;
        let (h, wp) = (height * SPATIAL_RATIO, width * SPATIAL_RATIO);
        let pixels = Array::empty(&w.device, tiles * 3, h * wp)?;
        w.device.run(
            "video_unpatch",
            &[&projected.buffer, &pixels.buffer],
            &[pixels.len() as u32, 1, h as u32, wp as u32, 1, tiles as u32],
            pixels.len(),
            false,
        )?;
        Ok(pixels)
    }

    /// Planar RGB `[3, height, width]` in [0, 1] of a DiT latent `[24, height / 16, width / 16]`,
    /// or `[24, 1, height / 16, width / 16]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let (lh, lw) = match latent.shape.as_slice() {
            &[LATENT_CHANNELS, lh, lw] | &[LATENT_CHANNELS, 1, lh, lw] if lh > 0 && lw > 0 => {
                (lh, lw)
            }
            _ => {
                return Err(Error::new(format!(
                    "an image latent must be [{LATENT_CHANNELS}, height, width], not {:?}",
                    latent.shape
                )));
            }
        };
        let device = &self.weights.device;
        let (height, width) = (lh * SPATIAL_RATIO, lw * SPATIAL_RATIO);
        let (rows, columns) = tiles(height, width);
        let (th, tw) = (rows.length, columns.length);
        let (tlh, tlw) = (th / SPATIAL_RATIO, tw / SPATIAL_RATIO);
        let registers = self.registers.len() / self.dim;
        let tokens = tlh * tlw + registers + 1;
        let angles = Array::from_f32(
            device,
            tokens,
            2 * mmh3_core::image_vae::ROPE_FREQUENCIES,
            &rope_angles(tlh, tlw, registers + 1),
        )?;
        let per_batch = self
            .tiles_per_batch
            .unwrap_or((BATCH_TOKENS / tokens).max(1));

        let places: Vec<(usize, usize)> = (0..rows.starts.len())
            .flat_map(|row| (0..columns.starts.len()).map(move |column| (row, column)))
            .collect();
        let canvas = Array::empty(device, 3, height * width)?;
        let mut above: Vec<Option<Array>> = vec![None; columns.starts.len()];
        let mut left: Option<Array> = None;
        for batch in places.chunks(per_batch) {
            let mut values = Vec::with_capacity(tlh * tlw * batch.len() * LATENT_CHANNELS);
            for y in 0..tlh {
                for x in 0..tlw {
                    for &(row, column) in batch {
                        let index = (rows.starts[row] / SPATIAL_RATIO + y) * lw
                            + columns.starts[column] / SPATIAL_RATIO
                            + x;
                        values.extend((0..LATENT_CHANNELS).map(|c| {
                            latent.data[c * lh * lw + index] * LATENTS_STD[c] + LATENTS_MEAN[c]
                        }));
                    }
                }
            }
            let decoded = self.decode_tiles(&values, batch.len(), tlh, tlw, &angles)?;

            for (index, &(row, column)) in batch.iter().enumerate() {
                let tile = decoded.slice(3 * index, 3, 0, th * tw)?;
                let (top, start) = (rows.starts[row], columns.starts[column]);
                let kept_h = rows.starts.get(row + 1).map_or(height, |&next| next) - top;
                let kept_w = columns.starts.get(column + 1).map_or(width, |&next| next) - start;
                let overlap_y = if row == 0 { 0 } else { rows.overlaps[row - 1] };
                let overlap_x = if column == 0 {
                    0
                } else {
                    columns.overlaps[column - 1]
                };
                let n = 3 * kept_h * kept_w;
                device.run(
                    "video_blend_tile",
                    &[
                        &tile.buffer,
                        &above[column].as_ref().unwrap_or(&tile).buffer,
                        &left.as_ref().unwrap_or(&tile).buffer,
                        &canvas.buffer,
                    ],
                    &[
                        n as u32,
                        width as u32,
                        height as u32,
                        th as u32,
                        tw as u32,
                        top as u32,
                        start as u32,
                        kept_h as u32,
                        kept_w as u32,
                        overlap_y as u32,
                        overlap_x as u32,
                    ],
                    n,
                    false,
                )?;
                above[column] = Some(tile.clone());
                left = Some(tile);
            }
        }

        let plane = height * width;
        let pixels = Array::empty(device, 3, plane)?;
        device.run(
            "video_write_frames",
            &[&canvas.buffer, &canvas.buffer, &pixels.buffer],
            &[pixels.len() as u32, plane as u32, 1, 1, 0, 1, 0, 0],
            pixels.len(),
            false,
        )?;
        Ok(Tensor::new(vec![3, height, width], pixels.to_f32()?))
    }

    /// `decode` as an 8-bit picture, truncated as the reference converts it.
    pub fn decode_picture(&self, latent: &Tensor) -> Result<Picture> {
        Ok(mmh3_core::image_vae::to_picture(&self.decode(latent)?))
    }
}
