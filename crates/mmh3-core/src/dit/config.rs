/// DiT hyperparameters. The released models have hidden 5376, 56 heads of 128, FFN 14336 and 50
/// blocks. MiniMax-H3-Image is the same DiT without the audio stream and with a bank of per-step
/// video heads.
#[derive(Clone, Debug, PartialEq)]
pub struct DitConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub text_dim: usize,
    pub video_channels: usize,
    /// 0 for a checkpoint without the audio stream.
    pub audio_channels: usize,
    /// Video heads in the final layer's bank, one per step of the grid a PDD checkpoint was
    /// distilled on, and 1 for a plain head.
    pub pdd_steps: usize,
    /// The sigma grid of the bank, `pdd_steps + 1` points from 1 down to 0, when the checkpoint's
    /// metadata names it.
    pub pdd_sigmas: Option<Vec<f64>>,
    /// Width of the time-embedding coordinates the AdaLN projections consume.
    pub adaln_rank: usize,
    /// Rows of the AdaLN curve table, covering t in [0, 1].
    pub adaln_grid: usize,
    /// RoPE frequencies per axis. Three axes rotate 6 × this many head dimensions.
    pub rope_frequencies: usize,
    pub norm_eps: f32,
}

/// Latent channels are patchified by 1 × 2 × 2 (time, height, width).
pub const PATCH_AREA: usize = 4;

impl DitConfig {
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn video_patch_features(&self) -> usize {
        self.video_channels * PATCH_AREA
    }

    pub fn has_audio(&self) -> bool {
        self.audio_channels != 0
    }

    pub fn rope_dims(&self) -> usize {
        self.rope_frequencies * 6
    }

    /// The shape of a DiT read from its checkpoint header alone, with no weights uploaded
    /// anywhere. A leader that hands every step to its ranks and only puts the parts back
    /// together needs this and none of the 21 GB behind it.
    pub fn of(file: &crate::safetensors::SafeTensors, prefix: &str) -> Result<Self, String> {
        Self::from_shapes(|name| {
            file.get(&format!("{prefix}{name}"))
                .map(|info| info.shape.clone())
        })?
        .with_metadata(file.metadata())
    }

    /// Takes the PDD bank's steps and sigma grid from the `transformer` section of the `config`
    /// metadata, which ComfyUI's checkpoints carry, and checks them against the head bank's shape.
    pub fn with_metadata(mut self, metadata: &[(String, String)]) -> Result<Self, String> {
        let Some((_, text)) = metadata.iter().find(|(key, _)| key == "config") else {
            return Ok(self);
        };
        let config =
            crate::json::parse(text).map_err(|error| format!("config metadata: {error}"))?;
        let Some(transformer) = config.get("transformer") else {
            return Ok(self);
        };
        if let Some(steps) = transformer.get("pdd_steps") {
            let steps = steps
                .as_u64()
                .ok_or("config metadata: pdd_steps is not a count")?
                as usize;
            if steps != self.pdd_steps {
                return Err(format!(
                    "the metadata names {steps} PDD steps, and final_layer.video_out holds {} heads",
                    self.pdd_steps
                ));
            }
        }
        if let Some(sigmas) = transformer.get("pdd_sigmas") {
            let sigmas: Vec<f64> = sigmas
                .as_array()
                .and_then(|values| values.iter().map(crate::json::Value::as_f64).collect())
                .ok_or("config metadata: pdd_sigmas is not a list of numbers")?;
            if sigmas.len() != self.pdd_steps + 1
                || sigmas[0] != 1.0
                || sigmas[sigmas.len() - 1] != 0.0
                || sigmas.windows(2).any(|pair| pair[1] >= pair[0])
            {
                return Err(format!(
                    "config metadata: pdd_sigmas {sigmas:?} is not a grid of {} steps from 1 to 0",
                    self.pdd_steps
                ));
            }
            self.pdd_sigmas = Some(sigmas);
        }
        Ok(self)
    }

    /// Whether this checkpoint has the VSA gate projections, which decides the regions a shared
    /// step exchanges through. Read from the header beside the shape.
    pub fn gated(file: &crate::safetensors::SafeTensors, prefix: &str) -> bool {
        file.get(&format!("{prefix}blocks.0.attn.to_gate_compress.weight"))
            .is_some()
    }

    /// Infers the configuration of a pruned checkpoint from tensor shapes, keyed by checkpoint
    /// tensor name.
    pub fn from_shapes(shape_of: impl Fn(&str) -> Option<Vec<usize>>) -> Result<Self, String> {
        let shape = |name: &str| shape_of(name).ok_or_else(|| format!("missing tensor {name}"));
        let count = |prefix: &str| {
            (0..)
                .take_while(|index| shape_of(&format!("{prefix}.{index}.norm1.weight")).is_some())
                .count()
        };

        let video_patch = shape("video_patch_proj.weight")?;
        let head_dim = shape("blocks.0.attn.q_norm.weight")?[0];
        let qkv = shape("blocks.0.attn.qkv_proj.weight")?;
        let table = shape("adaln_t_table")?;
        let video_features = video_patch[1];
        let heads = shape("final_layer.video_out.weight")?[0];
        if video_features % PATCH_AREA != 0
            || qkv[0] % (3 * head_dim) != 0
            || heads == 0
            || heads % video_features != 0
        {
            return Err("unexpected projection shapes".to_owned());
        }
        Ok(DitConfig {
            hidden: video_patch[0],
            heads: qkv[0] / 3 / head_dim,
            head_dim,
            ffn: shape("blocks.0.mlp.fc2.weight")?[1],
            layers: count("blocks"),
            refiner_layers: count("token_refiner.blocks"),
            text_dim: shape("condition_proj.weight")?[1],
            video_channels: video_features / PATCH_AREA,
            audio_channels: shape_of("audio_patch_proj.weight").map_or(0, |shape| shape[1]),
            pdd_steps: heads / video_features,
            pdd_sigmas: None,
            adaln_rank: table[1],
            adaln_grid: table[0],
            rope_frequencies: shape("rope.inv_freq")?[0],
            norm_eps: 1e-5,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes(audio: bool, heads: usize) -> impl Fn(&str) -> Option<Vec<usize>> {
        move |name: &str| {
            Some(match name {
                "video_patch_proj.weight" => vec![192, 96],
                "audio_patch_proj.weight" if audio => vec![192, 32],
                "final_layer.video_out.weight" => vec![96 * heads, 192],
                "blocks.0.attn.q_norm.weight" => vec![128],
                "blocks.0.attn.qkv_proj.weight" => vec![768, 192],
                "blocks.0.mlp.fc2.weight" => vec![192, 256],
                "blocks.0.norm1.weight" | "token_refiner.blocks.0.norm1.weight" => vec![192],
                "condition_proj.weight" => vec![192, 64],
                "adaln_t_table" => vec![1025, 8],
                "rope.inv_freq" => vec![16],
                _ => return None,
            })
        }
    }

    fn metadata(transformer: &str) -> Vec<(String, String)> {
        vec![(
            "config".to_owned(),
            format!("{{\"transformer\": {transformer}}}"),
        )]
    }

    #[test]
    fn reads_a_video_checkpoint() {
        let config = DitConfig::from_shapes(shapes(true, 1))
            .unwrap()
            .with_metadata(&[("format".to_owned(), "pt".to_owned())])
            .unwrap();
        assert_eq!(
            (config.audio_channels, config.pdd_steps, &config.pdd_sigmas),
            (32, 1, &None)
        );
        assert!(config.has_audio());
    }

    #[test]
    fn reads_an_image_checkpoint_with_its_head_bank() {
        let inferred = DitConfig::from_shapes(shapes(false, 6)).unwrap();
        assert_eq!((inferred.audio_channels, inferred.pdd_steps), (0, 6));
        assert!(!inferred.has_audio());
        let sigmas = [
            1.0,
            0.9675236940383911,
            0.8875485062599182,
            0.75,
            0.5327739119529724,
            0.23200887441635132,
            0.0,
        ];
        let config = inferred
            .clone()
            .with_metadata(&metadata(&format!(
                "{{\"pdd_steps\": 6, \"pdd_sigmas\": {sigmas:?}}}"
            )))
            .unwrap();
        assert_eq!(config.pdd_sigmas.as_deref(), Some(&sigmas[..]));
        // The tiny fixture's metadata names ComfyUI's constructor arguments, with no transformer
        // section.
        let plain = vec![("config".to_owned(), "{\"hidden_size\": 192}".to_owned())];
        assert_eq!(inferred.clone().with_metadata(&plain).unwrap(), inferred);
    }

    #[test]
    fn refuses_metadata_that_does_not_match_the_heads() {
        let config = DitConfig::from_shapes(shapes(false, 6)).unwrap();
        assert!(
            config
                .clone()
                .with_metadata(&metadata("{\"pdd_steps\": 4}"))
                .is_err()
        );
        assert!(
            config
                .clone()
                .with_metadata(&metadata("{\"pdd_sigmas\": [1.0, 0.5, 0.0]}"))
                .is_err()
        );
        assert!(
            config
                .with_metadata(&metadata(
                    "{\"pdd_sigmas\": [1.0, 0.9, 0.8, 0.8, 0.5, 0.2, 0.0]}"
                ))
                .is_err()
        );
        assert!(DitConfig::from_shapes(shapes(false, 0)).is_err());
    }
}
