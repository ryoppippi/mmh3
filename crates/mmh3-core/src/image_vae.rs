//! MiniMax-H3-Image VAE decoding plans shared by every backend: the latent statistics, the tiles
//! and the rotary angles of its 2D ViT decoder.
//!
//! The decoder turns each latent token into a 16 × 16 block of pixels. It runs on 256-pixel tiles
//! that overlap by at least 64 pixels, as the reference always does, and blends the overlaps
//! linearly against the neighbours' unblended decodes.

use crate::picture::Picture;
use crate::tensor::Tensor;
use crate::vae::{TileAxis, rope_angles as video_rope_angles, split_tiles};

/// Latent channels of a MiniMax-H3-Image latent.
pub const LATENT_CHANNELS: usize = 24;
/// Side of the square tiles a decode takes.
pub const TILE_SIZE: usize = 256;
/// Pixels neighbouring tiles overlap by at least.
pub const TILE_OVERLAP_MIN: usize = 64;
/// Rotary frequencies per axis and their base.
pub const ROPE_FREQUENCIES: usize = 8;
pub const ROPE_BASE: f32 = 100.0;

/// The per-channel statistics that take a DiT latent back to the VAE's: `z × std + mean`. They are
/// the release's `vae/config.json`; the single-file checkpoint does not carry them.
pub const LATENTS_MEAN: [f32; LATENT_CHANNELS] = [
    0.85809034,
    -0.96065915,
    1.066164,
    -0.50903255,
    -0.2727582,
    -1.3675414,
    -0.2553255,
    -0.26907554,
    -0.5376841,
    -0.04640973,
    0.66573703,
    0.19690128,
    -0.5460608,
    -0.4035342,
    -0.23683025,
    0.25928453,
    -0.30133945,
    0.21134199,
    -1.1206849,
    0.35819334,
    -0.042251438,
    0.260483,
    0.22864093,
    0.7056032,
];
pub const LATENTS_STD: [f32; LATENT_CHANNELS] = [
    1.2223774, 1.2767264, 1.6831775, 1.7549455, 1.5636216, 2.1941435, 0.9653138, 1.0569886,
    0.8419489, 0.7729953, 1.8955938, 0.94684184, 0.79968095, 0.449889, 0.71974, 0.6936293,
    2.961095, 2.76942, 3.0496185, 2.1088054, 3.2762263, 3.1627357, 2.2816813, 2.6127844,
];

/// The tiles of a picture of `height × width` pixels, rows then columns.
pub fn tiles(height: usize, width: usize) -> (TileAxis, TileAxis) {
    (
        split_tiles(height, TILE_SIZE, TILE_OVERLAP_MIN),
        split_tiles(width, TILE_SIZE, TILE_OVERLAP_MIN),
    )
}

/// Rotary angles `[tokens + suffix, 2 × ROPE_FREQUENCIES]` of one tile of `height × width` latent
/// tokens, rows then columns: the video decoder's without its time axis, whose single frame sits
/// at the origin.
pub fn rope_angles(height: usize, width: usize, suffix: usize) -> Vec<f32> {
    video_rope_angles(1, height, width, suffix, ROPE_FREQUENCIES, ROPE_BASE)
        .as_chunks::<{ 3 * ROPE_FREQUENCIES }>()
        .0
        .iter()
        .flat_map(|row| &row[ROPE_FREQUENCIES..])
        .copied()
        .collect()
}

/// An 8-bit picture of planar RGB `[3, height, width]` in [0, 1], truncated as the reference
/// converts it.
pub fn to_picture(pixels: &Tensor) -> Picture {
    let &[3, height, width] = pixels.shape.as_slice() else {
        panic!("pixels must be [3, height, width]");
    };
    let plane = height * width;
    let pixels = (0..plane)
        .flat_map(|i| (0..3).map(move |c| (c, i)))
        .map(|(c, i)| (pixels.data[c * plane + i].clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    Picture {
        width,
        height,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_like_the_reference() {
        let (rows, columns) = tiles(4096, 4096);
        assert_eq!(rows.starts.len(), 21);
        assert_eq!(rows.overlaps, vec![64; 20]);
        assert_eq!(columns, rows);
        // 384 pixels fit two tiles, and the 64 pixels they have to spare widen the one overlap.
        let (rows, columns) = tiles(384, 640);
        assert_eq!((rows.starts, rows.overlaps), (vec![0, 128], vec![128]));
        assert_eq!(
            (columns.starts, columns.overlaps),
            (vec![0, 192, 384], vec![64, 64])
        );
        assert_eq!(tiles(256, 128).1.length, 128);
    }

    #[test]
    fn rotates_rows_then_columns() {
        let angles = rope_angles(2, 4, 5);
        assert_eq!(angles.len(), (2 * 4 + 5) * 16);
        let coordinate = |index: usize, size: usize| (index as f32 + 0.5) / size as f32 * 2.0 - 1.0;
        let token = 4 + 3;
        for k in 0..8 {
            let frequency = 100f32.powf(-(k as f32) / 8.0);
            let angle = |position: f32| 2.0 * std::f32::consts::PI * position * frequency;
            assert_eq!(angles[token * 16 + k], angle(coordinate(1, 2)));
            assert_eq!(angles[token * 16 + 8 + k], angle(coordinate(3, 4)));
        }
        assert!(angles[8 * 16..].iter().all(|&angle| angle == 0.0));
    }

    #[test]
    fn truncates_to_eight_bits() {
        let pixels = Tensor::new(vec![3, 1, 2], vec![0.0, 1.0, 0.5, 0.999, -0.5, 1.5]);
        let picture = to_picture(&pixels);
        assert_eq!((picture.width, picture.height), (2, 1));
        assert_eq!(picture.pixels, vec![0, 127, 0, 255, 254, 255]);
    }
}
