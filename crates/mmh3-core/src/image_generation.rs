//! Shapes of one MiniMax-H3-Image request: the canvas a picture is generated on, its latent, the
//! prompt's length and the sigma grid of the per-step heads, as the reference pipeline sets them.

use crate::generation::CANVAS_MULTIPLE;
use crate::image_vae::LATENT_CHANNELS;
use crate::vae::SPATIAL_RATIO;

/// The smallest area a picture is generated at by default. A smaller request is generated on a
/// larger canvas of the same aspect ratio and resized back after decoding.
pub const MIN_GENERATION_AREA: usize = 2048 * 2048;
/// The values the reference pipeline accepts for the smallest area. The lowest generates every
/// picture at its own size.
pub const MIN_GENERATION_AREA_RANGE: std::ops::RangeInclusive<usize> = 1024..=4096 * 4096;
/// The prompt tokens the reference pipeline keeps.
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// The grid the released checkpoint's heads were distilled on, for a checkpoint whose metadata
/// does not name it.
pub const PDD_SIGMAS: [f64; 7] = [
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
];

/// The `(width, height)` of the canvas a `width × height` picture is generated on: scaled up, never
/// down, to `min_area` at the same aspect ratio, each side rounded half to even to a multiple of 32.
pub fn canvas_size(width: usize, height: usize, min_area: usize) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 {
        return Err(format!(
            "width and height must be positive, got {width}×{height}"
        ));
    }
    if !MIN_GENERATION_AREA_RANGE.contains(&min_area) {
        return Err(format!(
            "the smallest generation area must be within {} and {}, got {min_area}",
            MIN_GENERATION_AREA_RANGE.start(),
            MIN_GENERATION_AREA_RANGE.end()
        ));
    }
    let scale = (min_area as f64 / (width as f64 * height as f64))
        .sqrt()
        .max(1.0);
    let snap = |side: usize| {
        let multiple = CANVAS_MULTIPLE as f64;
        ((side as f64 * scale / multiple).round_ties_even() as usize * CANVAS_MULTIPLE)
            .max(CANVAS_MULTIPLE)
    };
    Ok((snap(width), snap(height)))
}

/// `[channels, 1, height, width]` of the latent of a `width × height` canvas.
pub fn latent_shape(width: usize, height: usize) -> Vec<usize> {
    vec![
        LATENT_CHANNELS,
        1,
        height / SPATIAL_RATIO,
        width / SPATIAL_RATIO,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_small_requests_up_to_the_area() {
        let canvas = |width, height| canvas_size(width, height, MIN_GENERATION_AREA).unwrap();
        assert_eq!(canvas(4096, 4096), (4096, 4096));
        assert_eq!(canvas(1024, 1024), (2048, 2048));
        assert_eq!(canvas(1000, 1000), (2048, 2048));
        assert_eq!(canvas(800, 600), (2368, 1760));
        // Above the area the sides only snap to the grid.
        assert_eq!(canvas(5000, 3000), (4992, 3008));
        assert_eq!(canvas(1, 1), (2048, 2048));
    }

    #[test]
    fn rounds_half_to_even_like_python() {
        // 1024 leaves the size as it is, so 32.5 and 33.5 multiples of 32 meet the tie.
        assert_eq!(canvas_size(1040, 1072, 1024).unwrap(), (1024, 1088));
        assert_eq!(canvas_size(1, 1, 1024).unwrap(), (32, 32));
        assert_eq!(canvas_size(48, 80, 1024).unwrap(), (64, 64));
    }

    #[test]
    fn refuses_what_the_reference_refuses() {
        assert!(canvas_size(0, 1024, MIN_GENERATION_AREA).is_err());
        assert!(canvas_size(1024, 1024, 1023).is_err());
        assert!(canvas_size(1024, 1024, 4096 * 4096 + 1).is_err());
        assert!(canvas_size(1024, 1024, 4096 * 4096).is_ok());
    }

    #[test]
    fn one_latent_frame_per_picture() {
        assert_eq!(latent_shape(4096, 2048), vec![24, 1, 128, 256]);
    }
}
