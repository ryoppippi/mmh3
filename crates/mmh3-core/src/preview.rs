//! A rough picture of a latent while it is still being sampled, without the VAE: the clean latent
//! a step predicts, mapped to RGB one latent pixel at a time by a linear map fitted against the
//! decoder's pictures.

/// A linear map from a latent pixel's channels to RGB in [0, 1].
pub struct LatentPreview<const CHANNELS: usize> {
    pub weights: [[f32; CHANNELS]; 3],
    pub bias: [f32; 3],
}

/// The map of MiniMax H3 Image's picture VAE, fitted by least squares on the final latents of six
/// prompts at 2048×2048 against their decoded pictures averaged over 16×16 pixels. It came 26.7 and
/// 28.1 dB PSNR from such averages of two prompts it was not fitted on.
pub const PICTURE_PREVIEW: LatentPreview<24> = LatentPreview {
    weights: [
        [
            -0.033347, 0.028027, 0.173781, 0.092236, 0.028378, 0.047712, 0.018741, 0.021266,
            0.007675, 0.140165, -0.012226, -0.036972, 0.012575, -0.006643, 0.007433, -0.006214,
            -0.002580, -0.007988, -0.013892, 0.021311, -0.003767, 0.000055, -0.002153, 0.005634,
        ],
        [
            -0.071705, 0.026284, 0.167905, 0.007312, 0.070206, 0.031153, -0.002299, 0.011970,
            0.005383, 0.053164, -0.010928, -0.012511, 0.007672, -0.000588, 0.001175, -0.000474,
            0.005800, 0.008271, -0.019114, 0.013268, -0.008194, 0.003597, 0.008196, 0.007660,
        ],
        [
            -0.085632, 0.001489, 0.090088, -0.121071, -0.036552, -0.020056, 0.007644, 0.001998,
            0.008202, 0.073241, 0.002864, 0.006870, 0.022907, -0.003415, 0.016089, -0.015725,
            0.000592, 0.006791, -0.016796, 0.020199, 0.006464, -0.003268, 0.010777, 0.002301,
        ],
    ],
    bias: [0.566344, 0.596592, 0.539360],
};

impl<const CHANNELS: usize> LatentPreview<CHANNELS> {
    /// RGB bytes, row after row, of the clean latent `x − sigma · v` that a step from `latent`
    /// `[CHANNELS, 1, height, width]` with velocity `velocity` predicts, each latent pixel drawn
    /// as `scale × scale` pixels.
    pub fn render(
        &self,
        latent: &[f32],
        velocity: &[f32],
        sigma: f32,
        height: usize,
        width: usize,
        scale: usize,
    ) -> Vec<u8> {
        let plane = height * width;
        assert_eq!(latent.len(), CHANNELS * plane, "the latent's size");
        assert_eq!(velocity.len(), latent.len(), "the velocity's size");
        let mut colours = vec![[0u8; 3]; plane];
        for (pixel, colour) in colours.iter_mut().enumerate() {
            let mut rgb = self.bias;
            for channel in 0..CHANNELS {
                let index = channel * plane + pixel;
                let clean = latent[index] - sigma * velocity[index];
                for (value, weights) in rgb.iter_mut().zip(&self.weights) {
                    *value += weights[channel] * clean;
                }
            }
            for (byte, value) in colour.iter_mut().zip(rgb) {
                *byte = (value * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        let mut bytes = Vec::with_capacity(3 * plane * scale * scale);
        for row in colours.chunks(width) {
            for _ in 0..scale {
                for colour in row {
                    for _ in 0..scale {
                        bytes.extend_from_slice(colour);
                    }
                }
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_predicted_clean_latent_and_scales_it_up() {
        let preview = LatentPreview::<2> {
            weights: [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            bias: [0.0, 0.0, 0.1],
        };
        // Two latent pixels, one row: channel 0 then channel 1.
        let latent = [1.0, 0.5, 0.0, 1.0];
        let velocity = [1.0, 0.0, -1.0, 0.0];
        // Clean: channel 0 = [0.5, 0.5], channel 1 = [0.5, 1.0].
        let bytes = preview.render(&latent, &velocity, 0.5, 1, 2, 2);
        let first = [128, 128, 153];
        let second = [128, 255, 217];
        let row: Vec<u8> = [first, first, second, second].concat();
        assert_eq!(bytes, [row.clone(), row].concat());
    }
}
