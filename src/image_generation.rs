//! Text-to-image generation with MiniMax-H3-Image, written to a PNG file.
//!
//! The image checkpoint is the H3 DiT without its audio stream, with one output head per step of
//! a fixed six-step grid. The prompt encodes as it does for a video, and a 2D ViT decodes the one
//! latent frame.

use crate::cli::{option_number, parse_options};
use mmh3_core::dit::config::DitConfig;
use mmh3_core::image_generation::{MIN_GENERATION_AREA, PDD_SIGMAS, canvas_size};
use std::collections::HashMap;
use std::error::Error;

/// The options of `mmh3 image`.
pub const OPTIONS: &[&str] = &[
    "prompt",
    "prompt-file",
    "out",
    "models",
    "width",
    "height",
    "min-generation-area",
    "seed",
    "dit",
    "text-encoder",
    "image-vae",
    "attention-precision",
    "linear-precision",
];

pub const IMAGE_DIT_FILE: &str =
    "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors";
pub const IMAGE_VAE_FILE: &str = "vae/minimax_h3_image_vae_fp16.safetensors";

/// The side of a picture when `--width` or `--height` is left out.
const DEFAULT_SIDE: usize = 4096;

/// What one picture is generated as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSettings {
    /// The size of the PNG.
    pub width: usize,
    pub height: usize,
    /// The size the DiT and the decoder work at, which a resize takes to the PNG's.
    pub canvas_width: usize,
    pub canvas_height: usize,
    pub seed: u64,
}

impl ImageSettings {
    pub fn parse(options: &HashMap<&str, &str>) -> Result<Self, Box<dyn Error>> {
        let width = option_number(options, "width", DEFAULT_SIDE)?;
        let height = option_number(options, "height", DEFAULT_SIDE)?;
        let area = option_number(options, "min-generation-area", MIN_GENERATION_AREA)?;
        let (canvas_width, canvas_height) = canvas_size(width, height, area)
            .map_err(|error| format!("--width, --height and --min-generation-area: {error}"))?;
        Ok(Self {
            width,
            height,
            canvas_width,
            canvas_height,
            seed: option_number(options, "seed", 0)? as u64,
        })
    }
}

/// Refuses the options of a video generation by name, so that one that would change a video is not
/// taken for a usage error, nor silently dropped.
pub fn refuse_video_options(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let video = crate::generation::OPTIONS
        .iter()
        .chain(crate::generation::FLAGS)
        .chain(&["ffmpeg"]);
    for argument in arguments {
        if let Some(name) = argument.strip_prefix("--")
            && !OPTIONS.contains(&name)
            && video.clone().any(|option| *option == name)
        {
            return Err(format!(
                "--{name} is an option of `mmh3 generate`. `mmh3 image` runs the checkpoint's own \
                 steps with dense attention on this machine, with no LoRA, patch or worker"
            )
            .into());
        }
    }
    Ok(())
}

/// The sigma grid of an image checkpoint's heads: its metadata's, or the release's when it names
/// none. A video checkpoint is refused.
pub fn image_grid(config: &DitConfig) -> Result<Vec<f64>, String> {
    if config.has_audio() || config.pdd_steps < 2 {
        return Err(
            "it is not a MiniMax-H3-Image DiT, which has no audio stream and a head per step"
                .to_owned(),
        );
    }
    let grid = config
        .pdd_sigmas
        .clone()
        .unwrap_or_else(|| PDD_SIGMAS.to_vec());
    if grid.len() != config.pdd_steps + 1 {
        return Err(format!(
            "its {} heads name no grid in its metadata, and the release's has {} steps",
            config.pdd_steps,
            PDD_SIGMAS.len() - 1
        ));
    }
    Ok(grid)
}

/// The prompt of `--prompt` or `--prompt-file`.
#[cfg(feature = "metal")]
fn prompt(options: &HashMap<&str, &str>) -> Result<String, Box<dyn Error>> {
    let prompt = match (options.get("prompt"), options.get("prompt-file")) {
        (Some(prompt), None) => (*prompt).to_owned(),
        (None, Some(path)) => std::fs::read_to_string(path)?,
        _ => return Err("pass either --prompt or --prompt-file".into()),
    };
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    Ok(prompt)
}

/// Generates one picture from a prompt and writes it as a PNG.
#[cfg(feature = "metal")]
pub fn run(arguments: &[String], usage: &'static str) -> Result<(), Box<dyn Error>> {
    use crate::models::{TEXT_ENCODER_FILE, load_dit, option_path};
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::{Schedule, euler_step};
    use mmh3_core::image_generation::{MAX_PROMPT_TOKENS, latent_shape};
    use mmh3_core::random::NormalSampler;
    use mmh3_core::safetensors::SafeTensors;
    use mmh3_core::tensor::Tensor;
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_metal::image_vae::MetalImageDecoder;
    use mmh3_metal::text_encoder::MetalTextEncoder;
    use std::path::Path;
    use std::time::Instant;

    refuse_video_options(arguments)?;
    let options = parse_options(arguments, OPTIONS, &[], usage)?;
    let settings = ImageSettings::parse(&options)?;
    let out = Path::new(options.get("out").ok_or(usage)?);
    let output = crate::output::PngOutput::prepare(out)?;
    crate::metal::validate_step_options(&options)?;
    let prompt = prompt(&options)?;
    // The DiT's header says whether it is an image checkpoint before minutes go into the prompt.
    let dit_path = option_path(&options, "dit", IMAGE_DIT_FILE)?;
    let file = SafeTensors::open(Path::new(&dit_path))
        .map_err(|error| format!("reading {dit_path}: {error}"))?;
    let config =
        DitConfig::of(&file, "").map_err(|error| format!("reading {dit_path}: {error}"))?;
    drop(file);
    let schedule =
        Schedule::pdd(&image_grid(&config).map_err(|error| format!("{dit_path}: {error}"))?);
    let vae_path = option_path(&options, "image-vae", IMAGE_VAE_FILE)?;

    let started = Instant::now();
    let mut ids = Tokenizer::h3().encode(&prompt);
    let tokens = ids.len();
    ids.truncate(MAX_PROMPT_TOKENS);
    let context = {
        let path = option_path(&options, "text-encoder", TEXT_ENCODER_FILE)?;
        let mut encoder = MetalTextEncoder::load_fitting(&SafeTensors::open(Path::new(&path))?)?;
        encoder.set_precision(
            crate::metal::default_linear_precision(),
            crate::metal::attention_precision(&options)?,
        )?;
        encoder.read_once();
        encoder.encode(&ids, &[])?.context
    };
    let truncated = match tokens > ids.len() {
        true => format!(" (truncated from {tokens})"),
        false => String::new(),
    };
    println!(
        "encoded {} prompt tokens{truncated} in {:.1} s",
        ids.len(),
        started.elapsed().as_secs_f64()
    );

    let dit = load_dit(&options, "dit", IMAGE_DIT_FILE)?;
    let shape = latent_shape(settings.canvas_width, settings.canvas_height);
    let mut latent = Tensor::new(
        shape.clone(),
        NormalSampler::new(settings.seed).samples(shape.iter().product()),
    );
    let steps = schedule.steps();
    println!(
        "{}×{} on a {}×{} canvas, {} text tokens, {steps} steps, seed {}",
        settings.width,
        settings.height,
        settings.canvas_width,
        settings.canvas_height,
        context.shape[0],
        settings.seed
    );
    let started = Instant::now();
    let prepared = dit.prepare_text(&context)?;
    println!(
        "refined the prompt in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    for step in 0..steps {
        let started = Instant::now();
        let inputs = DitInputs {
            video: latent.clone(),
            audio: DitInputs::no_audio(),
            context: context.clone(),
            context_modalities: Vec::new(),
            keyframes: Vec::new(),
            references: Vec::new(),
            sigma: schedule.video[step],
            shift_video: 1.0,
            shift_audio: 1.0,
            pdd_step: step,
        };
        let outputs = prepared.forward(&inputs, &[], None)?;
        euler_step(
            &mut latent.data,
            &outputs.video,
            schedule.video[step],
            schedule.video[step + 1],
        );
        println!(
            "step {}/{steps} at sigma {:.4} in {:.1} s",
            step + 1,
            schedule.video[step],
            started.elapsed().as_secs_f64()
        );
    }
    drop(prepared);
    drop(dit);

    let started = Instant::now();
    let mut decoder = MetalImageDecoder::load(Path::new(&vae_path))?;
    decoder.set_precision(
        crate::metal::default_linear_precision(),
        crate::metal::attention_precision(&options)?,
    )?;
    let mut picture = decoder.decode_picture(&latent)?;
    drop(decoder);
    println!(
        "decoded the picture with {vae_path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    if (picture.width, picture.height) != (settings.width, settings.height) {
        let started = Instant::now();
        picture = picture.resize(settings.width, settings.height);
        println!(
            "resized it to {}×{} in {:.1} s",
            settings.width,
            settings.height,
            started.elapsed().as_secs_f64()
        );
    }
    let started = Instant::now();
    output.write(&picture)?;
    println!(
        "wrote {} in {:.1} s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(not(feature = "metal"))]
pub fn run(arguments: &[String], usage: &'static str) -> Result<(), Box<dyn Error>> {
    refuse_video_options(arguments)?;
    parse_options(arguments, OPTIONS, &[], usage)?;
    Err("mmh3 image runs on Metal: rebuild with --features metal".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn settings(values: &[&str]) -> Result<ImageSettings, Box<dyn Error>> {
        let arguments = arguments(values);
        ImageSettings::parse(&parse_options(&arguments, OPTIONS, &[], "usage")?)
    }

    #[test]
    fn defaults_to_a_square_of_4096() {
        let settings = settings(&["--prompt", "a cat", "--out", "cat.png"]).unwrap();
        assert_eq!(
            settings,
            ImageSettings {
                width: 4096,
                height: 4096,
                canvas_width: 4096,
                canvas_height: 4096,
                seed: 0,
            }
        );
    }

    #[test]
    fn generates_a_small_picture_larger_unless_told_not_to() {
        let scaled = settings(&["--width", "1000", "--height", "1000", "--seed", "7"]).unwrap();
        assert_eq!((scaled.width, scaled.height), (1000, 1000));
        assert_eq!((scaled.canvas_width, scaled.canvas_height), (2048, 2048));
        assert_eq!(scaled.seed, 7);
        let exact = settings(&[
            "--width",
            "1040",
            "--height",
            "1072",
            "--min-generation-area",
            "1024",
        ])
        .unwrap();
        assert_eq!((exact.canvas_width, exact.canvas_height), (1024, 1088));
    }

    #[test]
    fn refuses_an_area_out_of_range_and_an_empty_side() {
        for values in [
            &["--min-generation-area", "1023"][..],
            &["--min-generation-area", "16777217"][..],
            &["--width", "0"][..],
            &["--height", "-1"][..],
        ] {
            assert!(settings(values).is_err(), "{values:?}");
        }
    }

    #[test]
    fn names_the_video_options_it_refuses() {
        for option in [
            "--steps",
            "--lora",
            "--attention",
            "--worker",
            "--frames",
            "--ffmpeg",
        ] {
            let error = refuse_video_options(&arguments(&["--prompt", "a", option, "4"]))
                .unwrap_err()
                .to_string();
            assert!(
                error.starts_with(&format!("{option} is an option of")),
                "{error}"
            );
        }
        assert!(
            refuse_video_options(&arguments(&["--prompt", "a", "--seed", "3", "--dit", "x"]))
                .is_ok()
        );
    }

    #[test]
    fn takes_the_grid_of_an_image_checkpoint_only() {
        let shapes = |audio: bool, heads: usize| {
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
        };
        let image = DitConfig::from_shapes(shapes(false, 6)).unwrap();
        assert_eq!(image_grid(&image).unwrap(), PDD_SIGMAS);
        let named = DitConfig {
            pdd_sigmas: Some(vec![1.0, 0.9, 0.8, 0.5, 0.3, 0.1, 0.0]),
            ..image.clone()
        };
        assert_eq!(image_grid(&named).unwrap()[1], 0.9);
        let video = DitConfig::from_shapes(shapes(true, 1)).unwrap();
        assert!(image_grid(&video).is_err());
        let other = DitConfig::from_shapes(shapes(false, 4)).unwrap();
        assert!(image_grid(&other).is_err());
    }
}
