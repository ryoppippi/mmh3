#![cfg(target_os = "macos")]

//! Runs the Metal image decoder on the tiny golden fixture from tools/golden/image_vae_tiny.py.

use mmh3_core::image_vae::{LATENTS_MEAN, LATENTS_STD};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_metal::image_vae::MetalImageDecoder;
use mmh3_metal::{AttentionPrecision, Device, LinearPrecision};
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/image_vae_tiny.safetensors"
);

fn tensor(file: &SafeTensors, name: &str) -> Tensor {
    let mut tensor = Tensor::load(
        file,
        file.get(name).unwrap_or_else(|| panic!("missing {name}")),
    )
    .unwrap();
    if tensor.shape.len() == 4 {
        tensor.shape.remove(0);
    }
    tensor
}

fn worst_error(actual: &Tensor, expected: &Tensor) -> f32 {
    assert_eq!(actual.shape, expected.shape);
    actual
        .data
        .iter()
        .zip(&expected.data)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        })
}

#[test]
fn carries_the_reference_latent_statistics() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    assert_eq!(
        tensor(&file, "reference.latents_mean").data,
        LATENTS_MEAN.to_vec()
    );
    assert_eq!(
        tensor(&file, "reference.latents_std").data,
        LATENTS_STD.to_vec()
    );
}

#[test]
fn matches_diffusers_golden_decode() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut decoder = MetalImageDecoder::from_file(&file, "weight.").unwrap();
    // FP32 throughout differs from the reference by summation order alone. FP16 products and
    // attention round their inputs to 11 bits, about 5e-4 relative, which the ImageNet scale of
    // about a quarter shrinks in the pixels; on the fixture a pixel moves by under 1e-4.
    let mut precisions = vec![(LinearPrecision::Fp32, AttentionPrecision::Fp32, 1e-5)];
    if Device::shared().unwrap().supports_tensor_ops() {
        precisions.push((LinearPrecision::Fp16, AttentionPrecision::Fp16, 5e-4));
        precisions.push((LinearPrecision::Int8, AttentionPrecision::Fp16, 5e-4));
    }
    for (linear, attention, tolerance) in precisions {
        decoder.set_precision(linear, attention).unwrap();
        for suffix in ["", "_small"] {
            let latent = tensor(&file, &format!("input.latent{suffix}"));
            let expected = tensor(&file, &format!("output.pixels{suffix}"));
            // One batch takes every tile, and batches of four split a row of tiles.
            for tiles in [0, 4] {
                decoder.set_tiles_per_batch(tiles);
                let pixels = decoder.decode(&latent).unwrap();
                let worst = worst_error(&pixels, &expected);
                eprintln!(
                    "{linear:?}/{attention:?} latent{suffix} {:?}, {tiles} tiles a batch: max error {worst:.3e}",
                    latent.shape
                );
                assert!(worst <= tolerance, "max error {worst}");
            }
        }
    }

    let latent = tensor(&file, "input.latent");
    let picture = decoder.decode_picture(&latent).unwrap();
    assert_eq!((picture.width, picture.height), (640, 384));
    let mut framed = latent.clone();
    framed.shape.insert(1, 1);
    assert_eq!(
        decoder.decode(&framed).unwrap(),
        decoder.decode(&latent).unwrap()
    );
    assert!(
        decoder
            .decode(&Tensor::new(vec![23, 2, 2], vec![0.0; 92]))
            .is_err()
    );
}
