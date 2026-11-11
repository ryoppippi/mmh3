# Pictures

[MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) generates pictures from a
prompt. Its DiT is MiniMax H3's without the audio stream, distilled to six steps with one output
head for each step, and a 2D ViT decodes its latent. `mmh3 image` runs it on [Metal](metal.md) and
writes a PNG.

## Download

`tools/download-models.sh --image`, or `make download-image-models`, downloads these files from
Comfy-Org/MiniMax-H3-Image instead of the video ones:

| File | |
| --- | --- |
| `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` | the INT8 ConvRot DiT |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | the text encoder of the video models |
| `vae/minimax_h3_image_vae_fp16.safetensors` | the FP16 image VAE |

`--dit`, `--text-encoder` and `--image-vae` take single files in their place.

## Generate

```sh
target/release/mmh3 image --models models --prompt-file prompt.txt --out out.png --seed 1
```

`make generate-image` runs this with a sample prompt and writes `out.png`, and `IMAGE_OPTIONS`
adds options to it.

- `--width` and `--height` give the size of the PNG, 4096 × 4096 by default. The model was trained
  at about 4 to 20 megapixels, and 16 megapixels suits it best, such as 4096 × 4096, 4736 × 3552,
  5472 × 3072 or 6272 × 2688.
- A picture smaller than `--min-generation-area` pixels, 4194304 (2048 × 2048) by default, is
  generated on a larger canvas of the same aspect ratio and resized back with Lanczos, as the
  reference pipeline does. It takes 1024 to 16777216, and 1024 generates every picture at its own
  size. Either way, each side of the canvas is rounded to a multiple of 32.
- `--seed` picks the noise.
- `--attention-precision` and `--linear-precision` take what they take for [Metal](metal.md)'s video
  generation.

The prompt is encoded as it is for a video, without a template, and only its first 4096 tokens are
kept. The model expects a detailed description, and a short prompt gives a plainer picture.

The steps follow the sigma grid in the DiT's metadata, with no shift and no guidance, and the
attention is dense. The options of `mmh3 generate` that change the steps, such as `--steps`,
`--lora`, `--patch`, `--attention` and `--worker`, are refused. The decoder runs on 256-pixel
tiles that overlap by at least 64 pixels, as the reference does, and the 8-bit picture is
truncated from its values as the reference converts it.

mmh3 prints the time of the prompt, each step and the decode.
