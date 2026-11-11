# Server

`mmh3 server` takes a generation over HTTP and keeps its models loaded between the generations it
runs, so a machine asked for one video after another reads the checkpoints once. It also generates
pictures with [Image](image.md), see [Asking for a picture](#asking-for-a-picture).

```sh
export MMH3_MODELS=$PWD/models

target/release/mmh3 server
```

It waits on `127.0.0.1:8833` and writes each generation into a directory of its own under
`$XDG_CACHE_HOME/mmh3/jobs`. `--listen ADDR` and `--jobs DIR` change either.

Each video is MP4, written the way `mmh3 generate --out FILE.mp4` writes it, which on CUDA takes
NVENC. `--format webm` writes WebM instead, from a build with the `webm` feature. A GPU without
NVENC hands every video to ffmpeg with `--ffmpeg`, which comes last and takes the rest of the line
as ffmpeg's arguments, as it does for a generation. See [Output](output.md).

```sh
target/release/mmh3 server --ffmpeg
```

The rest of the server's line is what `mmh3 generate` takes about the machine: `--models`, `--dit`,
`--text-encoder`, `--video-vae`, `--audio-vae`, `--devices`, `--worker`, `--worker-units`,
`--local-worker`, `--token`, `--vram-budget` and `--consistent`. Every generation is told them, and
a request may not give them. What to generate is a request's, and the server refuses it on its own
line.

There is no authentication. A machine that should answer anybody but itself wants something in
front of this that decides who may ask, which is why the default is loopback.

## Asking for a generation

A form field is the option of the same name, so what `mmh3 generate` takes on the command line about
what to generate this takes as a field. A field that carries a file is written beside the generation
and stands for the path it was written to. `first-frame`, `last-frame`, the references,
`prompt-file` and `context` take only such a file. `lora`, `patch` and `latent-upscaler` take the
name of a file in the models directory, looked for where `mmh3 generate` looks for it.

```sh
curl -X POST http://localhost:8833/v1/generations \
  -F prompt="A red panda sips tea on a sunny wooden porch while birds chirp in the garden." \
  -F steps=4 -F seed=2
```

```json
{
  "id": "6ab1cf3b0000",
  "kind": "video",
  "state": "queued",
  "queued": 1790037819,
  "seconds": null,
  "step": null,
  "steps": null,
  "error": null,
  "request": {
    "prompt": "A red panda sips tea on a sunny wooden porch while birds chirp in the garden.",
    "width": null,
    "height": null,
    "seed": 2,
    "min_generation_area": null,
    "attention_precision": null,
    "attention": null,
    "sparse_tau": null,
    "sparse_start": null,
    "draft_steps": null,
    "latent_upscaler": null,
    "references": []
  },
  "created_at": 1790037819.52,
  "started_at": null,
  "finished_at": null,
  "timings": null,
  "preview_step": null
}
```

`state` is `queued`, `running`, `done`, `failed` or `cancelled`, `seconds` is how long it took once
it has ended, and `error` says why it is `failed`. `kind` is `video` or `image`. `request` is what
the request asked for: the prompt, from `prompt-file` when it came as a file, the names of its
references and of its latent upscaler, and the attention, its precision, the draft steps and
Sol-Attn's `sparse_tau` and `sparse_start` it asked for, with a picture's defaults where the request
left an option out and `null` where a video's are its own or a setting has the backend's default. `created_at`, `started_at` and `finished_at` are Unix times in seconds.
`timings` and `preview_step` are a picture's, see [Asking for a picture](#asking-for-a-picture).

The answer is the generation rather than the video, which is not there yet. There is one generation
at a time, which takes every GPU the machine has, and the rest wait their turn.

Reference pictures, sounds and clips go the same way, as repeated fields in the order the prompt
refers to them:

```sh
curl -X POST http://localhost:8833/v1/generations \
  -F prompt="<Picture 1> walks past <Picture 2>." \
  -F reference=@cat.png -F reference=@street.jpg
```

A field that is not an option of a generation or that is the machine's is refused by name, and so
is a file longer than the server takes. See [Usage](usage.md) for what the options mean.

## Asking for a picture

`POST /v1/images` takes a picture to generate with [Image](image.md) the same way. Its fields are the
options of `mmh3 image` about what to generate: `prompt`, `prompt-file`, `reference`, `width`,
`height`, `min-generation-area`, `seed`, `attention-precision`, `attention`, `sparse-tau`,
`sparse-start`, `draft-steps`, `latent-upscaler`, `renoise-sigma`, `steps`, `shift`, `patch`,
`lora`, `lora-strength` and `lora-mode`. `latent-upscaler`, `patch` and `lora` take the name of a
file in the models directory as a video's do. A field that starts with `<`, as a prompt that refers
to `<Picture 1>` does, goes to curl with `--form-string`, since curl reads `-F name=<file` from a
file.

```sh
curl -X POST http://localhost:8833/v1/images \
  --form-string prompt="<Picture 1> at night, the street lamps lit." \
  -F reference=@street.jpg -F width=4096 -F height=4096
```

The answer is a generation like any other, which waits its turn with the videos and is watched the
same way, and its picture is a PNG at `GET /v1/generations/{id}/image`.

While it runs, a picture says how long each stage took in `timings`, in seconds, and fills in
`attention_precision` with the precision its DiT runs at:

```json
{
  "load_text": 1.07,
  "encode": 0.29,
  "load_dit": 7.48,
  "steps": [4.22, 2.83, 3.00, 2.97, 3.18, 3.15],
  "decode_read": 0.0004,
  "decode": 1.18,
  "write": 0.03
}
```

`load_text` and `encode` are the text encoder's read and the encode of the prompt and the
references, both `null` when the server kept the encoded prompt from an earlier picture.
`decode_read` is how long the decode waited for the picture VAE, which is read during the steps.
A stage not reached yet is `null`.

After each step, a picture writes a preview of what that step predicts, mapped from its latent
without the VAE at an eighth of the canvas, such as 256×256 for 2048×2048. See [Image](image.md#previews).
`preview_step` is the last step with one, and `GET /v1/generations/{id}/preview` is its PNG, 404
until the first step. `GET /v1/generations/{id}/preview/{step}` is the preview of a given step,
counted from 1.

A picture is generated on this machine's card, whatever `--worker` says, and keeps its text encoder,
DiT and picture VAE in the same tables as the worker of the server. The text encoder is the one
MiniMax H3 reads, so a picture after a video, or a video after a picture, finds it loaded and reads
only the other DiT and VAE. The server takes the DiT and the picture VAE from the models directory,
or from `--image-dit FILE` and `--image-vae FILE`, which are a picture's `--dit` and `--vae`. A kept
DiT runs INT8 linear layers, so a request does not give `linear-precision`.

The server also keeps the text encoder's states of the last prompt it encoded for a picture, so a
picture of the same prompt and references through the same encoder skips the encoder's load and
the encode, which on an M5 Max is about 6 s at 87 tokens and more at longer prompts. The kept
encoder and the vision tower that reads the references run at the default attention precision,
whatever `attention-precision` a request gives its DiT, so the states are the same for any request.

On a DGX Spark, two 1024×1024 pictures against a server just started took 16.2 s and then 9.5 s.
Two 4096×4096 pictures with the [faster settings](image.md#faster) took 34.0 s and then 26.1 s:

```sh
curl -X POST http://localhost:8833/v1/images \
  -F prompt="A red panda in a knitted scarf sits on a mossy log in a misty pine forest at dawn." \
  -F width=4096 -F height=4096 -F draft-steps=2 \
  -F latent-upscaler=minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  -F attention=sol -F attention-precision=int8-fp8
```

## Watching one

```sh
curl -N http://localhost:8833/v1/generations/6ab1cf3b0000/events
```

One event a step, then one for the end, and then the stream closes.

```
data: {"error":null,"id":"6ab1cf3b0000","queued":1790037819,"seconds":null,"state":"running","step":3,"steps":4}

data: {"error":null,"id":"6ab1cf3b0000","queued":1790037819,"seconds":10.9,"state":"done","step":null,"steps":null}
```

Asking again works as well, at `GET /v1/generations/{id}`, and says the same thing.

## Watching every one

```sh
curl -N http://localhost:8833/v1/events
```

One stream for every generation of the server. It starts with a `snapshot`, which is what
`GET /v1/generations` answers, and then says a `generation` with the whole generation whenever one
changes state or step, and a `preview` whenever a picture has a new preview:

```
event: snapshot
data: {"generations":[...]}

event: generation
data: {"id":"6af4fadf0000","kind":"image","state":"running","step":2,"steps":6,...}

event: preview
data: {"id":"6af4fadf0000","step":2,"steps":6,"url":"/v1/generations/6af4fadf0000/preview/2"}

: keep-alive
```

A comment every 15 s keeps the stream alive while nothing happens. A client that falls too far
behind is sent a new `snapshot` in place of what it missed. The stream does not end.

## Cancelling one

```sh
curl -X POST http://localhost:8833/v1/generations/6ab1cf3b0000/cancel
```

This answers 202 and the generation. One waiting its turn is `cancelled` at once and never runs.
One running stops at the boundary of its next step, or before the model it is reading next, and is
`cancelled` once it has, which for a picture is within one step. A generation that has ended
answers 409. A cancelled picture keeps the previews of the steps it took.

## Taking the video

```sh
curl http://localhost:8833/v1/generations/6ab1cf3b0000/video -o out.mp4
```

The video is `video/webm` rather than `video/mp4` from a server started with `--format webm`. Until
the generation is done this answers 409 rather than a video.

## What a generation leaves behind

Each generation has a directory of its own under the jobs directory, named by its id:

- `image.png` or `video.mp4`, what it wrote.
- `previews/step-N.png`, a picture's previews.
- `references/`, the references the request uploaded, numbered when two share a name, and beside
  them any other file the request uploaded.
- `metadata.json`, the generation as `GET /v1/generations/{id}` describes it, with what made it:
  `models`, the checkpoints it read as `{file, size, digest}`, `mmh3`, the `version` and the
  `commit` of the build, and `machine`, the Mac's `model` and `chip`, its `memory_bytes`, its
  `power_mode` (`automatic`, `low` or `high`) and its `adapter_watts`, `null` on battery.
  `gpu_clock_mhz_median` is `null`, since only `powermetrics` reads the GPU's clock, as root.

A server writes `metadata.json` as a generation starts and as it ends, and reads every one under the
jobs directory back when it starts, so the generations of the server before it are listed again,
the most recent first, and their pictures and previews are there to be taken. One that was waiting
or running when that server stopped is `failed`, since it never ended.

## What a server holds

```sh
curl http://localhost:8833/v1/status
```

```json
{
  "generating": "6ab1cf3b0000",
  "generations": 7,
  "memory": {"free": 57481990144, "total": 130660909056},
  "models": ["text_encoder", "video_vae", "image_vae", "audio_vae", "dit"],
  "queued": 1
}
```

`models` is `null` rather than a list while a generation has them, since a machine that is busy is
not a machine holding nothing. `memory` is what the GPUs have between them, and `models` is what
any of them holds.

The models are held by a worker in the server's own process, which is the same worker `mmh3 worker`
runs and holds its models the same way. A generation is a leader and reads no model itself, so
without one there would be nothing to keep: the server lends its cards through that worker even when
a run could have taken every step on one card by itself. The exception is the latent upscaler of
[draft steps](draft-steps.md), which goes into the same tables on the card the generation computes
on and stays there with the others. A server told to use machines of its own with `--worker` lends
none of its cards unless `--local-worker` says so, as a run does. So `--vram-budget GB` and
`--idle-unload SECONDS` mean here what they mean there. See [Distributed
generation](distributed.md). `--consistent` given to the server holds for every generation it runs.

On a DGX Spark, reading the models is what the first generation pays and the ones after it do not,
whatever they generate. Three generations of the same request against a server just started:

| Request | First | Then |
| --- | ---: | ---: |
| 448×256, one step | 12.3 s | 5.3 s, 5.0 s |
| 1344×768, four steps, [FastH3](fasth3.md) | 88.3 s | 77.0 s, 80.5 s |

So it is the short generation that gains most: reading the models is more than half of the first
one and less than a tenth of the second. The rest of the second takes as long however the models
arrived:

| Stage of the 1344×768 FastH3 generation | Time |
| --- | ---: |
| Reading the models, the first time only | 6.4 s |
| The four steps | 58 s |
| The video VAE | 19 to 22 s |

## Every request

| | |
| --- | --- |
| `POST /v1/generations` | Take a generation, as `multipart/form-data`. Answers 202 and the generation. |
| `GET /v1/generations` | Every generation this server was asked for, the most recent first. |
| `GET /v1/generations/{id}` | What one is doing, with the step it is on while it runs. |
| `GET /v1/generations/{id}/events` | The same as it happens, as server-sent events. |
| `GET /v1/events` | Every generation as it happens, with its previews, as server-sent events. |
| `POST /v1/generations/{id}/cancel` | Stop one, at its next step if it is running. Answers 202 and the generation. |
| `POST /v1/images` | Take a picture, as `multipart/form-data`. Answers 202 and the generation. |
| `GET /v1/generations/{id}/video` | The video, once there is one. |
| `GET /v1/generations/{id}/image` | The picture, once there is one. |
| `GET /v1/generations/{id}/preview` | A picture's latest preview, once there is one. |
| `GET /v1/generations/{id}/preview/{step}` | A picture's preview after step `step`. |
| `DELETE /v1/generations/{id}` | Take one away with whatever it wrote. |
| `GET /v1/status` | What this machine is doing and what it is holding. |

A generation that is running is not taken away: `DELETE` answers 409 and leaves it. Cancel it
first.

## Building it

The server is the `server` Cargo feature, which `make` includes. A build that leaves it out has no
`mmh3 server` and none of the crates behind it. See [Build](build.md).
