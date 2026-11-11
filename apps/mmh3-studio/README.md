# mmh3 studio

A native macOS client for a resident `mmh3 server` running MiniMax-H3-Image. The server keeps the text encoder, DiT and VAE loaded, so a generation never pays a cold start. The studio never loads a model itself. It is built with [Rune](https://github.com/anomalyco/rune), pinned to a `dev` revision in `Cargo.toml`.

This folder is a standalone Cargo project with its own `[workspace]` table, so it is not part of the mmh3 workspace. `flake.nix` supplies Rust 1.99.1.

## Running it

Start the server from your mmh3 checkout:

```sh
export MMH3_MODELS=$PWD/models
target/release/mmh3 server
```

Then start the studio:

```sh
nix develop path:$PWD -c cargo run --release
```

The studio connects to `http://127.0.0.1:8833`. To use another address, pass `-- --server http://host:port` or set `MMH3_SERVER`. If the server isn't running, the studio shows the command that starts it and tries again every two seconds.

To try the studio without the engine, run the mock server in place of `mmh3 server`. It implements the same API in image mode: six timed steps, then a painted PNG. A prompt that contains `[fail]` fails.

```sh
nix develop path:$PWD -c cargo run --release --bin mock-server -- --listen 127.0.0.1:8833 --step 0.6
```

## Using it

- **Composer:** write a prompt, then choose a size. The presets are 2048×2048 for a 4 MP draft, and 16 MP finals at 1:1, 4:3, 3:4, 16:9 and 9:16. A custom size takes multiples of 32, from 256 to 8192 pixels on each side. Set a seed or randomise it. ⌘↩ generates; plain Return starts a new line. You can queue as many generations as you like.
- **Gallery** (left pane): shows every image generation on the server, newest first. Each row shows its live step (k of 6) and elapsed time. Right-click a row to save the PNG, reveal it in Finder, copy the prompt or seed, re-run it with a new seed, or delete it. You can also drag a thumbnail out to Finder.
- **Preview:** the selected image at full size, with its size and seed. A timeline under it shows how long each stage took: text encoder, each step, and VAE decode. To zoom about the pointer, click, pinch or use Ctrl+wheel. To pan, drag or scroll. The buttons zoom in, zoom out and fit.
- **Status bar:** shows whether the server is connected, which models are resident, free and total memory, and the queue.

Generated PNGs go into the app's cache folder (`~/Library/Caches/dev.mmh3.studio/images`). Each PNG is downloaded once. Thumbnails are decoded off the UI thread and scaled down to 256 px. Only the selected image is kept at full resolution, and Rune decodes it with mipmaps on its own workers.

## Checks

```sh
nix develop path:$PWD -c cargo test --release      # client, mock, state machine and UI tests
nix develop path:$PWD -c cargo clippy --release --all-targets
nix develop path:$PWD -c sh scripts/screenshots.sh # headless PNGs and element trees in screenshots/
```

`scripts/screenshots.sh` first renders the fixture scenes, which never touch the network:

- `empty`
- `disconnected`
- `running`
- `gallery`
- `failed`
- `custom-size`
- `offline-selected`

It then runs the studio headless with `--realtime` against the mock server. In that run, `scripts/generate.json` queues two 16 MP generations. To render a single scene:

```sh
target/release/mmh3-studio --snap out.png --scene gallery --theme dark --scale 2
```
