#!/bin/sh
# Renders every scene, then a live run against the mock server, into screenshots/.
# Run from the app's folder after `cargo build --release`.
set -eu
app=target/release/mmh3-studio
root=target/screenshot-dirs
mkdir -p screenshots
for scene in empty disconnected running gallery failed custom-size offline-selected confirm-delete; do
  for theme in light dark; do
    $app --snap "screenshots/scene-$scene-$theme.png" --scene "$scene" --size 1320x860 --scale 2 --theme "$theme" --at 1.0 >/dev/null
  done
done
$app --tree screenshots/scene-gallery-tree.json --scene gallery --at 1.0 >/dev/null
for scene in disconnected running failed offline-selected; do
  $app --snap "screenshots/small-$scene.png" --scene "$scene" --size 1040x680 --scale 2 --at 1.0 >/dev/null
done

live() {
  name=$1
  shift
  target/release/mock-server --listen 127.0.0.1:8899 >/dev/null &
  mock=$!
  sleep 0.4
  RUNE_DIRS_ROOT="$root/$name-$$" $app --server http://127.0.0.1:8899 --realtime --script scripts/generate.json --scale 2 "$@" >/dev/null
  kill "$mock"
}
for theme in light dark; do
  live "progress-$theme" --snap "screenshots/live-progress-$theme.png" --tree "screenshots/live-progress-$theme-tree.json" --at mark:sent+2.0 --theme "$theme"
  live "gallery-$theme" --snap "screenshots/live-gallery-$theme.png" --at mark:sent+14.0 --theme "$theme"
  RUNE_DIRS_ROOT="$root/off-$$" $app --server http://127.0.0.1:1 --realtime --snap "screenshots/live-disconnected-$theme.png" --at 2.0 --theme "$theme" --scale 2 >/dev/null
done
