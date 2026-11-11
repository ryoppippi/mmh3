# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2", "pillow>=11"]
# ///
"""Fits the linear latent-to-RGB map of mmh3's step previews.

Development tool. It reads pairs of a final latent, written by `mmh3 image --save-latent`, and
the picture of the same run, fits the 24 x 3 weights and the bias that map a latent pixel to the
mean of the 16 x 16 pixels of the picture it decodes to by least squares, and prints them in the
form of PICTURE_PREVIEW in crates/mmh3-core/src/preview.rs. Each NAME stands for NAME.safetensors
and NAME.png in DIRECTORY. The pairs after --held-out are left out of the fit, and the PSNR of
the map on each pair says how well it generalises; NAME.preview.png and NAME.target.png show the
map's picture and the decode's averages of each held-out pair side by side at twice the latent's
size.
"""

import argparse
import json
import struct

import numpy as np
from PIL import Image


def load_latent(path):
    """The latent [channels, height, width] of the one frame a picture has."""
    with open(path, "rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(length))
        data = file.read()
    info = header["video"]
    start, end = info["data_offsets"]
    return np.frombuffer(data[start:end], dtype=np.float32).reshape(info["shape"])[:, 0]


def averaged(path, height, width):
    """The picture of `path` in [0, 1], averaged over the pixels of each latent pixel."""
    picture = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    rows, columns = picture.shape[0] // height, picture.shape[1] // width
    picture = picture[: height * rows, : width * columns]
    return picture.reshape(height, rows, width, columns, 3).mean(axis=(1, 3))


def pair(directory, name):
    latent = load_latent(f"{directory}/{name}.safetensors")
    channels, height, width = latent.shape
    features = latent.reshape(channels, -1).T.astype(np.float64)
    features = np.hstack([features, np.ones((len(features), 1))])
    target = averaged(f"{directory}/{name}.png", height, width).reshape(-1, 3)
    return features, target, (height, width)


def psnr(mapped, target):
    error = np.mean((np.clip(mapped, 0, 1) - target) ** 2)
    return 10 * np.log10(1.0 / error)


def save(path, pixels, height, width):
    pixels = (np.clip(pixels.reshape(height, width, 3), 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(pixels).resize((width * 2, height * 2), Image.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--held-out", nargs="*", default=[])
    arguments = parser.parse_args()

    fitted = [pair(arguments.directory, name) for name in arguments.names]
    features = np.concatenate([features for features, _, _ in fitted])
    target = np.concatenate([target for _, target, _ in fitted])
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    for name, (features, target, _) in zip(arguments.names, fitted):
        print(f"fitted   {name}: {psnr(features @ coefficients, target):.2f} dB")
    for name in arguments.held_out:
        features, target, (height, width) = pair(arguments.directory, name)
        mapped = features @ coefficients
        print(f"held out {name}: {psnr(mapped, target):.2f} dB")
        save(f"{arguments.directory}/{name}.preview.png", mapped, height, width)
        save(f"{arguments.directory}/{name}.target.png", target, height, width)

    print("weights: [")
    for row in coefficients[:-1].T:
        print("    [" + ", ".join(f"{value:.6f}" for value in row) + "],")
    print("],")
    print("bias: [" + ", ".join(f"{value:.6f}" for value in coefficients[-1]) + "],")


if __name__ == "__main__":
    main()
