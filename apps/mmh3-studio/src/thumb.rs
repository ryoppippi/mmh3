//! Gallery thumbnails, decoded a row at a time and averaged into place, so a 16 MP PNG never
//! sits in memory as full-size RGBA.

use std::io::Cursor;

use png::{ColorType, Transformations};

/// The longest side of a gallery thumbnail, in pixels.
pub const SIDE: u32 = 256;

/// `width` × `height` scaled to fit in a `side` square, never enlarged.
pub fn fit(width: u32, height: u32, side: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= side {
        return (width.max(1), height.max(1));
    }
    let scale = |v: u32| ((v as u64 * side as u64 + longest as u64 / 2) / longest as u64).max(1) as u32;
    (scale(width), scale(height))
}

pub fn thumbnail(png: &[u8]) -> Result<image::RgbaImage, String> {
    let mut decoder = png::Decoder::new(Cursor::new(png));
    decoder.set_transformations(Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let (width, height) = (reader.info().width, reader.info().height);
    let channels = match reader.output_color_type().0 {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => return whole(png),
    };
    if reader.info().interlaced {
        return whole(png);
    }
    let (tw, th) = fit(width, height, SIDE);
    let column: Vec<usize> = (0..width as u64).map(|x| (x * tw as u64 / width as u64) as usize).collect();
    let mut sums = vec![0u32; tw as usize * th as usize * 4];
    let mut counts = vec![0u32; tw as usize * th as usize];
    let mut y = 0u64;
    while let Some(row) = reader.next_row().map_err(|e| e.to_string())? {
        let base = (y * th as u64 / height as u64) as usize * tw as usize;
        for (x, px) in row.data().chunks_exact(channels).enumerate() {
            let rgba = match px {
                [l] => [*l, *l, *l, 255],
                [l, a] => [*l, *l, *l, *a],
                [r, g, b] => [*r, *g, *b, 255],
                [r, g, b, a] => [*r, *g, *b, *a],
                _ => unreachable!(),
            };
            let at = base + column[x];
            for (sum, v) in sums[at * 4..at * 4 + 4].iter_mut().zip(rgba) {
                *sum += v as u32;
            }
            counts[at] += 1;
        }
        y += 1;
    }
    let pixels = sums.as_chunks::<4>().0.iter().zip(&counts).flat_map(|(sum, &n)| sum.iter().map(move |s| ((s + n / 2) / n.max(1)) as u8)).collect();
    image::RgbaImage::from_raw(tw, th, pixels).ok_or_else(|| "The thumbnail's size didn't add up.".to_owned())
}

/// Decodes the whole image first; for the rare PNG that can't be read a row at a time.
fn whole(png: &[u8]) -> Result<image::RgbaImage, String> {
    let full = image::load_from_memory_with_format(png, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(full.thumbnail(SIDE, SIDE).to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_the_aspect_and_never_enlarges() {
        assert_eq!(fit(5472, 3072, 256), (256, 144));
        assert_eq!(fit(3072, 5472, 256), (144, 256));
        assert_eq!(fit(100, 40, 256), (100, 40));
        assert_eq!(fit(8192, 16, 256), (256, 1));
    }

    #[test]
    fn a_row_by_row_thumbnail_matches_a_whole_decode() {
        let png = crate::mock::paint(1024, 576, 7);
        let rows = thumbnail(&png).unwrap();
        let full = whole(&png).unwrap();
        assert_eq!(rows.dimensions(), (256, 144));
        assert_eq!(rows.dimensions(), full.dimensions());
        let diff: u64 = rows.as_raw().iter().zip(full.as_raw()).map(|(a, b)| a.abs_diff(*b) as u64).sum();
        let mean = diff as f64 / rows.as_raw().len() as f64;
        assert!(mean < 4.0, "mean difference {mean}");
    }

    #[test]
    fn a_broken_png_is_an_error() {
        assert!(thumbnail(b"not a png").is_err());
    }
}
