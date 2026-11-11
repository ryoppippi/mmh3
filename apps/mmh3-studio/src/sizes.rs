//! Image sizes the studio offers, and the rule a custom size must follow.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preset {
    pub width: u32,
    pub height: u32,
    pub tier: &'static str,
    pub aspect: &'static str,
}

pub const PRESETS: [Preset; 6] = [
    Preset { width: 2048, height: 2048, tier: "Draft", aspect: "1:1" },
    Preset { width: 4096, height: 4096, tier: "Final", aspect: "1:1" },
    Preset { width: 4736, height: 3552, tier: "Final", aspect: "4:3" },
    Preset { width: 3552, height: 4736, tier: "Final", aspect: "3:4" },
    Preset { width: 5472, height: 3072, tier: "Final", aspect: "16:9" },
    Preset { width: 3072, height: 5472, tier: "Final", aspect: "9:16" },
];

pub const MULTIPLE: u32 = 32;
pub const MIN_SIDE: u32 = 256;
/// The largest side the preview can decode and draw.
pub const MAX_SIDE: u32 = 8192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SizeError {
    NotANumber,
    NotMultiple,
    OutOfRange,
}

impl std::fmt::Display for SizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizeError::NotANumber => write!(f, "Width and height must be whole numbers."),
            SizeError::NotMultiple => write!(f, "Width and height must be multiples of {MULTIPLE}."),
            SizeError::OutOfRange => write!(f, "Each side must be between {MIN_SIDE} and {MAX_SIDE} pixels."),
        }
    }
}

pub fn side(text: &str) -> Result<u32, SizeError> {
    let n: u32 = text.trim().parse().map_err(|_| SizeError::NotANumber)?;
    if !(MIN_SIDE..=MAX_SIDE).contains(&n) {
        return Err(SizeError::OutOfRange);
    }
    if !n.is_multiple_of(MULTIPLE) {
        return Err(SizeError::NotMultiple);
    }
    Ok(n)
}

pub fn custom(width: &str, height: &str) -> Result<(u32, u32), SizeError> {
    Ok((side(width)?, side(height)?))
}

pub fn megapixels(width: u32, height: u32) -> f64 {
    f64::from(width) * f64::from(height) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid_custom_sizes() {
        for p in PRESETS {
            assert_eq!(custom(&p.width.to_string(), &p.height.to_string()), Ok((p.width, p.height)));
        }
    }

    #[test]
    fn custom_sizes_need_multiples_of_32_within_range() {
        assert_eq!(custom("1024", "768"), Ok((1024, 768)));
        assert_eq!(custom(" 1024 ", "768"), Ok((1024, 768)));
        assert_eq!(custom("1000", "768"), Err(SizeError::NotMultiple));
        assert_eq!(custom("1024", "abc"), Err(SizeError::NotANumber));
        assert_eq!(custom("", "768"), Err(SizeError::NotANumber));
        assert_eq!(custom("128", "768"), Err(SizeError::OutOfRange));
        assert_eq!(custom("8224", "768"), Err(SizeError::OutOfRange));
    }

    #[test]
    fn the_final_presets_are_sixteen_megapixels() {
        for p in &PRESETS[1..] {
            let mp = megapixels(p.width, p.height);
            assert!((16.0..17.0).contains(&mp), "{}x{} is {mp} MP", p.width, p.height);
        }
    }
}
