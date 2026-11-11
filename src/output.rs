//! Select output by explicit CLI override, then container extension. Native frame
//! adapters live at the application boundary. The MP4 writer is platform-independent.
use mmh3_core::{media::Yuv420, picture::Picture, tensor::Tensor};
use mmh3_output::{DecodedMedia, FfmpegOutput, MediaSpec, OutputBackend, Result};
use std::path::{Path, PathBuf};

pub struct HostSession {
    backend: Box<dyn OutputBackend>,
    spec: MediaSpec,
    destination: PathBuf,
    video: Option<Yuv420>,
    audio: Option<Tensor>,
}

pub enum PreparedOutput {
    Host(HostSession),
    #[cfg(feature = "metal")]
    Apple(
        mmh3_output::mp4::Mp4Session<
            mmh3_output_videotoolbox::MetalVideoToolboxEncoder,
            mmh3_output::AacOutput,
        >,
    ),
    #[cfg(feature = "mp4")]
    Nvidia(mmh3_output::mp4::Mp4Session<mmh3_output_nvenc::NvencEncoder, mmh3_output::AacOutput>),
}

pub fn prepare(
    ffmpeg_arguments: Option<&[String]>,
    spec: MediaSpec,
    destination: &Path,
) -> Result<PreparedOutput> {
    spec.validate()?;
    let backend: Box<dyn OutputBackend> = if let Some(arguments) = ffmpeg_arguments {
        Box::new(FfmpegOutput::new(
            arguments.iter().map(Into::into).collect(),
        ))
    } else {
        let extension = destination
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match extension.as_str() {
            "mp4" => {
                #[cfg(feature = "metal")]
                {
                    return Ok(PreparedOutput::Apple(
                        mmh3_output::mp4::Mp4Session::prepare(
                            spec,
                            destination,
                            mmh3_output::AacOutput,
                            || mmh3_output_videotoolbox::MetalVideoToolboxEncoder::new(spec),
                        )?,
                    ));
                }
                #[cfg(feature = "mp4")]
                {
                    return Ok(PreparedOutput::Nvidia(
                        mmh3_output::mp4::Mp4Session::prepare(
                            spec,
                            destination,
                            mmh3_output::AacOutput,
                            || mmh3_output_nvenc::NvencEncoder::new(spec),
                        )?,
                    ));
                }
                #[cfg(not(any(feature = "mp4", feature = "metal")))]
                return Err(
                    "native MP4 needs --features mp4. Use --out FILE.webm or --ffmpeg".into(),
                );
            }
            "webm" => {
                #[cfg(feature = "webm")]
                {
                    Box::new(mmh3_output::WebmOutput::default())
                }
                #[cfg(not(feature = "webm"))]
                return Err(
                    "this build has no WebM backend. Rebuild with --features webm or pass --ffmpeg"
                        .into(),
                );
            }
            "png" => {
                return Err("generate writes a video: a picture comes from `mmh3 image`".into());
            }
            _ => {
                return Err(
                    "choose --out FILE.mp4 or FILE.webm, or pass --ffmpeg for other formats".into(),
                );
            }
        }
    };
    backend.validate(&spec, destination)?;
    Ok(PreparedOutput::Host(HostSession {
        backend,
        spec,
        destination: destination.into(),
        video: None,
        audio: None,
    }))
}
impl PreparedOutput {
    /// CPU-readable video, including frames decoded by the Metal backend.
    pub fn write_host_video(&mut self, frames: Yuv420) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Apple(_) => Err("VideoToolbox output requires Metal frames".into()),
            Self::Host(session) => {
                if session.video.is_some() {
                    return Err("video was already submitted".into());
                }
                session.video = Some(frames);
                Ok(())
            }
            #[cfg(feature = "mp4")]
            Self::Nvidia(_) => {
                Err("NVENC output requires CUDA frames. Use --ffmpeg for host frames".into())
            }
        }
    }
    #[cfg(feature = "metal")]
    pub fn streams_metal_video(&self) -> bool {
        matches!(self, Self::Apple(_))
    }
    #[cfg(feature = "metal")]
    pub fn write_metal_video_chunk(
        &mut self,
        frames: &mmh3_metal::vae::MetalVideoFrames,
    ) -> Result<()> {
        match self {
            Self::Apple(session) => session.write_video_chunk(frames, frames.frame_range()),
            Self::Host(_) => Err("host output requires the complete video".into()),
        }
    }
    #[cfg(feature = "metal")]
    pub fn write_metal_video(&mut self, frames: &mmh3_metal::vae::MetalVideoFrames) -> Result<()> {
        match self {
            Self::Apple(session) => session.write_video(frames),
            Self::Host(session) => {
                if session.video.is_some() {
                    return Err("video was already submitted".into());
                }
                session.video = Some(Yuv420::from_pixels(&frames.to_pixels()?)?);
                Ok(())
            }
        }
    }
    #[cfg(feature = "cuda")]
    pub fn write_cuda_video(&mut self, frames: &mmh3_cuda::vae::CudaVideoFrames) -> Result<()> {
        match self {
            Self::Host(session) => {
                if session.video.is_some() {
                    return Err("video was already submitted".into());
                }
                session.video = Some(frames.to_yuv420()?);
                Ok(())
            }
            #[cfg(feature = "mp4")]
            Self::Nvidia(session) => session.write_video(frames),
        }
    }
    pub fn write_audio(&mut self, audio: Tensor) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Apple(session) => session.write_audio(&audio),
            Self::Host(session) => {
                if session.audio.is_some() {
                    return Err("audio was already submitted".into());
                }
                session.audio = Some(audio);
                Ok(())
            }
            #[cfg(feature = "mp4")]
            Self::Nvidia(session) => session.write_audio(&audio),
        }
    }
    pub fn finish(self) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Apple(session) => session.finish(),
            Self::Host(session) => session.backend.write(
                &DecodedMedia {
                    spec: session.spec,
                    video: session.video.as_ref().ok_or("missing output video")?,
                    audio: session.audio.as_ref().ok_or("missing output audio")?,
                },
                &session.destination,
            ),
            #[cfg(feature = "mp4")]
            Self::Nvidia(session) => session.finish(),
        }
    }
}

/// A PNG file a picture goes to, checked before anything is generated.
pub struct PngOutput {
    destination: PathBuf,
}

impl PngOutput {
    pub fn prepare(destination: &Path) -> Result<Self> {
        if !destination
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        {
            return Err("image writes PNG: choose --out FILE.png".into());
        }
        mmh3_output::validate_destination(destination)?;
        Ok(Self {
            destination: destination.into(),
        })
    }

    /// Writes the picture beside the destination first, so that a failed write leaves an existing
    /// file as it was.
    pub fn write(self, picture: &Picture) -> Result<()> {
        let file = tempfile::NamedTempFile::new_in(mmh3_output::output_parent(&self.destination))?;
        encode_png(picture, file.as_file())?;
        file.persist(&self.destination)?;
        Ok(())
    }
}

fn encode_png(picture: &Picture, destination: impl std::io::Write) -> Result<()> {
    use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
    use std::io::Write;

    let mut writer = std::io::BufWriter::new(destination);
    PngEncoder::new(&mut writer).write_image(
        &picture.pixels,
        u32::try_from(picture.width)?,
        u32::try_from(picture.height)?,
        ExtendedColorType::Rgb8,
    )?;
    // The encoder leaves the PNG's end in the buffer, which a drop would write without reporting
    // a failure.
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_goes_to_a_png_file() {
        let directory = tempfile::tempdir().unwrap();
        for name in ["picture.mp4", "picture", "picture.jpg"] {
            assert!(
                PngOutput::prepare(&directory.path().join(name)).is_err(),
                "{name}"
            );
        }
        assert!(PngOutput::prepare(directory.path()).is_err());
        assert!(PngOutput::prepare(&directory.path().join("missing/picture.png")).is_err());

        let path = directory.path().join("picture.PNG");
        let picture = Picture {
            width: 3,
            height: 2,
            pixels: (0..18).collect(),
        };
        PngOutput::prepare(&path).unwrap().write(&picture).unwrap();
        let read = image::open(&path).unwrap().to_rgb8();
        assert_eq!((read.width(), read.height()), (3, 2));
        assert_eq!(read.into_raw(), picture.pixels);
    }

    #[test]
    fn a_failed_png_write_is_reported() {
        struct Full {
            room: usize,
        }
        impl std::io::Write for Full {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > self.room {
                    return Err(std::io::ErrorKind::StorageFull.into());
                }
                self.room -= bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let picture = Picture {
            width: 3,
            height: 2,
            pixels: (0..18).collect(),
        };
        assert!(encode_png(&picture, Full { room: usize::MAX }).is_ok());
        assert!(encode_png(&picture, Full { room: 16 }).is_err());
    }

    #[test]
    fn generate_points_a_picture_to_image() {
        let spec = mmh3_output::MediaSpec {
            width: 64,
            height: 64,
            frames: 5,
            fps: 24,
            sample_rate: 32000,
            channels: 2,
        };
        let error = prepare(None, spec, Path::new("picture.png")).err().unwrap();
        assert!(error.to_string().contains("mmh3 image"), "{error}");
    }
}
