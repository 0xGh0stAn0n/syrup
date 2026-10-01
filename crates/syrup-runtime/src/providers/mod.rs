//! Capabilities the host offers generated modules.

pub mod motion;
pub mod tesseract;
#[cfg(feature = "face-yunet")]
pub mod yunet;

use std::path::PathBuf;
use std::sync::Arc;

use crate::abi::SyrupDetection;
use crate::catalog::Capability;
use crate::contract::ProviderInfo;
use crate::custom;
use crate::error::{ErrorKind, Result, Stage, SyrupError};

/// A window of the input image, borrowed for one `detect` call.
pub struct ViewRef<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub channels: u32,
}

impl ViewRef<'_> {
    fn pixel(&self, x: u32, y: u32) -> &[u8] {
        let c = self.channels as usize;
        let at = y as usize * self.stride + x as usize * c;
        &self.data[at..at + c]
    }

    /// The pixels with rows packed tightly: `height * width * channels` bytes.
    pub fn packed(&self) -> Vec<u8> {
        let row = self.width as usize * self.channels as usize;
        (0..self.height as usize)
            .flat_map(|y| &self.data[y * self.stride..][..row])
            .copied()
            .collect()
    }

    pub fn to_rgba(&self) -> image::RgbaImage {
        image::RgbaImage::from_fn(self.width, self.height, |x, y| {
            let p = self.pixel(x, y);
            image::Rgba(match p.len() {
                1 => [p[0], p[0], p[0], 255],
                3 => [p[0], p[1], p[2], 255],
                _ => [p[0], p[1], p[2], p[3]],
            })
        })
    }

    pub fn to_rgb(&self) -> image::RgbImage {
        image::RgbImage::from_fn(self.width, self.height, |x, y| {
            let p = self.pixel(x, y);
            image::Rgb(if p.len() == 1 {
                [p[0]; 3]
            } else {
                [p[0], p[1], p[2]]
            })
        })
    }
}

/// Boxes in the view's pixel coordinates, unclipped, and for providers that
/// read text, one string per box.
#[derive(Debug, Default)]
pub struct Detections {
    pub boxes: Vec<SyrupDetection>,
    pub texts: Vec<String>,
}

pub trait Provider: Send + Sync {
    fn info(&self) -> ProviderInfo;

    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    Bundled,
    File(PathBuf),
}

pub struct Providers {
    face: Arc<dyn Provider>,
    text: Arc<dyn Provider>,
}

impl Providers {
    pub fn new(face_model: ModelSource) -> Providers {
        #[cfg(feature = "face-yunet")]
        let face: Arc<dyn Provider> = Arc::new(yunet::YuNet::new(face_model));
        #[cfg(not(feature = "face-yunet"))]
        let face: Arc<dyn Provider> = {
            let _ = face_model;
            Arc::new(Unavailable)
        };
        Providers {
            face,
            text: Arc::new(tesseract::Tesseract),
        }
    }

    pub fn get(&self, capability: Capability) -> Result<Arc<dyn Provider>> {
        match capability {
            Capability::FaceDetection => Ok(self.face.clone()),
            Capability::TextRecognition => Ok(self.text.clone()),
            Capability::Motion => Err(SyrupError::new(
                Stage::Execute,
                ErrorKind::BadParameter,
                "moving regions are found between frames, so only sessions find them",
            )
            .with_hint("op.session(), then call it with each frame")),
            Capability::Custom(name) => custom::provider(name).ok_or_else(|| {
                SyrupError::new(
                    Stage::Execute,
                    ErrorKind::MissingDependency,
                    format!(
                        "no detector for {} is registered in this process",
                        name.as_str()
                    ),
                )
                .with_hint("add the target (syrup.add_target) before running operations on it")
            }),
        }
    }
}

#[cfg(not(feature = "face-yunet"))]
struct Unavailable;

#[cfg(not(feature = "face-yunet"))]
impl Provider for Unavailable {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: Capability::FaceDetection.as_str(),
            name: "unavailable",
            model_sha256: None,
            runtime: "none",
        }
    }

    fn detect(&self, _: &ViewRef<'_>) -> Result<Detections> {
        Err(SyrupError::new(
            Stage::Execute,
            ErrorKind::MissingDependency,
            "this build of syrup-runtime has no face detector",
        )
        .with_hint("build with the `face-yunet` feature"))
    }
}
