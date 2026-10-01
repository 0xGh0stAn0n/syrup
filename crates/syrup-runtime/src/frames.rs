//! Where a session's frames come from: image files anywhere, or a live
//! window on Windows.

use std::path::PathBuf;

use crate::contract::OwnedImage;
use crate::error::{ErrorKind, Result, Stage, SyrupError};

pub trait FrameSource {
    /// The next frame, or `None` when the source has ended.
    fn next_frame(&mut self) -> Result<Option<OwnedImage>>;
}

/// Image files, in the order given.
pub struct ImageFiles {
    paths: std::vec::IntoIter<PathBuf>,
}

impl ImageFiles {
    pub fn new(paths: impl IntoIterator<Item = PathBuf>) -> ImageFiles {
        ImageFiles {
            paths: paths.into_iter().collect::<Vec<_>>().into_iter(),
        }
    }
}

impl FrameSource for ImageFiles {
    fn next_frame(&mut self) -> Result<Option<OwnedImage>> {
        self.paths
            .next()
            .map(|path| OwnedImage::open(&path))
            .transpose()
    }
}

/// A window's client area, captured with the core's `capture` module, until
/// the window goes away. Windows only.
pub struct WindowCapture {
    title: String,
    captured: u64,
}

impl WindowCapture {
    /// The first window whose title contains `title`.
    pub fn new(title: &str) -> Result<WindowCapture> {
        if !cfg!(target_os = "windows") {
            return Err(SyrupError::new(
                Stage::Input,
                ErrorKind::MissingDependency,
                "window capture is only available on Windows",
            )
            .with_hint("on other systems, feed frames from files or your own capture"));
        }
        Ok(WindowCapture {
            title: title.to_string(),
            captured: 0,
        })
    }

    /// Titles of the windows that can be captured.
    pub fn windows() -> Vec<String> {
        syrup::capture::list_windows()
    }
}

impl FrameSource for WindowCapture {
    /// Fails if the window never existed; ends when it closes.
    fn next_frame(&mut self) -> Result<Option<OwnedImage>> {
        match syrup::capture::capture_window_by_title_info(&self.title) {
            Some((_, image)) => {
                self.captured += 1;
                Ok(Some(OwnedImage {
                    width: image.width(),
                    height: image.height(),
                    channels: 4,
                    data: image.into_raw(),
                }))
            }
            None if self.captured == 0 => Err(SyrupError::new(
                Stage::Input,
                ErrorKind::BadParameter,
                format!("no window title contains {:?}", self.title),
            )
            .with_hint("`syrup windows` lists the windows that can be captured")),
            None => Ok(None),
        }
    }
}
