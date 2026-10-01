//! Live window capture by title.
//!
//! [`Window::find`] opens the first window whose title contains a query
//! (ignoring case) and [`Window::capture`] returns its current contents as
//! RGBA. Each platform uses its own mechanism:
//!
//! - **Windows**: `PrintWindow`, so a covered window is captured as drawn.
//! - **Linux and the BSDs, X11** (including X11 applications on a Wayland
//!   desktop, via XWayland): the Composite extension, so a covered window
//!   is captured as drawn.
//! - **Linux and the BSDs, Wayland**: the desktop's screen-cast portal and
//!   PipeWire. Wayland does not show other applications' window titles, so
//!   the desktop asks the user to pick the window. The choice is remembered
//!   for that query, and later runs capture the same window without asking.
//! - **macOS 14 and later**: ScreenCaptureKit. The process needs the Screen
//!   Recording permission (System Settings > Privacy & Security).

use std::fmt;

use image::RgbaImage;

#[cfg_attr(target_os = "windows", path = "windows.rs")]
#[cfg_attr(target_os = "macos", path = "macos.rs")]
#[cfg_attr(all(unix, not(target_os = "macos")), path = "unix.rs")]
#[cfg_attr(not(any(unix, windows)), path = "unsupported.rs")]
mod platform;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// No window's title contains the query.
    NotFound,
    /// The window has been closed.
    Closed,
    /// This system cannot capture windows, e.g. there is no display.
    Unavailable(String),
    /// Capture was refused: a missing permission, or the user declined.
    Denied(String),
    /// The platform reported an error.
    Failed(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaptureError::NotFound => f.write_str("no window matches"),
            CaptureError::Closed => f.write_str("the window was closed"),
            CaptureError::Unavailable(reason)
            | CaptureError::Denied(reason)
            | CaptureError::Failed(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for CaptureError {}

/// An open window, captured on demand.
pub struct Window {
    title: String,
    inner: platform::Window,
}

impl Window {
    /// The first window whose title contains `query`, ignoring case.
    pub fn find(query: &str) -> Result<Window, CaptureError> {
        let (title, inner) = platform::find(query)?;
        Ok(Window { title, inner })
    }

    /// The window's full title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The window's current contents, without its frame.
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        self.inner.capture()
    }
}

/// Titles of the windows [`Window::find`] can match without asking the
/// user. On Wayland those are X11 applications' windows only.
pub fn list_windows() -> Result<Vec<String>, CaptureError> {
    platform::list_windows()
}

/// Does `title` contain `query`, ignoring case?
#[cfg_attr(not(any(unix, windows)), allow(dead_code))]
fn matches(title: &str, query: &str) -> bool {
    title.to_lowercase().contains(&query.to_lowercase())
}
