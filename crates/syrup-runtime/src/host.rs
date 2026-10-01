//! The host side of a real run: serves `detect` from the providers, checks
//! what they return, and collects what the module emits.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::abi::*;
use crate::catalog::Capability;
use crate::contract::ImageInput;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::providers::{Providers, ViewRef};

pub struct ExecHost<'a> {
    image: ImageInput<'a>,
    providers: &'a Providers,
    scratch: Vec<SyrupDetection>,
    pub out: Vec<SyrupDetection>,
    pub error: Option<SyrupError>,
}

impl<'a> ExecHost<'a> {
    pub fn new(image: ImageInput<'a>, providers: &'a Providers) -> Self {
        ExecHost {
            image,
            providers,
            scratch: vec![],
            out: vec![],
            error: None,
        }
    }

    /// `self` must stay put while the table is in use.
    pub fn table(&mut self) -> SyrupHost {
        SyrupHost {
            abi_version: SYRUP_ABI_VERSION,
            struct_size: size_of::<SyrupHost>() as u32,
            ctx: (self as *mut Self).cast(),
            detect: host_detect,
            emit: host_emit,
        }
    }

    fn view(&self, view: &SyrupImageView) -> Result<ViewRef<'a>> {
        let data = self.image.data();
        let (stride, channels) = (self.image.stride(), self.image.channels() as usize);
        let offset = (view.data as usize).wrapping_sub(data.as_ptr() as usize);
        let (x, y) = ((offset % stride) / channels, offset / stride);
        let inside = view.stride == stride
            && view.channels as usize == channels
            && offset < data.len()
            && (offset % stride).is_multiple_of(channels)
            && view.width > 0
            && view.height > 0
            && x + view.width as usize <= self.image.width() as usize
            && y + view.height as usize <= self.image.height() as usize;
        if !inside {
            return Err(SyrupError::new(
                Stage::Execute,
                ErrorKind::ModuleFailed,
                format!("the module passed a view outside the input image ({view:?})"),
            ));
        }
        let len = (view.height as usize - 1) * stride + view.width as usize * channels;
        Ok(ViewRef {
            data: &data[offset..offset + len],
            width: view.width,
            height: view.height,
            stride,
            channels: view.channels,
        })
    }

    fn detect(&mut self, capability: u32, view: &SyrupImageView) -> Result<()> {
        let capability = Capability::from_abi_id(capability).ok_or_else(|| {
            SyrupError::new(
                Stage::Execute,
                ErrorKind::ModuleFailed,
                format!("the module asked for unknown capability {capability}"),
            )
        })?;
        let view = self.view(view)?;
        let detections = self.providers.get(capability).detect(&view)?;
        if let Some(bad) = detections.iter().find(|d| !well_formed(d)) {
            return Err(SyrupError::new(
                Stage::Execute,
                ErrorKind::ProviderFailed,
                format!(
                    "the {} provider returned a malformed detection: {bad:?}",
                    capability.as_str()
                ),
            ));
        }
        self.scratch = detections;
        Ok(())
    }
}

pub fn well_formed(d: &SyrupDetection) -> bool {
    [d.x, d.y, d.w, d.h]
        .iter()
        .chain(&d.keypoints)
        .all(|v| v.is_finite())
        && (0.0..=1.0).contains(&d.score)
        && d.n_keypoints as usize <= SYRUP_MAX_KEYPOINTS
}

unsafe extern "C" fn host_detect(
    ctx: *mut c_void,
    capability: u32,
    view: *const SyrupImageView,
    out_ptr: *mut *const SyrupDetection,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: ctx is the ExecHost that built this table; view and the out
    // pointers come from the module and are valid per the ABI.
    let host = unsafe { &mut *(ctx as *mut ExecHost) };
    let view = unsafe { &*view };
    match catch_unwind(AssertUnwindSafe(|| host.detect(capability, view))) {
        Ok(Ok(())) => {
            unsafe {
                *out_ptr = host.scratch.as_ptr();
                *out_len = host.scratch.len();
            }
            SYRUP_OK
        }
        Ok(Err(e)) => {
            host.error = Some(e);
            SYRUP_ERR_PROVIDER
        }
        Err(_) => {
            host.error = Some(SyrupError::new(
                Stage::Execute,
                ErrorKind::Panic,
                "a provider panicked",
            ));
            SYRUP_ERR_PROVIDER
        }
    }
}

unsafe extern "C" fn host_emit(ctx: *mut c_void, detection: *const SyrupDetection) -> i32 {
    // SAFETY: as in host_detect.
    unsafe { (*(ctx as *mut ExecHost)).out.push(*detection) };
    SYRUP_OK
}
