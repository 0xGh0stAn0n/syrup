//! The host side of a real run: serves `detect` from the providers, checks
//! what they return, and collects what the module emits.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::abi::*;
use crate::catalog::Capability;
use crate::contract::ImageInput;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::interp::group_runs;
use crate::providers::{Detections, Providers, ViewRef};

pub struct ExecHost<'a> {
    image: ImageInput<'a>,
    providers: &'a Providers,
    scratch: Vec<SyrupDetection>,
    rects: Vec<SyrupRect>,
    pub out: Vec<SyrupDetection>,
    /// Text read by providers; a detection's payload is its index + 1.
    pub texts: Vec<String>,
    pub error: Option<SyrupError>,
}

impl<'a> ExecHost<'a> {
    pub fn new(image: ImageInput<'a>, providers: &'a Providers) -> Self {
        ExecHost {
            image,
            providers,
            scratch: vec![],
            rects: vec![],
            out: vec![],
            texts: vec![],
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
            group: host_group,
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
        let Detections { mut boxes, texts } = self.providers.get(capability).detect(&view)?;
        let malformed = |what: String| {
            SyrupError::new(
                Stage::Execute,
                ErrorKind::ProviderFailed,
                format!("the {} provider returned {what}", capability.as_str()),
            )
        };
        if let Some(bad) = boxes.iter().find(|d| !well_formed(d)) {
            return Err(malformed(format!("a malformed detection: {bad:?}")));
        }
        if !texts.is_empty() && texts.len() != boxes.len() {
            return Err(malformed(format!(
                "{} texts for {} boxes",
                texts.len(),
                boxes.len()
            )));
        }
        for d in &mut boxes {
            d.payload = 0;
        }
        for (d, text) in boxes.iter_mut().zip(texts) {
            self.texts.push(text);
            d.payload = self.texts.len() as u64;
        }
        self.scratch = boxes;
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

unsafe extern "C" fn host_group(
    ctx: *mut c_void,
    runs: *const SyrupRun,
    n_runs: usize,
    min_height: u32,
    max_gap: u32,
    out_ptr: *mut *const SyrupRect,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: as in host_detect; `runs` holds `n_runs` runs for the call.
    let host = unsafe { &mut *(ctx as *mut ExecHost) };
    let runs = match n_runs {
        0 => &[][..],
        n => unsafe { std::slice::from_raw_parts(runs, n) },
    };
    match catch_unwind(AssertUnwindSafe(|| group_runs(runs, min_height, max_gap))) {
        Ok(rects) => {
            host.rects = rects;
            unsafe {
                *out_ptr = host.rects.as_ptr();
                *out_len = host.rects.len();
            }
            SYRUP_OK
        }
        Err(_) => {
            host.error = Some(SyrupError::new(
                Stage::Execute,
                ErrorKind::Panic,
                "region grouping panicked",
            ));
            SYRUP_ERR_PROVIDER
        }
    }
}
