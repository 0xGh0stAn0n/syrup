//! The host side of a real run: serves `detect` from the providers, checks
//! what they return, and collects what the module emits.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use syrup::color::is_color_pixel;
use syrup::geometry::{Rect, measure_bar_fill};
use syrup::quality::{Legibility, assess_text_quality};

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
            measure: host_measure,
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
        let Detections { mut boxes, texts } = self.providers.get(capability)?.detect(&view)?;
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

/// The core's measurements, on the pixels of `view`.
pub fn measure(
    view: &ViewRef<'_>,
    what: &SyrupMeasure,
    boxes: &[SyrupDetection],
) -> Result<Vec<f32>> {
    let pixels = view.to_rgba();
    let whole = Rect {
        x: 0,
        y: 0,
        w: view.width,
        h: view.height,
    };
    let rect = |d: &SyrupDetection| {
        let x0 = (d.x.floor().max(0.0) as u32).min(view.width);
        let y0 = (d.y.floor().max(0.0) as u32).min(view.height);
        let x1 = ((d.x + d.w).ceil().max(0.0) as u32).min(view.width);
        let y1 = ((d.y + d.h).ceil().max(0.0) as u32).min(view.height);
        Rect {
            x: x0,
            y: y0,
            w: x1.saturating_sub(x0),
            h: y1.saturating_sub(y0),
        }
    };
    match what.kind {
        SYRUP_MEASURE_SHARPNESS => Ok(boxes
            .iter()
            .map(|d| {
                let quality = assess_text_quality(&pixels, rect(d));
                match quality.legibility {
                    Legibility::NoText => f32::NAN,
                    _ => quality.sharpness,
                }
            })
            .collect()),
        SYRUP_MEASURE_FILL => {
            let hue = (what.hue_lo as f32, what.hue_hi as f32);
            let (sat, val) = (
                what.min_saturation_pct as f32 / 100.0,
                what.min_value_pct as f32 / 100.0,
            );
            Ok(boxes
                .iter()
                .map(|d| {
                    measure_bar_fill(&pixels, rect(d), whole, |p| {
                        is_color_pixel(p, hue, sat, val)
                    })
                    .map_or(f32::NAN, |pct| pct / 100.0)
                })
                .collect())
        }
        kind => Err(SyrupError::new(
            Stage::Execute,
            ErrorKind::ModuleFailed,
            format!("the module asked for unknown measurement {kind}"),
        )),
    }
}

pub fn well_formed(d: &SyrupDetection) -> bool {
    [d.x, d.y, d.w, d.h]
        .iter()
        .chain(&d.keypoints)
        .all(|v| v.is_finite())
        && d.w >= 0.0
        && d.h >= 0.0
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

unsafe extern "C" fn host_measure(
    ctx: *mut c_void,
    what: *const SyrupMeasure,
    view: *const SyrupImageView,
    boxes: *const SyrupDetection,
    n_boxes: usize,
    values: *mut f32,
) -> i32 {
    // SAFETY: as in host_detect; `boxes` and `values` hold `n_boxes` items.
    let host = unsafe { &mut *(ctx as *mut ExecHost) };
    let (what, view) = unsafe { (&*what, &*view) };
    let boxes = match n_boxes {
        0 => &[][..],
        n => unsafe { std::slice::from_raw_parts(boxes, n) },
    };
    let measured = catch_unwind(AssertUnwindSafe(|| {
        host.view(view).and_then(|view| measure(&view, what, boxes))
    }));
    match measured {
        Ok(Ok(measured)) => {
            for (i, value) in measured.into_iter().enumerate() {
                unsafe { *values.add(i) = value };
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
                "a measurement panicked",
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
