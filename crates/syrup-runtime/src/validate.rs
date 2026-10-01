//! Differential validation. Before an artifact is published it runs on
//! synthetic images against a mock detector, and its calls and results must
//! match the reference interpreter's exactly. The mock works out which
//! window it was handed from the view's pointer, so a crop at the wrong
//! offset, stride or channel count is caught, not just wrong arithmetic.

use std::ffi::c_void;

use crate::abi::*;
use crate::catalog::Capability;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent::PixelRect;
use crate::interp::{self, Detector};
use crate::loader::Module;
use crate::plan::Plan;

const SIZES: &[(u32, u32)] = &[
    (1, 1),
    (2, 3),
    (5, 7),
    (16, 9),
    (31, 17),
    (64, 48),
    (97, 61),
    (3, 100),
];

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Detections that depend only on where the detector looked.
fn mock_detections(seed: u64, capability: u32, view: PixelRect) -> Vec<SyrupDetection> {
    let mut rng = SplitMix(
        seed ^ ((capability as u64) << 56)
            ^ ((view.x as u64) << 42)
            ^ ((view.y as u64) << 28)
            ^ ((view.w as u64) << 14)
            ^ view.h as u64,
    );
    let (w, h) = (view.w as f32, view.h as f32);
    let mut out: Vec<SyrupDetection> = vec![];
    for _ in 0..rng.below(7) {
        if let Some(last) = out.last().copied()
            && rng.below(4) == 0
        {
            out.push(last); // exact tie
            continue;
        }
        let mut keypoints = [0.0; 2 * SYRUP_MAX_KEYPOINTS];
        for (i, k) in keypoints.iter_mut().enumerate() {
            *k = rng.range(-0.2, 1.2) * if i % 2 == 0 { w } else { h };
        }
        out.push(SyrupDetection {
            x: rng.range(-0.3, 1.1) * w,
            y: rng.range(-0.3, 1.1) * h,
            w: rng.range(-0.05, 0.8) * w,
            h: rng.range(-0.05, 0.8) * h,
            score: [0.0, 0.3, 0.5, 0.6, 0.6, 0.9, 1.0, rng.unit()][rng.below(8) as usize],
            n_keypoints: [0, 5, 7][rng.below(3) as usize],
            keypoints,
            payload: rng.next(),
        });
    }
    out
}

struct MockHost {
    base: usize,
    len: usize,
    width: u32,
    height: u32,
    stride: usize,
    channels: u32,
    seed: u64,
    scratch: Vec<SyrupDetection>,
    calls: Vec<PixelRect>,
    out: Vec<SyrupDetection>,
    fault: Option<String>,
}

impl MockHost {
    fn locate(&self, view: &SyrupImageView) -> std::result::Result<PixelRect, String> {
        if view.stride != self.stride || view.channels != self.channels {
            return Err(format!(
                "view has stride {} and {} channels, image has {} and {}",
                view.stride, view.channels, self.stride, self.channels
            ));
        }
        let offset = (view.data as usize).wrapping_sub(self.base);
        if offset >= self.len || !(offset % self.stride).is_multiple_of(self.channels as usize) {
            return Err(format!(
                "view starts at byte {offset}, which is not a pixel of the image"
            ));
        }
        let (x, y) = (
            ((offset % self.stride) / self.channels as usize) as u32,
            (offset / self.stride) as u32,
        );
        if x + view.width > self.width || y + view.height > self.height {
            return Err(format!(
                "view {}x{} at ({x}, {y}) leaves the {}x{} image",
                view.width, view.height, self.width, self.height
            ));
        }
        Ok(PixelRect {
            x,
            y,
            w: view.width,
            h: view.height,
        })
    }
}

unsafe extern "C" fn mock_detect(
    ctx: *mut c_void,
    capability: u32,
    view: *const SyrupImageView,
    out_ptr: *mut *const SyrupDetection,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: ctx is the MockHost passed to the module by `check`.
    let host = unsafe { &mut *(ctx as *mut MockHost) };
    // SAFETY: the module passes a valid view.
    match host.locate(unsafe { &*view }) {
        Ok(rect) => {
            host.calls.push(rect);
            host.scratch = mock_detections(host.seed, capability, rect);
            // SAFETY: out pointers are valid per the ABI.
            unsafe {
                *out_ptr = host.scratch.as_ptr();
                *out_len = host.scratch.len();
            }
            SYRUP_OK
        }
        Err(fault) => {
            host.fault = Some(fault);
            SYRUP_ERR_PROVIDER
        }
    }
}

unsafe extern "C" fn mock_emit(ctx: *mut c_void, detection: *const SyrupDetection) -> i32 {
    // SAFETY: as in mock_detect.
    unsafe { (*(ctx as *mut MockHost)).out.push(*detection) };
    SYRUP_OK
}

struct Oracle {
    seed: u64,
    calls: Vec<PixelRect>,
}

impl Detector for Oracle {
    fn detect(&mut self, capability: Capability, view: PixelRect) -> Result<Vec<SyrupDetection>> {
        self.calls.push(view);
        Ok(mock_detections(self.seed, capability.abi_id(), view))
    }
}

fn same(a: &SyrupDetection, b: &SyrupDetection) -> bool {
    let close = |x: f32, y: f32| (x - y).abs() <= 1e-4;
    let n = (a.n_keypoints as usize).min(SYRUP_MAX_KEYPOINTS);
    close(a.x, b.x)
        && close(a.y, b.y)
        && close(a.w, b.w)
        && close(a.h, b.h)
        && close(a.score, b.score)
        && a.n_keypoints == b.n_keypoints
        && a.payload == b.payload
        && a.keypoints[..2 * n]
            .iter()
            .zip(&b.keypoints[..2 * n])
            .all(|(x, y)| close(*x, *y))
}

/// Returns the number of cases run.
pub fn check(module: &Module, plan: &Plan) -> Result<u32> {
    let mut rng = SplitMix(0x5eed);
    let mut cases = 0;
    for (index, &(width, height)) in SIZES.iter().enumerate() {
        for channels in [1u32, 3, 4] {
            let stride = (width * channels) as usize + index % 3;
            let image = vec![0u8; stride * height as usize];
            let region = (
                rng.below(width as u64 + 1) as u32,
                rng.below(height as u64 + 1) as u32,
            );
            let params = SyrupParams {
                min_confidence: [0.0, 0.5, 0.6, 1.0][rng.below(4) as usize],
                max_results: [0, 1, 2, 5][rng.below(4) as usize],
                has_region: plan.needs_caller_region() as u32,
                region_x: region.0,
                region_y: region.1,
                region_w: rng.below((width - region.0) as u64 + 1) as u32,
                region_h: rng.below((height - region.1) as u64 + 1) as u32,
            };
            let seed = rng.next();

            let mut host = MockHost {
                base: image.as_ptr() as usize,
                len: image.len(),
                width,
                height,
                stride,
                channels,
                seed,
                scratch: vec![],
                calls: vec![],
                out: vec![],
                fault: None,
            };
            let table = SyrupHost {
                abi_version: SYRUP_ABI_VERSION,
                struct_size: size_of::<SyrupHost>() as u32,
                ctx: (&raw mut host).cast(),
                detect: mock_detect,
                emit: mock_emit,
            };
            let input = SyrupImageView {
                data: image.as_ptr(),
                width,
                height,
                stride,
                channels,
            };
            // SAFETY: table.ctx points at `host`, which outlives the call.
            let status = unsafe { module.run(&table, &input, &params) };

            let mut oracle = Oracle {
                seed,
                calls: vec![],
            };
            let expected = interp::run(plan, width, height, &params, &mut oracle)?;

            let case = format!("{width}x{height}x{channels} stride {stride}, {params:?}");
            let mismatch = |reason: String| {
                SyrupError::new(
                    Stage::Validate,
                    ErrorKind::Mismatch,
                    format!("{reason} ({case})"),
                )
                .with_hint("this is a code generator bug; the module was not published")
            };
            if let Some(fault) = host.fault {
                return Err(mismatch(fault));
            }
            if status != SYRUP_OK {
                return Err(mismatch(format!("module returned status {status}")));
            }
            if host.calls != oracle.calls {
                return Err(mismatch(format!(
                    "module looked at {:?}, expected {:?}",
                    host.calls, oracle.calls
                )));
            }
            if host.out.len() != expected.len()
                || !host.out.iter().zip(&expected).all(|(a, b)| same(a, b))
            {
                return Err(mismatch(format!(
                    "module emitted {:?}, expected {:?}",
                    host.out, expected
                )));
            }
            cases += 1;
        }
    }
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use super::*;
    use crate::codegen::generate;
    use crate::compiler::Rustc;
    use crate::intent::parse;

    struct Built {
        plan: Plan,
        module: Module,
        dir: PathBuf,
    }

    impl Drop for Built {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Compiles the module for `name` after applying `edit` to its source.
    fn build(name: &str, test: &str, edit: impl Fn(&str) -> String) -> Built {
        let intent = parse(name).unwrap();
        let plan = Plan::build(&intent).unwrap();
        let source = generate(&plan, &intent, &plan.hash());
        let edited = edit(&source);
        assert_ne!(edited, source, "the edit did not apply");
        let dir =
            std::env::temp_dir().join(format!("syrup-validate-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("op.rs"), edited).unwrap();
        let library = dir.join(format!(
            "{}op{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ));
        let rustc = Rustc::find(None).unwrap();
        rustc
            .compile(&dir.join("op.rs"), &library, "op", Duration::from_secs(300))
            .unwrap();
        let module = Module::load(&library, &plan.hash()).unwrap();
        Built { plan, module, dir }
    }

    fn mismatch(built: &Built) -> SyrupError {
        let e = check(&built.module, &built.plan).unwrap_err();
        assert_eq!(
            (e.stage, e.kind),
            (Stage::Validate, ErrorKind::Mismatch),
            "{e}"
        );
        e
    }

    #[test]
    fn faithful_modules_pass() {
        let built = build(
            "find_2_largest_faces_in_region_larger_than_1pct",
            "faithful",
            |s| format!("{s}\n"),
        );
        assert_eq!(
            check(&built.module, &built.plan).unwrap(),
            SIZES.len() as u32 * 3
        );
    }

    #[test]
    fn a_crop_one_row_off_is_caught() {
        let built = build("find_faces_in_bottom_half", "row", |s| {
            s.replace(
                "data.wrapping_add(offset)",
                "data.wrapping_add(offset + parent.view.stride)",
            )
        });
        mismatch(&built);
    }

    #[test]
    fn a_missing_coordinate_restore_is_caught() {
        let built = build("find_faces_in_right_half", "restore", |s| {
            s.replace("x: parent.x + x,", "x: parent.x,")
        });
        mismatch(&built);
    }

    #[test]
    fn a_reversed_ordering_is_caught() {
        let built = build("find_largest_face", "order", |s| {
            s.replace(
                "(b.w * b.h).total_cmp(&(a.w * a.h))",
                "(a.w * a.h).total_cmp(&(b.w * b.h))",
            )
        });
        mismatch(&built);
    }

    #[test]
    fn a_panic_is_contained_at_the_boundary() {
        let built = build("find_face", "panic", |s| {
            let entry = "params: &SyrupParams) -> Result<(), i32> {\n";
            s.replace(
                entry,
                &format!("{entry}    if input.width > 0 {{\n        panic!(\"boom\");\n    }}\n"),
            )
        });
        assert!(
            mismatch(&built)
                .reason
                .contains(&format!("status {SYRUP_ERR_PANIC}"))
        );
    }
}
