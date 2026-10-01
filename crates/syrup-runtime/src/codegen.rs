//! Plan → Rust source for a dependency-free `cdylib`. Each step kind has a
//! small template; `run` in the generated file is the composed program.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent::{Intent, OrderKey, RegionSpec};
use crate::plan::{Plan, Step};

pub const CODEGEN_VERSION: u32 = 1;

pub const EXPORTS: [&str; 3] = ["syrup_op_abi_version", "syrup_op_plan_hash", "syrup_op_run"];

const PRELUDE: &str = include_str!("abi_prelude.rs");

const ENTRY: &str = r#"
#[unsafe(no_mangle)]
pub extern "C" fn syrup_op_abi_version() -> u32 {
    SYRUP_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn syrup_op_plan_hash() -> *const u8 {
    PLAN_HASH.as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_op_run(
    host: *const SyrupHost,
    input: *const SyrupImageView,
    params: *const SyrupParams,
) -> i32 {
    if host.is_null() || input.is_null() || params.is_null() {
        return SYRUP_ERR_INPUT;
    }
    // SAFETY: the host passes pointers that stay valid for the whole call.
    let (host, input, params) = unsafe { (&*host, &*input, &*params) };
    if host.abi_version != SYRUP_ABI_VERSION
        || (host.struct_size as usize) < core::mem::size_of::<SyrupHost>()
    {
        return SYRUP_ERR_ABI;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(host, input, params))) {
        Ok(Ok(())) => SYRUP_OK,
        Ok(Err(status)) => status,
        Err(_) => SYRUP_ERR_PANIC,
    }
}
"#;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Helper {
    Window,
    SelectFixed,
    SelectCaller,
    Detect,
    Restore,
    TieBreak,
    Emit,
}

impl Helper {
    fn source(self) -> &'static str {
        match self {
            Helper::Window => {
                r#"
#[derive(Clone, Copy)]
struct Window {
    view: SyrupImageView,
    x: u32,
    y: u32,
}

fn sub_window(parent: &Window, x: u32, y: u32, w: u32, h: u32) -> Window {
    let offset = y as usize * parent.view.stride + x as usize * parent.view.channels as usize;
    Window {
        view: SyrupImageView {
            data: parent.view.data.wrapping_add(offset),
            width: w,
            height: h,
            stride: parent.view.stride,
            channels: parent.view.channels,
        },
        x: parent.x + x,
        y: parent.y + y,
    }
}
"#
            }
            Helper::SelectFixed => {
                r#"
fn round_frac(num: u64, den: u64, extent: u32) -> u32 {
    ((2 * num * extent as u64 + den) / (2 * den)) as u32
}

fn select_fixed(parent: &Window, x0: (u64, u64), y0: (u64, u64), x1: (u64, u64), y1: (u64, u64)) -> Window {
    let (w, h) = (parent.view.width, parent.view.height);
    let left = round_frac(x0.0, x0.1, w);
    let top = round_frac(y0.0, y0.1, h);
    let right = round_frac(x1.0, x1.1, w);
    let bottom = round_frac(y1.0, y1.1, h);
    sub_window(parent, left, top, right - left, bottom - top)
}
"#
            }
            Helper::SelectCaller => {
                r#"
fn select_caller(parent: &Window, params: &SyrupParams) -> Result<Window, i32> {
    if params.has_region != 1 {
        return Err(SYRUP_ERR_INPUT);
    }
    let (w, h) = (parent.view.width, parent.view.height);
    let left = params.region_x.min(w);
    let top = params.region_y.min(h);
    let right = (params.region_x as u64 + params.region_w as u64).min(w as u64) as u32;
    let bottom = (params.region_y as u64 + params.region_h as u64).min(h as u64) as u32;
    Ok(sub_window(parent, left, top, right.saturating_sub(left), bottom.saturating_sub(top)))
}
"#
            }
            Helper::Detect => {
                r#"
fn detect(host: &SyrupHost, capability: u32, window: &Window) -> Result<Vec<SyrupDetection>, i32> {
    if window.view.width == 0 || window.view.height == 0 {
        return Ok(Vec::new());
    }
    let mut ptr: *const SyrupDetection = core::ptr::null();
    let mut len = 0usize;
    // SAFETY: the view borrows the input image, which outlives the call.
    let status = unsafe { (host.detect)(host.ctx, capability, &window.view, &mut ptr, &mut len) };
    if status != SYRUP_OK {
        return Err(status);
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(SYRUP_ERR_PROVIDER);
    }
    // SAFETY: the host keeps `len` detections at `ptr` until our next call into it.
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec())
}
"#
            }
            Helper::Restore => {
                r#"
fn restore(boxes: Vec<SyrupDetection>, window: &Window) -> Vec<SyrupDetection> {
    let left = window.x as f32;
    let top = window.y as f32;
    let right = (window.x + window.view.width) as f32;
    let bottom = (window.y + window.view.height) as f32;
    let mut out = Vec::with_capacity(boxes.len());
    for d in boxes {
        let x0 = (d.x + left).max(left).min(right);
        let y0 = (d.y + top).max(top).min(bottom);
        let x1 = (d.x + left + d.w).max(left).min(right);
        let y1 = (d.y + top + d.h).max(top).min(bottom);
        let (w, h) = (x1 - x0, y1 - y0);
        if !(w > 0.0 && h > 0.0) {
            continue;
        }
        let n = (d.n_keypoints as usize).min(SYRUP_MAX_KEYPOINTS);
        let mut keypoints = [0.0f32; 2 * SYRUP_MAX_KEYPOINTS];
        for k in 0..n {
            keypoints[2 * k] = (d.keypoints[2 * k] + left).max(left).min(right);
            keypoints[2 * k + 1] = (d.keypoints[2 * k + 1] + top).max(top).min(bottom);
        }
        out.push(SyrupDetection {
            x: x0,
            y: y0,
            w,
            h,
            score: d.score,
            n_keypoints: n as u32,
            keypoints,
            payload: d.payload,
        });
    }
    out
}
"#
            }
            Helper::TieBreak => {
                r#"
fn tie_break(a: &SyrupDetection, b: &SyrupDetection) -> core::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then(a.y.total_cmp(&b.y))
        .then(a.x.total_cmp(&b.x))
        .then(a.h.total_cmp(&b.h))
        .then(a.w.total_cmp(&b.w))
}
"#
            }
            Helper::Emit => {
                r#"
fn emit(host: &SyrupHost, d: &SyrupDetection) -> Result<(), i32> {
    // SAFETY: the host copies the detection before returning.
    match unsafe { (host.emit)(host.ctx, d) } {
        SYRUP_OK => Ok(()),
        status => Err(status),
    }
}
"#
            }
        }
    }
}

fn frac(r: crate::intent::Ratio) -> String {
    format!("({}, {})", r.num, r.den)
}

fn primary_order(key: OrderKey) -> Option<&'static str> {
    Some(match key {
        OrderKey::ConfidenceDesc => return None,
        OrderKey::AreaDesc => "(b.w * b.h).total_cmp(&(a.w * a.h))",
        OrderKey::AreaAsc => "(a.w * a.h).total_cmp(&(b.w * b.h))",
        OrderKey::LeftToRight => "a.x.total_cmp(&b.x)",
        OrderKey::RightToLeft => "(b.x + b.w).total_cmp(&(a.x + a.w))",
        OrderKey::TopToBottom => "a.y.total_cmp(&b.y)",
        OrderKey::BottomToTop => "(b.y + b.h).total_cmp(&(a.y + a.h))",
    })
}

/// The module's source. The plan must have passed `Plan::check`.
pub fn generate(plan: &Plan, intent: &Intent, plan_hash: &str) -> String {
    let mut helpers = BTreeSet::new();
    let mut body = String::new();
    let b = &mut body;
    if plan
        .steps
        .iter()
        .any(|s| matches!(s, Step::FilterArea { .. }))
    {
        let _ = writeln!(
            b,
            "    let total = input.width as f64 * input.height as f64;"
        );
    }
    for (i, step) in plan.steps.iter().enumerate() {
        match *step {
            Step::Input => {
                helpers.insert(Helper::Window);
                let _ = writeln!(b, "    let v{i} = Window {{ view: *input, x: 0, y: 0 }};");
            }
            Step::SelectRegion {
                view,
                region: RegionSpec::Fixed { rect },
            } => {
                helpers.extend([Helper::Window, Helper::SelectFixed]);
                let _ = writeln!(
                    b,
                    "    let v{i} = select_fixed(&v{view}, {}, {}, {}, {});",
                    frac(rect.x),
                    frac(rect.y),
                    frac(rect.right()),
                    frac(rect.bottom())
                );
            }
            Step::SelectRegion {
                view,
                region: RegionSpec::Caller,
            } => {
                helpers.extend([Helper::Window, Helper::SelectCaller]);
                let _ = writeln!(b, "    let v{i} = select_caller(&v{view}, params)?;");
            }
            Step::Detect { view, capability } => {
                helpers.insert(Helper::Detect);
                let _ = writeln!(
                    b,
                    "    let v{i} = detect(host, {}, &v{view})?;",
                    capability.abi_id()
                );
            }
            Step::Restore { boxes, view } => {
                helpers.insert(Helper::Restore);
                let _ = writeln!(b, "    let v{i} = restore(v{boxes}, &v{view});");
            }
            Step::FilterConfidence { boxes } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.retain(|d| d.score >= params.min_confidence);");
            }
            Step::FilterArea {
                boxes,
                min_pct,
                max_pct,
            } => {
                let mut tests = vec![];
                if let Some(p) = min_pct {
                    tests.push(format!("area >= {p}_f64 * total"));
                }
                if let Some(p) = max_pct {
                    tests.push(format!("area < {p}_f64 * total"));
                }
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.retain(|d| {{");
                let _ = writeln!(b, "        let area = d.w as f64 * d.h as f64 * 100.0;");
                let _ = writeln!(b, "        {}", tests.join(" && "));
                let _ = writeln!(b, "    }});");
            }
            Step::Order { boxes, key } => {
                helpers.insert(Helper::TieBreak);
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                match primary_order(key) {
                    Some(primary) => {
                        let _ = writeln!(
                            b,
                            "    v{i}.sort_by(|a, b| {primary}.then_with(|| tie_break(a, b)));"
                        );
                    }
                    None => {
                        let _ = writeln!(b, "    v{i}.sort_by(tie_break);");
                    }
                }
            }
            Step::Limit { boxes, n } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.truncate({n});");
            }
            Step::LimitParam { boxes } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    if params.max_results > 0 {{");
                let _ = writeln!(b, "        v{i}.truncate(params.max_results as usize);");
                let _ = writeln!(b, "    }}");
            }
            Step::Emit { boxes } => {
                helpers.insert(Helper::Emit);
                let _ = writeln!(b, "    for d in &v{boxes} {{");
                let _ = writeln!(b, "        emit(host, d)?;");
                let _ = writeln!(b, "    }}");
            }
        }
    }

    let mut src = String::new();
    let _ = writeln!(
        src,
        "// Generated by syrup-runtime {} (codegen {CODEGEN_VERSION}).",
        env!("CARGO_PKG_VERSION")
    );
    let _ = writeln!(src, "// intent: {intent}");
    let _ = writeln!(src, "// plan: {plan_hash}");
    for line in plan.describe().lines() {
        let _ = writeln!(src, "//   {line}");
    }
    src.push_str("\n#![allow(dead_code, unused_variables)]\n#![deny(unsafe_op_in_unsafe_fn)]\n\n");
    let _ = writeln!(src, "pub mod abi {{\n{PRELUDE}}}\n\nuse abi::*;\n");
    let _ = writeln!(src, "const PLAN_HASH: &[u8] = b\"{plan_hash}\\0\";");
    src.push_str(ENTRY);
    let _ = write!(
        src,
        "\nfn run(host: &SyrupHost, input: &SyrupImageView, params: &SyrupParams) -> Result<(), i32> {{\n{body}    Ok(())\n}}\n"
    );
    for helper in helpers {
        src.push_str(helper.source());
    }
    src
}

const FORBIDDEN: &[&str] = &[
    "std::fs",
    "std::net",
    "std::process",
    "std::env",
    "std::os",
    "std::thread",
    "std::io",
    "core::arch",
    "asm!",
    "include!",
    "include_str!",
    "include_bytes!",
    "env!",
    "extern crate",
    "#[link",
    "extern \"C\" {",
    "extern \"system\"",
    "static mut",
    "export_name",
    "link_section",
    "#[no_mangle]",
];

/// Generated code may only compute over what the host hands it.
pub fn check_policy(source: &str) -> Result<()> {
    let policy = |reason: String| SyrupError::new(Stage::Generate, ErrorKind::Policy, reason);
    if let Some(token) = FORBIDDEN.iter().find(|t| source.contains(*t)) {
        return Err(policy(format!("generated source uses `{token}`")));
    }
    let exports: Vec<&str> = source
        .match_indices("#[unsafe(no_mangle)]")
        .filter_map(|(at, _)| {
            let rest = &source[at..];
            let name = &rest[rest.find("fn ")? + 3..];
            name.split('(').next()
        })
        .collect();
    if exports != EXPORTS {
        return Err(policy(format!(
            "generated source exports {exports:?}, expected {EXPORTS:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::parse;

    fn source(name: &str) -> String {
        let intent = parse(name).unwrap();
        let plan = Plan::build(&intent).unwrap();
        generate(&plan, &intent, &plan.hash())
    }

    #[test]
    fn generated_sources_pass_the_policy() {
        for name in [
            "find_face",
            "find_2_largest_faces_in_top_half_larger_than_2pct",
            "find_faces_in_region_left_to_right",
        ] {
            check_policy(&source(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn only_used_templates_are_emitted() {
        let plain = source("find_face");
        assert!(!plain.contains("fn select_fixed") && !plain.contains("fn select_caller"));
        let region = source("find_faces_in_top_half");
        assert!(
            region.contains("select_fixed(&v0, (0, 1), (0, 1), (1, 1), (1, 2))"),
            "{region}"
        );
    }

    #[test]
    fn the_policy_rejects_escapes() {
        let ok = source("find_face");
        for extra in [
            "\nfn x() { std::fs::remove_file(\"a\"); }",
            "\nunsafe extern \"C\" { fn system(); }",
            "\n#[unsafe(no_mangle)]\npub extern \"C\" fn other() {}",
        ] {
            assert_eq!(
                check_policy(&format!("{ok}{extra}")).unwrap_err().kind,
                ErrorKind::Policy,
                "{extra}"
            );
        }
    }
}
