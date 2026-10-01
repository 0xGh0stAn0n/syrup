//! Text recognition for small on-screen UI text.
//!
//! The default backend is a Tesseract subprocess, located via
//! `TESSERACT_BIN`, `PATH`, or the standard Windows install locations. On
//! Windows the OS's built-in OCR engine is also available via
//! [`windows`] — it is trained on screen content and often beats
//! Tesseract on pixel-font UI text.

pub mod windows;

use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::atomic::{AtomicU64, Ordering};

use image::{DynamicImage, RgbaImage};

use crate::geometry::Rect;

/// OCR configuration for a single crop.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Page segmentation mode passed to Tesseract.
    pub psm: u8,
    /// Optional whitelist of characters to prefer.
    pub whitelist: Option<String>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            // PSM 0 only performs orientation detection and never recognizes text.
            // Sparse text handles interfaces where labels and values are separated.
            psm: 11,
            whitelist: None,
        }
    }
}

impl OcrConfig {
    /// Option flags only, without the leading positional arguments or the
    /// trailing config-file name.
    fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            "--oem".to_string(),
            "3".to_string(),
            "--psm".to_string(),
            self.psm.to_string(),
        ];
        if let Some(whitelist) = &self.whitelist {
            args.push("-c".to_string());
            args.push(format!("tessedit_char_whitelist={whitelist}"));
        }

        args.push("-c".to_string());
        args.push("preserve_interword_spaces=1".to_string());
        args
    }
}

/// Build the full Tesseract command line for one crop.
///
/// Tesseract's grammar is `tesseract IMAGE OUTPUTBASE [options...] [configfile...]`
/// and its parser stops reading options at the first non-flag argument that
/// follows the two positional ones. Putting the `tsv` config file before
/// `--oem`/`--psm`/`-c` makes Tesseract treat every flag as a config file name
/// ("read_params_file: Can't open --oem") and silently fall back to its default
/// page segmentation mode, so `tsv` has to come last.
fn tesseract_args(input: &Path, config: &OcrConfig) -> Vec<OsString> {
    let mut args = vec![input.as_os_str().to_os_string(), OsString::from("stdout")];
    args.extend(config.to_args().into_iter().map(OsString::from));
    args.push(OsString::from("tsv"));
    args
}

/// Owns a temporary OCR input image and deletes it on drop.
///
/// The Tesseract call has several fallible steps after the PNG is written; an
/// early return from any of them used to leak the file into the temp
/// directory. Tying deletion to the value's lifetime covers every exit path.
struct TempImage {
    path: PathBuf,
}

impl TempImage {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempImage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Clone, Copy)]
enum Preprocess {
    ContrastSharp,
}

/// A recognised word, in the coordinates of the image given to [`recognize`].
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub bounds: Rect,
    /// Tesseract's confidence, scaled to `[0, 1]`.
    pub confidence: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recognition {
    /// The words joined by spaces and normalised.
    pub text: String,
    pub words: Vec<Word>,
}

/// Why recognition did not run. Recognising no text is not an error: it is
/// an empty [`Recognition`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrError {
    /// No Tesseract on `TESSERACT_BIN`, `PATH` or the standard install paths.
    EngineMissing,
    /// The region does not overlap the image.
    EmptyRegion,
    /// The temporary input image could not be written.
    Io(String),
    /// Tesseract ran and failed.
    EngineFailed { status: Option<i32>, stderr: String },
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrError::EngineMissing => {
                f.write_str("Tesseract is not installed (set TESSERACT_BIN or add it to PATH)")
            }
            OcrError::EmptyRegion => f.write_str("the region does not overlap the image"),
            OcrError::Io(e) => write!(f, "cannot write the OCR input image: {e}"),
            OcrError::EngineFailed { status, stderr } => match status {
                Some(code) => write!(f, "Tesseract exited with status {code}: {stderr}"),
                None => write!(f, "Tesseract could not run: {stderr}"),
            },
        }
    }
}

impl std::error::Error for OcrError {}

/// Recognise the text in `region` of `image`. Word boxes come back in
/// `image`'s coordinates, whatever preprocessing happened in between.
pub fn recognize(
    image: &RgbaImage,
    region: Rect,
    config: &OcrConfig,
) -> Result<Recognition, OcrError> {
    let binary = find_tesseract_binary().ok_or(OcrError::EngineMissing)?;
    recognize_with(&binary, image, region, config)
}

fn recognize_with(
    binary: &Path,
    image: &RgbaImage,
    region: Rect,
    config: &OcrConfig,
) -> Result<Recognition, OcrError> {
    let run = run_tesseract(binary, image, region, config)?;
    let words: Vec<Word> = run
        .words
        .iter()
        .map(|w| Word {
            text: w.text.clone(),
            bounds: run.to_image(w),
            confidence: (w.confidence / 100.0).clamp(0.0, 1.0),
        })
        .collect();
    let text = normalize_text(
        &words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );
    Ok(Recognition { text, words })
}

/// OCR result for a single crop, from [`ocr_region`].
#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub available: bool,
    pub words: Vec<OcrWord>,
}

/// A word from [`ocr_region`], in the pixels of the upscaled crop that was
/// sent to Tesseract (not the crop itself).
#[derive(Debug, Clone)]
pub struct OcrWord {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Transitional: the original entry point, kept unchanged for existing
/// callers. It returns `None` for every failure as well as for no text, and
/// its word boxes are in the upscaled crop's pixels. Use [`recognize`].
pub fn ocr_region(image: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Option<OcrResult> {
    let binary = find_tesseract_binary()?;
    let run = run_tesseract(&binary, image, Rect { x, y, w, h }, &OcrConfig::default()).ok()?;
    let words: Vec<OcrWord> = run
        .words
        .into_iter()
        .map(|w| OcrWord {
            text: w.text,
            x: w.left,
            y: w.top,
            w: w.width,
            h: w.height,
        })
        .collect();
    let text = normalize_text(
        &words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );
    if text.trim().is_empty() {
        return None;
    }
    Some(OcrResult {
        text,
        available: true,
        words,
    })
}

/// A word as Tesseract reported it, in the preprocessed image's pixels.
struct TsvWord {
    text: String,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
    confidence: f32,
}

struct TesseractRun {
    words: Vec<TsvWord>,
    region: Rect,
    /// How much the crop was enlarged before recognition.
    factor: u32,
}

impl TesseractRun {
    fn to_image(&self, w: &TsvWord) -> Rect {
        let (crop_w, crop_h) = (self.region.w, self.region.h);
        let x0 = (w.left / self.factor).min(crop_w);
        let y0 = (w.top / self.factor).min(crop_h);
        let x1 = (w.left + w.width).div_ceil(self.factor).min(crop_w);
        let y1 = (w.top + w.height).div_ceil(self.factor).min(crop_h);
        Rect {
            x: self.region.x + x0,
            y: self.region.y + y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }
}

fn run_tesseract(
    binary: &Path,
    image: &RgbaImage,
    region: Rect,
    config: &OcrConfig,
) -> Result<TesseractRun, OcrError> {
    let crop =
        crop_region(image, region.x, region.y, region.w, region.h).ok_or(OcrError::EmptyRegion)?;
    let region = Rect {
        w: crop.width(),
        h: crop.height(),
        ..region
    };
    let (input_image, factor) = preprocess_image(&crop, Preprocess::ContrastSharp);
    // `input` deletes the PNG when it drops, on every path out of here.
    let input = write_temp_image(&input_image).map_err(|e| OcrError::Io(e.to_string()))?;
    let output = Command::new(binary)
        .args(tesseract_args(input.path(), config))
        .output()
        .map_err(|e| OcrError::EngineFailed {
            status: None,
            stderr: e.to_string(),
        })?;
    if !output.status.success() {
        return Err(OcrError::EngineFailed {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(TesseractRun {
        words: parse_tsv(&String::from_utf8_lossy(&output.stdout)),
        region,
        factor,
    })
}

/// Word rows of Tesseract's TSV output; other levels have no text.
fn parse_tsv(tsv: &str) -> Vec<TsvWord> {
    tsv.lines()
        .skip(1)
        .filter_map(|line| {
            let fields = line.split('\t').collect::<Vec<_>>();
            let text = fields.get(11)?.trim();
            if text.is_empty() {
                return None;
            }
            Some(TsvWord {
                text: text.to_string(),
                left: fields[6].parse().ok()?,
                top: fields[7].parse().ok()?,
                width: fields[8].parse().ok()?,
                height: fields[9].parse().ok()?,
                confidence: fields[10].parse().ok()?,
            })
        })
        .collect()
}

/// Check whether an OCR backend is available on the current machine.
pub fn is_ocr_available() -> bool {
    find_tesseract_binary().is_some()
}

fn crop_region(image: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Option<RgbaImage> {
    if w == 0 || h == 0 {
        return None;
    }
    let x_end = (x + w).min(image.width());
    let y_end = (y + h).min(image.height());
    if x_end <= x || y_end <= y {
        return None;
    }

    let mut crop = RgbaImage::new(x_end - x, y_end - y);
    for yy in 0..(y_end - y) {
        for xx in 0..(x_end - x) {
            let src_x = x + xx;
            let src_y = y + yy;
            crop.put_pixel(xx, yy, *image.get_pixel(src_x, src_y));
        }
    }
    Some(crop)
}

/// Crop height, in pixels, that OCR is given to work with.
///
/// On-screen UI text is often drawn 8-10 pixels tall, far below what
/// Tesseract is trained for, and at that size it returns near-noise.
/// Enlarging the crop first is what makes small UI text legible to it, so
/// crops are scaled up to roughly this height.
const MIN_OCR_TEXT_HEIGHT: u32 = 48;

/// The image to send to Tesseract, and how much it was enlarged.
fn preprocess_image(image: &RgbaImage, mode: Preprocess) -> (DynamicImage, u32) {
    let gray = DynamicImage::ImageRgba8(image.clone())
        .grayscale()
        .to_luma8();
    match mode {
        Preprocess::ContrastSharp => {
            let (image, factor) = upscale_for_ocr(DynamicImage::ImageLuma8(gray));
            (image.adjust_contrast(45.0).unsharpen(1.0, 1), factor)
        }
    }
}

/// Enlarge a crop so its text is tall enough for OCR, preserving aspect
/// ratio. Uses Lanczos3, which keeps thin glyph strokes intact where a
/// nearest-neighbour blow-up would leave them jagged and unreadable.
fn upscale_for_ocr(image: DynamicImage) -> (DynamicImage, u32) {
    let height = image.height();
    if height == 0 || height >= MIN_OCR_TEXT_HEIGHT {
        return (image, 1);
    }
    // Integer factors avoid resampling artefacts on pixel-art UI text.
    let factor = MIN_OCR_TEXT_HEIGHT.div_ceil(height).clamp(2, 8);
    let (width, height) = (
        image.width().saturating_mul(factor),
        height.saturating_mul(factor),
    );
    let resized = image.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    (resized, factor)
}

fn write_temp_image(image: &DynamicImage) -> io::Result<TempImage> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = format!(
        "syrup-ocr-{}-{}.png",
        process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    // Take ownership before writing: a failed save can still leave a partial
    // file on disk, and the guard cleans that up when this returns an error.
    let image_file = TempImage {
        path: env::temp_dir().join(name),
    };
    image.save(image_file.path()).map_err(io::Error::other)?;
    Ok(image_file)
}

fn find_tesseract_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("TESSERACT_BIN") {
        let candidate = PathBuf::from(path);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Ok(path) = env::var("PATH") {
        for entry in env::split_paths(&path) {
            let candidate = entry.join("tesseract.exe");
            if candidate.exists() {
                return Some(candidate);
            }
            let candidate = entry.join("tesseract");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // The standard Windows installer locations, which are not on PATH by
    // default. Anything else can be pointed at via TESSERACT_BIN.
    let candidates = [
        PathBuf::from(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        PathBuf::from(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn normalize_text(text: &str) -> String {
    let mut normalized = text
        .replace('\r', "")
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    normalized = normalized
        .chars()
        .map(|ch| match ch {
            '\u{2019}' | '\u{2018}' => '\'',
            '\u{2013}' | '\u{2014}' => '-',
            '\u{00A0}' => ' ',
            _ => ch,
        })
        .collect();
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn tsv_config_comes_after_option_flags() {
        let args = as_strings(&tesseract_args(
            Path::new("crop.png"),
            &OcrConfig {
                psm: 11,
                whitelist: Some("0123456789".to_string()),
            },
        ));

        assert_eq!(args[0], "crop.png");
        assert_eq!(args[1], "stdout");
        // Tesseract stops parsing options at the first config-file argument,
        // so every flag must precede `tsv` or it is read as a config name.
        assert_eq!(args.last().map(String::as_str), Some("tsv"));
        let tsv = args.iter().position(|arg| arg == "tsv").unwrap();
        for flag in ["--oem", "--psm", "-c"] {
            let at = args.iter().position(|arg| arg == flag).unwrap();
            assert!(at < tsv, "{flag} must come before the tsv config file");
        }
        assert!(
            args.iter()
                .any(|arg| arg == "tessedit_char_whitelist=0123456789")
        );
    }

    const TSV_HEADER: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext";

    /// A stand-in for Tesseract that prints `tsv` (or fails, without one).
    #[cfg(unix)]
    fn fake_tesseract(name: &str, tsv: Option<&str>) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = env::temp_dir().join(format!("syrup-fake-tesseract-{}-{name}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        let body = match tsv {
            Some(tsv) => {
                fs::write(dir.join("out.tsv"), tsv).unwrap();
                format!("cat '{}'", dir.join("out.tsv").display())
            }
            None => "echo 'cannot read the image' >&2; exit 1".to_string(),
        };
        let script = dir.join("tesseract");
        fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    #[test]
    fn only_rows_with_text_are_words() {
        let tsv = format!(
            "{TSV_HEADER}\n1\t1\t0\t0\t0\t0\t0\t0\t90\t40\t-1\t\n5\t1\t1\t1\t1\t1\t8\t4\t30\t12\t91.5\tHP\n"
        );
        let words = parse_tsv(&tsv);
        assert_eq!(words.len(), 1);
        assert_eq!(
            (words[0].text.as_str(), words[0].left, words[0].confidence),
            ("HP", 8, 91.5)
        );
    }

    #[cfg(unix)]
    #[test]
    fn word_boxes_come_back_in_image_coordinates() {
        // A 12-pixel-tall crop is enlarged 4x for Tesseract, so a word it
        // reports at (40, 8) 80x32 sits at (10, 2) 20x8 within the crop.
        let tsv = format!("{TSV_HEADER}\n5\t1\t1\t1\t1\t1\t40\t8\t80\t32\t88\tHELLO\n");
        let binary = fake_tesseract("words", Some(&tsv));
        let image = RgbaImage::new(300, 300);
        let region = Rect {
            x: 100,
            y: 200,
            w: 50,
            h: 12,
        };
        let result = recognize_with(&binary, &image, region, &OcrConfig::default()).unwrap();
        assert_eq!(result.text, "HELLO");
        assert_eq!(
            result.words[0].bounds,
            Rect {
                x: 110,
                y: 202,
                w: 20,
                h: 8
            }
        );
        assert!((result.words[0].confidence - 0.88).abs() < 1e-6);
    }

    #[cfg(unix)]
    #[test]
    fn no_text_is_an_empty_recognition_and_failure_is_an_error() {
        let image = RgbaImage::new(40, 40);
        let region = Rect {
            x: 0,
            y: 0,
            w: 40,
            h: 40,
        };
        let silent = fake_tesseract("silent", Some(TSV_HEADER));
        let empty = recognize_with(&silent, &image, region, &OcrConfig::default()).unwrap();
        assert_eq!(empty, Recognition::default());

        let broken = fake_tesseract("broken", None);
        let error = recognize_with(&broken, &image, region, &OcrConfig::default()).unwrap_err();
        assert_eq!(
            error,
            OcrError::EngineFailed {
                status: Some(1),
                stderr: "cannot read the image".to_string()
            }
        );
    }

    #[test]
    fn a_region_off_the_image_is_an_error() {
        let image = RgbaImage::new(10, 10);
        let region = Rect {
            x: 20,
            y: 20,
            w: 5,
            h: 5,
        };
        let error = recognize_with(Path::new("unused"), &image, region, &OcrConfig::default());
        assert_eq!(error, Err(OcrError::EmptyRegion));
    }

    #[test]
    fn temp_image_is_removed_on_drop() {
        let image = DynamicImage::ImageRgba8(RgbaImage::new(4, 4));
        let path = {
            let temp = write_temp_image(&image).expect("temp image written");
            let path = temp.path().to_path_buf();
            assert!(path.exists());
            path
        };
        assert!(!path.exists(), "temp OCR image outlived its guard");
    }
}
