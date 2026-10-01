//! `ocr_region` is kept as a transitional adapter for existing callers. These
//! pin its behaviour: words in the upscaled crop's pixels, `None` for no text
//! and for failures. It runs against a fake Tesseract via TESSERACT_BIN, so
//! it needs no installed engine.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use image::RgbaImage;
use syrup::geometry::Rect;
use syrup::ocr::{OcrConfig, OcrError, ocr_region, recognize};

fn fake_tesseract(dir: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syrup-ocr-adapter-{}-{dir}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let script = dir.join("tesseract");
    fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    script
}

// One test, because TESSERACT_BIN is process-wide.
#[test]
fn ocr_region_keeps_its_old_behaviour() {
    let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
               5\t1\t1\t1\t1\t1\t40\t8\t80\t32\t88\tHP\n\
               5\t1\t1\t1\t1\t2\t130\t8\t60\t32\t75\t1291/1351\n";
    let tsv_path =
        std::env::temp_dir().join(format!("syrup-ocr-adapter-{}.tsv", std::process::id()));
    fs::write(&tsv_path, tsv).unwrap();
    let image = RgbaImage::new(200, 100);
    let region = Rect {
        x: 10,
        y: 20,
        w: 60,
        h: 12,
    };

    let reads = fake_tesseract("reads", &format!("cat '{}'", tsv_path.display()));
    // SAFETY: the only test in this binary, so nothing else reads the environment.
    unsafe { std::env::set_var("TESSERACT_BIN", &reads) };
    let old = ocr_region(&image, region.x, region.y, region.w, region.h).unwrap();
    assert_eq!(old.text, "HP 1291/1351");
    assert!(old.available);
    let first = &old.words[0];
    assert_eq!(
        (first.x, first.y, first.w, first.h),
        (40, 8, 80, 32),
        "upscaled crop pixels"
    );

    let new = recognize(&image, region, &OcrConfig::default()).unwrap();
    assert_eq!(new.text, old.text);
    assert_eq!(
        new.words[0].bounds,
        Rect {
            x: 20,
            y: 22,
            w: 20,
            h: 8
        },
        "image pixels"
    );

    let silent = fake_tesseract("silent", "echo 'level\tpage_num'");
    unsafe { std::env::set_var("TESSERACT_BIN", &silent) };
    assert!(ocr_region(&image, region.x, region.y, region.w, region.h).is_none());
    assert_eq!(
        recognize(&image, region, &OcrConfig::default())
            .unwrap()
            .words,
        vec![]
    );

    let broken = fake_tesseract("broken", "echo 'oops' >&2; exit 3");
    unsafe { std::env::set_var("TESSERACT_BIN", &broken) };
    assert!(ocr_region(&image, region.x, region.y, region.w, region.h).is_none());
    assert_eq!(
        recognize(&image, region, &OcrConfig::default()),
        Err(OcrError::EngineFailed {
            status: Some(3),
            stderr: "oops".into()
        })
    );
}
