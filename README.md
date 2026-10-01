# Syrup

A small Rust toolkit for turning captured frames into structured,
confidence-scored visual observations.

Syrup gives you the pixel-level building blocks for reading a screen the
way a person does — "there is a bar here and it is about 60% full", "that
region moved left", "this text says 1291/1351" — without pretending to more
certainty than the pixels support. Every detector result carries a
confidence score, a reliability grade, and a failure reason when nothing was
found.

## Operations by name

Name the operation you want and Syrup builds it:

```python
from syrup.ops import find_largest_face_in_top_half

faces = find_largest_face_in_top_half("photo.jpg")
for face in faces:
    print(face.box, face.confidence, face.keypoints["left_eye"])
```

Nobody wrote `find_largest_face_in_top_half`. The name is parsed into an
intent, the intent into a typed plan, and the plan into a small Rust module
that is compiled with `rustc`, checked against a reference interpreter,
cached, and loaded. Later calls and later processes reuse the compiled
module. Names Syrup cannot honour fail at import with a reason:
`find_best_face` is refused because "best" does not say by what.

The same mechanism is available from Rust (`syrup_runtime::Runtime`) and
from the command line:

```sh
syrup explain find_2_largest_faces_in_center   # what the name means
syrup source find_face                         # the generated module
syrup run find_faces photo.jpg --draw out.png  # JSON results
```

[docs/contract.md](docs/contract.md) has the grammar, the result contract
and every way an operation can fail; [docs/architecture.md](docs/architecture.md)
the design.

## What it does

- **Geometry** — rectangle segmentation and grouping over pixel predicates,
  and horizontal-bar fill measurement that learns the bar's empty-track
  color from the frame instead of assuming it.
- **Color** — RGB→HSV conversion and the shared pixel predicates (hue-range
  match, "looks like UI text", opacity).
- **Motion** — single-pass frame differencing plus a centroid tracker that
  gives moving regions stable IDs, velocity, and occlusion grace.
- **OCR** — text recognition of small on-screen UI text via a Tesseract
  subprocess, with automatic crop upscaling (small pixel fonts are
  otherwise unreadable to it). `ocr::recognize` returns words in image
  coordinates and says why it failed; the older `ocr_region` is kept for
  existing callers. On Windows, the OS's built-in OCR engine is also
  exposed, which is trained on screen content.
- **Quality** — a sharpness metric that predicts whether OCR on a region
  can succeed at all, so blurred input is reported as *blurred* rather than
  silently producing wrong text.
- **Capture** — live window capture by title on Windows (works while the
  window is occluded); portable stubs elsewhere.
- **Debug drawing** — rectangles and a dependency-free 5×7 bitmap font for
  annotating frames with what a detector saw.

## What it deliberately does not do

- No input synthesis, no window manipulation, no process inspection: the
  library **reads pixels and reports observations**, nothing else.
- No trained models in the core: every primitive is deterministic and
  explainable, which keeps results reproducible in tests. Learned
  capabilities (YuNet face detection, pinned by SHA-256) live in
  `syrup-runtime`.
- No opinion about what an observation *means* — semantics belong to the
  application built on top.

## Example

```rust
use syrup::color::is_color_pixel;
use syrup::geometry::{Rect, find_color_bar, measure_bar_fill};

let image: image::RgbaImage = image::open("screen.png")?.to_rgba8();

// Look for a red horizontal bar in the bottom band of the screen…
let band = Rect { x: 0, y: image.height() * 9 / 10, w: image.width(), h: image.height() / 10 };
let red = |p: &image::Rgba<u8>| is_color_pixel(p, (340.0, 30.0), 0.35, 0.30);

if let Some(bar) = find_color_bar(&image, band, (340.0, 30.0), 0.35, 0.30) {
    // …and measure how full it is against its own track.
    if let Some(percent) = measure_bar_fill(&image, bar, band, red) {
        println!("bar at {bar:?} is {percent:.1}% full");
    }
}
# Ok::<(), image::ImageError>(())
```

Frames are plain `image::RgbaImage` buffers, so they can come from a
screenshot, frames extracted from a video, a synthetic fixture in a test,
or `syrup::capture` — every primitive behaves identically regardless of
the source.

## Architecture

```text
RgbaImage (any source)
    │
    ├─ geometry / color   locate regions by shape and color
    ├─ motion / tracking  what moved, with stable identity
    ├─ ocr / quality      what text says, and whether it is readable at all
    │
    ▼
Detection<T> — value + confidence + reliability + failure reason
```

The `Detection<T>` vocabulary is the library's one contract: a detector
never returns a bare "not found" — it says *why* not, and never returns a
value without saying *how sure* it is.

## Testing

```sh
cargo test --workspace   # core, runtime, and end-to-end face detection
cargo clippy --workspace --all-targets -- -D warnings
cargo bench              # criterion benchmarks for the per-frame primitives

cd python && maturin develop --release && pytest tests
```

The core's tests run against synthetic, in-code fixtures. The runtime's
compile real modules, so they need `rustc` on the PATH, and detect faces in
three public-domain and CC0 photographs in `crates/syrup-runtime/tests/fixtures`.
No network access is needed. OCR tests cover argument construction and
temp-file hygiene without invoking Tesseract. Nothing in the suite depends
on a domain edition, so this repository stands alone.

## Limitations

- The primitives are tuned for rendered UI content (flat colors, pixel
  fonts, hard edges), not for photographs or video of natural scenes.
- Tesseract must be installed separately for OCR (`TESSERACT_BIN` or
  `PATH`); without it, OCR reports itself unavailable rather than failing.
- Live capture is Windows-only. Other platforms consume file-based frames.
- Operations by name currently cover faces and printed words, and compiling
  a new operation needs `rustc` where it first runs (or a cache prepared
  elsewhere, with `SYRUP_MODE=frozen`).

## Syrup and MapleSyrup

Syrup is the generic engine. A domain edition consumes it and adds the
knowledge Syrup deliberately lacks — what the pixels *mean* in one
particular application:

```text
        MapleSyrup            the MapleStory edition
             │                github.com/boggioMichael/ms
             │ submodule
             ▼
           Syrup              this repository
```

[MapleSyrup](https://github.com/boggioMichael/ms) is the first such
edition, and is where these primitives were developed against real
captures before being generalised. Syrup itself knows nothing about
MapleStory, or any other application — a second edition for a different
program would consume it exactly the same way.

## License

MIT
