# Architecture

The contract this design serves is in [contract.md](contract.md).

## Starting point

`main` (upstream `265ade5`) is a ~3,300-line Rust crate of deterministic
pixel primitives: geometry, colour, motion, tracking, Tesseract OCR,
sharpness, Windows capture and debug drawing. It has 45 passing tests and no
bindings, build machinery or models. Its shared result type is
`Detection<T>`.

Upstream also has two unmerged experiments, `claude/intents` and
`claude/intents-2`. Their planner is a closed enum whose variants each call a
handwritten `plans::*` function, and the "generated" crate is a one-line call
to that function built with cargo. That is the name-to-function mapping this
work replaces, so it starts from `main`.

## What stays

The core crate stays at the repository root, unchanged in this first step.
MapleSyrup consumes it as a submodule, and its primitives become
capabilities the runtime composes.

## What is added

```text
from syrup.ops import find_face        Runtime::resolve("find_face")
            └──────────────┬──────────────────┘
                           ▼
 crates/syrup-runtime
   intent.rs    name → canonical Intent, or a refusal that says why
   plan.rs      Intent → typed Plan; every value carries its coordinate space
   interp.rs    reference semantics, used only to validate modules
   codegen.rs   Plan → dependency-free Rust source (one template per step)
   compiler.rs  rustc directly: no cargo, no build scripts, no network
   cache.rs     content-addressed store, locked builds, atomic publish
   loader.rs    load by absolute path, check ABI version and plan hash
   validate.rs  module vs interpreter on synthetic images
   providers/   YuNet face detection via tract, behind the C ABI
 crates/syrup-python + python/   one generic binding, errors, results
```

## Decisions

1. **The generated module is the operation.** Region selection, the detector
   call, coordinate restoration, clipping, filtering, ordering and limiting
   are emitted from the plan. Only learned or external detectors (YuNet,
   Tesseract), the core's region grouping and the core's measurements
   (sharpness, bar fill) live in the host. For colour
   targets the per-pixel test itself is generated, specialised to the
   colour, and validated against the core's `is_color_pixel`.
2. **Generated code has no dependencies.** The ONNX runtime and the model
   are compiled into the host once, so a module builds with `rustc` in about
   half a second and never downloads anything.
3. **Names are parsed, not looked up.** Clauses compose, so
   `find_2_largest_faces_in_top_half` works without anyone writing it. Names
   with the same meaning share a plan, so they share one artifact.
4. **Validation is differential.** Before an artifact is published it runs on
   synthetic images against a mock detector and must match the interpreter.
   The mock reads the window it was given from the view pointer, so a crop
   at the wrong offset fails validation rather than production.
5. **Python resolution is real.** `syrup.ops` uses module `__getattr__`
   (PEP 562), so `from syrup.ops import find_face` resolves at import and
   compiles on first call. A bare name that was never imported cannot be
   intercepted honestly, so it is not attempted. Names outside the grammar
   go through `syrup.define`.
6. **State lives in sessions, not modules.** A generated module looks at one
   frame. `track_*` sessions run it per frame and keep what spans frames in
   the host: the core's tracker, and for moving regions the previous frame,
   served to the module as a capability like any detector.
7. **New detectors plug in without Rust.** `syrup.add_target(noun, detect)`
   registers a Python function (any ML library) as a provider; names using
   the noun compile to modules that call it through the same ABI.
8. **The model is a pinned dependency of the runtime.** YuNet 2023mar (MIT,
   232 KB, SHA-256 checked) runs through `tract-onnx`: pure Rust, no OpenCV.
   The core crate stays model-free.

## Delivery

1. Mechanism and the `find_face` proof: runtime, YuNet provider, Python
   bridge, contract, tests.
2. Migrate the existing core: OCR gets explicit errors and image-coordinate
   word boxes; existing primitives become catalog capabilities; old entry
   points stay as marked adapters.
3. Measurements, sessions for motion and tracking, prepared bundles, wheels
   for Linux, macOS and Windows.
4. Later: more providers, and an optional language-model planner that may
   only emit typed plans.
