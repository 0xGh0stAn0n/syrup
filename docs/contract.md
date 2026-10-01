# Contract

What every `find_*` operation honours, whichever name or language reached it.

## 1. How names reach Syrup

| | resolves | compiles |
|---|---|---|
| `from syrup.ops import find_face` / `syrup.ops.find_face` (PEP 562 module `__getattr__`) | on import or attribute access | on first call, or `op.prepare()` |
| `syrup.resolve("find_face")` | on the call | on first call, or `op.prepare()` |
| `syrup.define("find_header_faces", ...)`, for names outside the grammar | on the call | on first call, or `op.prepare()` |
| Rust: `Runtime::resolve` / `Runtime::define` | on the call | on first `run`, or `prepare()` |

Resolution never compiles, so a name Syrup cannot honour fails at import,
before any image is involved. A name that was never imported or defined is
an ordinary `NameError`; Python has no honest hook for that.

## 2. Grammar (v1)

```text
name      := verb "_" [count "_" selector "_" | selector "_" | "all_"] target ("_" clause)*
verb      := find | detect | locate
target    := face | faces | human_face | human_faces
selector  := largest | biggest | smallest | leftmost | rightmost | topmost | bottommost | most_confident
count     := 1..100, or one..ten
clause    := in_<region> | by_size | by_area | by_confidence | by_score
           | left_to_right | right_to_left | top_to_bottom | bottom_to_top
           | larger_than_<n>pct | smaller_than_<n>pct        (n = 1..100, % of image area)
region    := top_half | bottom_half | left_half | right_half
           | top_left | top_right | bottom_left | bottom_right
           | top_third | bottom_third | left_third | right_third | center | region
```

Singular and plural mean the same: `find_face` returns every face, so
`faces = find_face(img)` reads right. One result is spelled with a selector:
`find_largest_face`. Names that differ only by synonym, number or clause
order resolve to the same intent and share one compiled artifact.

Regions are fractions `(fx, fy, fw, fh)` of the image, converted with
`x0 = round(fx·W)`, `x1 = round((fx + fw)·W)` (halves round up), so adjacent
regions tile the image exactly. Halves, quadrants and thirds are what their
names say; `center` is `(¼, ¼, ½, ½)`; `region` is passed per call as pixels
`(x, y, w, h)` and clipped to the image.

`in_<region>` means the detector only sees those pixels. Results are in
whole-image coordinates, clipped to the region; an object cut by the region
edge may be found partially or not at all.

## 3. Results

Every `find_*` returns a `FindResult`: an ordered, possibly empty sequence
of `Found` plus provenance. It is a sequence even when a selector limits it
to one item.

- `label`: the target, e.g. `"face"`.
- `box`: `(x, y, w, h)` floats in pixels of the image passed in. Origin at
  the top-left corner of the top-left pixel, x right, y down, covering
  `[x, x+w) × [y, y+h)`, clipped to the image and to the region.
- `confidence`: the provider's score in `[0, 1]`. Not a calibrated
  probability; only comparable within one provider and model, which
  provenance names.
- `keypoints`: named points, same coordinates, clipped. Faces carry
  `right_eye`, `left_eye`, `nose_tip`, `right_mouth_corner`,
  `left_mouth_corner` (the subject's right and left).

Default order is confidence, highest first. Every order ends with the same
tie-breakers (confidence ↓, then y, x, h, w ↑), so it is total.

| clause (all results) | selector (one result) | order |
|---|---|---|
| none, `by_confidence`, `by_score` | `most_confident` | confidence ↓ |
| `by_size`, `by_area` | `largest`, `biggest` | area ↓ |
| | `smallest` | area ↑ |
| `left_to_right` | `leftmost` | x ↑ |
| `right_to_left` | `rightmost` | right edge ↓ |
| `top_to_bottom` | `topmost` | y ↑ |
| `bottom_to_top` | `bottommost` | bottom edge ↓ |

A count before a selector (`find_3_largest_faces`) keeps that many instead
of one. Area filters run before
ordering and limits.

Per-call parameters never trigger a rebuild:

| | default | |
|---|---|---|
| `min_confidence` | faces 0.6 | in `[0, 1]`; the face provider never reports below 0.1 |
| `max_results` | none | ≥ 1, applied after the operation's own limit |
| `region` | | required by `_in_region` operations, refused by all others |

**Empty is not failure.** An empty result means the operation ran and the
detector accepted nothing under the recorded configuration; it does not
prove the image has no faces. Anything that stops an operation from running
completely raises. Syrup never returns a success-shaped result for a failed
run, and never swaps in another operation, provider or model.

## 4. Inputs

8-bit images with 1, 3 (RGB) or 4 (RGBA, alpha ignored) channels, row-major,
1 to 16384 pixels per side. Python accepts NumPy `uint8` arrays shaped
`(H, W)`, `(H, W, 1)`, `(H, W, 3)` or `(H, W, 4)`; PIL images (`L`, `RGB`,
`RGBA`; palette images are converted to RGB); image file paths; and
`syrup.Image`. Channel order is RGB. OpenCV's BGR must be converted by the
caller, because Syrup cannot tell the difference. Anything else is an
`InputError`, not a coercion.

## 5. Failures

Every failure is a `SyrupError` with `stage`, `kind`, `operation`, `reason`
and `hint`. Python raises `DependencyError` for `missing_dependency`, and
otherwise the class for the stage.

| stage | class | kinds |
|---|---|---|
| `input` | `InputError` | `bad_image`, `bad_parameter` |
| `resolve` | `IntentError` | `malformed`, `ambiguous`, `unsupported`, `conflicting` |
| `plan` | `PlanError` | `invalid_plan` |
| `generate`, `compile` | `BuildError` | `policy`, `compiler_failed`, `timeout`, `missing_dependency` |
| `load` | `LoadError` | `not_prepared`, `integrity`, `abi_mismatch`, `dlopen` |
| `validate` | `ValidationError` | `mismatch` |
| `execute` | `ExecutionError` | `provider_failed`, `module_failed`, `panic`, `contract_violation`, `missing_dependency`, `integrity` |

Refused rather than guessed:

| request | kind | why |
|---|---|---|
| `find_best_face`, `find_main_face`, `find_first_face` | ambiguous | best, main or first by what? |
| `find_large_faces` | ambiguous | large compared to what? Use `larger_than_<n>pct` |
| `find_highest_face` | ambiguous | by position or by confidence? |
| `find_two_faces`, `find_largest_faces` | ambiguous | a count needs an ordering; a plural selector needs a count |
| `find_smiling_faces`, `find_cat_faces` | unsupported | no capability judges that |
| `find_cars`, `find_people` | unsupported | no capability finds that |
| `find_faces_in_top_half_in_left_half` | conflicting | one region per operation |
| any unknown word | malformed or unsupported | unknown words are never dropped |

## 6. Artifacts

An artifact is identified by its plan, code generator version, ABI version,
target triple and compiler flags; never by the operation name. The
compiler's version is recorded but not part of the key, since modules only
speak C ABI.

Artifacts are built in a private directory, validated, and published with
one rename; concurrent first use builds once; published artifacts never
change, and the library's SHA-256 is checked before every load. Damaged
artifacts are moved aside and rebuilt in development mode, and refused in
frozen mode (`SYRUP_MODE=frozen`), which never generates or compiles.

An artifact is published only after:

1. the plan type-checks, with results in input-image coordinates;
2. the source passes a policy check (no filesystem, network, process,
   environment or foreign-code access; only the three ABI exports);
3. `rustc` builds it with warnings denied;
4. the library exports the expected ABI version and plan hash;
5. on 24 synthetic cases its calls and results match the interpreter's
   exactly.

## 7. Provenance

Each result records the requested name, canonical intent, plan hash,
artifact key, library SHA-256, `rustc` version, target, whether this call
compiled the artifact or reused it from memory or disk, each provider and
model hash, the effective parameters, the image shape, and timings.
