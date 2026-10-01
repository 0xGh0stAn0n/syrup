# syrup-cv

Computer-vision operations you name instead of write.

```python
from syrup.ops import find_largest_face, find_red_bars_in_bottom_third

faces = find_largest_face("photo.jpg")
for face in faces:
    print(face.box, face.confidence)
```

The name is parsed into a plan, compiled to a small native module the
first time it runs, checked against a reference implementation, and reused
afterwards. Names Syrup cannot honour raise `syrup.IntentError` at import.

Compiling needs `rustc` 1.82 or newer. Machines without it can run
operations prepared elsewhere:

```python
syrup.bundle("ops-bundle", "find_largest_face", "find_words")
# then, on the target machine: SYRUP_MODE=frozen SYRUP_CACHE_DIR=ops-bundle
```

Detectors from other libraries plug in with `syrup.add_target`. The grammar,
result contract and failure classes are in
[docs/contract.md](https://github.com/scp-labs/C4/blob/main/docs/contract.md).
