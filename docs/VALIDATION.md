# Validation record

Validated locally on 2026-09-26. These checks establish implementation integrity, not a rerun of the benchmark tables.

| Check | Result |
|---|---|
| Unit/workflow suite | 13 passed |
| Detector parameter count | 6,983,784; backbone 5,585,600 |
| CWQR parameter count (six classes) | 468,506 |
| Legacy detector state dict | Strict loading succeeds; no missing/unexpected keys |
| Legacy detector forward | All four layers bit-exact on a 32³ CPU test input |
| Legacy CWQR state dict/forward | Strict loading succeeds; bit-exact with nonzero residual heads |
| Local / global objective parity | Local within 1e-6 relative tolerance; global bit-exact |
| Synthetic full workflow | Train detector → export → train CWQR → calibrate → predict/evaluate completed |
| Synthetic input/training | Three independent 96³ volumes; two classes; 11 detector epochs × 8 crops; effective batch 8; 40 CWQR epochs |
| Detector resume | Resume epoch 10 → 11 matches uninterrupted epoch 11 bit-for-bit |
| CWQR resume | Resume epoch 39 → 40 matches uninterrupted epoch 40 bit-for-bit |
| Packaging | Wheel built without fetching dependencies; installed in temporary target; CLI starts from outside source tree |

The regression suite covers cross-window exclusions and nearest-neighbor truncation; identity initialization and bounded coordinate updates; zero candidates and empty ground truth; nonempty-window center-loss reduction; one-to-one versus official CZII reference-coverage semantics; SHREC occupancy/exclusions; rotation/crop boundary coordinates; duplicate candidate IDs; invalid annotations; annotation-free prediction; frozen model identity; calibration/test overlap; and rejection of removed experimental configuration fields.

Environment: Python 3.10, PyTorch `2.3.0a0+6ddf5cf85e.nv24.04`, NVIDIA A100-SXM4-80GB. The preinstalled ONNX/protobuf combination required `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for optimizer construction. This was set only on validation processes; no environment changes or workarounds were embedded into model code.

Re-run portable checks:

```bash
python -m pytest -q
python examples/smoke.py --device cuda:0 --output /tmp/tomodetr-smoke-new
```

Re-run the optional legacy audit when the old research checkout is available:

```bash
python examples/verify_legacy_parity.py --legacy-root /path/to/legacy-checkout \
  --output /tmp/tomodetr-parity.json
```

The legacy checkout is deliberately not bundled and is not a runtime dependency. Detailed parity and resume results are recorded in `parity_validation.json` and `resume_validation.json`.

No benchmark dataset, published checkpoint, private inference output, or full 120-epoch training run was used for this validation. Reported paper metrics and timing are not asserted for the cleaned source. See `PAPER_ALIGNMENT.md` for the historical protocol discrepancies.

## Public release — 2026-10-15

The release passes 14 regression tests, including class-agnostic SHREC selection through the prediction pipeline and preservation of class-aware selection for the other datasets. SHREC uses a shared 10-voxel NMS radius.
