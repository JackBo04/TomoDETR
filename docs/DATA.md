# Data and coordinate contract

## Portable representation

- Volumes: `.npy` or MRC, array order `(D,H,W)=(z,y,x)`. Every axis must be at least 96 voxels. MRC headers do not automatically reorder axes or rescale coordinates.
- Annotations: CSV `z,y,x,class_id`, zero-based class IDs, continuous detector-voxel coordinates in `[0,S-1]`. An empty file with only the header is valid.
- Whole-volume mean/std normalization is reused for every crop. Nonfinite or constant images fail explicitly.
- Crops use the closed voxel-center extent `[0,95]`. Rotation and reflection apply to both images and points.
- Store all split roles in one manifest when possible; duplicate IDs and reused volume paths are rejected. Dataset content hashes tie caches to inputs and calibrated thresholds to model weights.

`examples/convert_annotations.py` converts CSV point annotations from an explicitly specified physical voxel spacing. It does not infer missing units or label conventions. Convert native annotation containers to CSV with named columns first; class IDs must already match the dataset YAML.

```bash
python examples/convert_annotations.py --input raw_xyz.csv --output detector_zyx.csv \
  --input-spacing 4.99 --detector-spacing 10.012444196428572
```

If the input is already in Ångström, use `--input-spacing 1`. If it is already in detector voxels, use equal spacings. Source CSV column order is irrelevant because columns are read by name.

## CZII

Class order is apo-ferritin, beta-amylase, beta-galactosidase, ribosome, thyroglobulin, virus-like-particle. Physical radii are `(60,65,90,150,130,135)` Å; metric weights are `(1,0,2,1,2,1)`. Detector spacing is `10.012444196428572` Å in the original source.

DS-10445/10446 annotation indices use a 4.99 Å grid. Convert with the exact ratio `4.99/10.012444196428572`, not `/2`. Copick DS-10440 coordinates expressed in Å are divided directly by the detector spacing.

The manuscript partition is seven DS-10440 training volumes, 40 DS-10445 validation volumes, and 364 DS-10446 test volumes. The seven training IDs are `TS_5_4`, `TS_6_4`, `TS_6_6`, `TS_69_2`, `TS_73_6`, `TS_86_3`, `TS_99_9`. The package does not substitute old Phantom20/Disjoint40 manifests for the current Val40 split. Supply the benchmark's actual validation/test lists explicitly; examples are not those lists.

CZII evaluation reproduces the official reference-coverage behavior, not strict bipartite matching. A prediction can cover multiple extremely close references; this can produce a negative FP count in that edge case, as in the official local scorer. Tests intentionally preserve this behavior. Counts are pooled across volumes before weighted macro F4. Thresholds use the historical coarse-to-fine Val40 search.

## DeePiCt

Voxel spacing: 13.48 Å. Class order: cytosolic ribosome, fatty-acid synthase complex. Reference radii: 10 voxels; NMS radius: 5 voxels; strict evaluation tolerance: 10 voxels.

Train and calibrate each imaging condition independently. Use seeds 1729, 1730, 1731 for splits 0, 1, 2. The YAML starts at seed 1729; create a separate YAML/output tree per split. IDs below have the common `TS_` prefix omitted.

| Condition | Split | Train | Validation/calibration | Test |
|---|---|---|---|---|
| VPP | 0 | 0001,0003,0004,0005,0009,0010 | 0006,0007 | 0002,0008 |
| VPP | 1 | 0001,0002,0003,0005,0008,0010 | 0009,0004 | 0006,0007 |
| VPP | 2 | 0002,0003,0005,0006,0007,0008 | 0010,0001 | 0009,0004 |
| Defocus | 0 | 026,028,029,030,043,045 | 034,037 | 027,041 |
| Defocus | 1 | 026,027,028,030,041,045 | 043,029 | 034,037 |
| Defocus | 2 | 027,028,030,034,037,041 | 045,026 | 043,029 |

Within a split all three roles are disjoint. No VPP-to-defocus threshold transfer or calibration on the training set is used. Match counts maximize valid one-to-one pairs within class; duplicates are false positives. Pooled F1 combines the two classes and both test volumes.

## SHREC2021

Use the class order and particle radii in `configs/shrec.yaml`. `4V94` is excluded by the evaluator configuration; predictions labeled vesicle are excluded from official submissions; fiducials are retained. The official-equivalent scorer uses the organizer's dilated instance occupancy map, **not** class-aware radius matching. It preserves the historical denominator convention where vesicles remain in the ground-truth count.

Development manifest: `model_0`–`model_7` as `train`, `model_8` as `validation`, `model_9` as `test`. For calibration/evaluation, each record additionally needs `particle_locations` and `occupancy` paths to the original `particle_locations.txt` and `occupancy_mask.mrc`. Their particle row order/instance IDs must remain unchanged; do not filter the particle table before occupancy scoring. The annotation CSV for detector/CWQR training is a separate point representation.

After selecting the development settings, create a refit manifest where `model_0`–`model_8` are `train`, and `model_9` remains `test`. Retrain the detector and CWQR, then carry the already-frozen model-8 thresholds to the refit checkpoints without recalibration:

```bash
python -m tomodetr bind-shrec-refit --config configs/shrec.yaml \
  --calibration outputs/development_calibration/calibration.json \
  --detector outputs/refit_detector/latest.pt --cwqr outputs/refit_cwqr/latest.pt \
  --output outputs/refit_calibration
```

Use this calibration file for refit test prediction. The command requires the same selected configuration and exactly nine refit training IDs. It reads no test annotations. Retain the original development calibration file as provenance.

The release follows the manuscript's class-aware NMS. The difference from historical class-agnostic SHREC postprocessing is explicitly documented in `PAPER_ALIGNMENT.md`.
