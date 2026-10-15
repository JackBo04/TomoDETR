# Getting started

## Install

Python 3.10+ and PyTorch 2.3+:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m tomodetr --help
```

A CUDA-enabled PyTorch installation is recommended for training. No custom CUDA extensions, downloaded weights, external repositories, or dense segmentation labels are required.

## Data

Provide a JSON manifest containing a `.npy` or MRC tomogram and a CSV annotation file for each record:

```json
[
  {"id": "volume_a", "volume": "data/a.mrc", "annotations": "data/a.csv", "split": "train"},
  {"id": "volume_b", "volume": "data/b.mrc", "annotations": "data/b.csv", "split": "validation"},
  {"id": "volume_c", "volume": "data/c.mrc", "split": "test"}
]
```

CSV columns are `z,y,x,class_id`; classes are zero-based in the configuration order. Coordinates must already be converted to the detector voxel grid. Paths are relative to the manifest. Exact duplicate annotation rows are removed. Prediction and export do not open annotation files. To evaluate predictions, include test annotations and pass `--evaluate` explicitly.

See [data preparation](DATA.md) for benchmark partitions, physical coordinate conversion, SHREC occupancy files, and DeePiCt split lists. Example manifests are schema examples, not actual dataset splits.

## Train and predict

Choose `configs/czii.yaml`, `configs/shrec.yaml`, or `configs/deepict.yaml`. The following commands use CZII as an example. Each output directory must be new or empty.

```bash
python -m tomodetr train-detector --config configs/czii.yaml \
  --manifest data.json --output outputs/detector --device cuda:0

python -m tomodetr export --config configs/czii.yaml \
  --manifest data.json --detector outputs/detector/latest.pt \
  --output cache --device cuda:0

python -m tomodetr train-cwqr --config configs/czii.yaml \
  --manifest data.json --cache cache --output outputs/cwqr --device cuda:0

python -m tomodetr calibrate --config configs/czii.yaml \
  --manifest data.json --cache cache --cwqr outputs/cwqr/latest.pt \
  --output outputs/calibration --device cuda:0

python -m tomodetr predict --config configs/czii.yaml \
  --manifest data.json --cache cache --cwqr outputs/cwqr/latest.pt \
  --calibration outputs/calibration/calibration.json \
  --output outputs/predictions --device cuda:0
```

The detector trains for 120 epochs (SHREC: 100 epochs on a 120-epoch schedule), with effective batch size 8, 10-epoch warmup and cosine decay. `--batch-size` must divide 8; accumulation preserves the effective batch. CWQR trains for 40 joint epochs with coordinate-loss weight 0.1. Training writes each epoch and `latest.pt`; `latest.pt` means the final epoch, **not a validation-selected best checkpoint**. Resume with `--resume outputs/detector/latest.pt` and the same output/config/data settings.

Calibration only reads `validation` annotations and writes model-bound frozen thresholds. Test prediction verifies the detector/refiner/config identities. For the paper's without-CWQR endpoint, omit `--cwqr` from both calibration and prediction and use separate output directories.

Prediction writes per-volume CSV and NPZ files. SHREC additionally writes `PDB x y z score` text submissions. With `--evaluate`, CZII uses official reference-coverage semantics, DeePiCt uses maximum-cardinality one-to-one matching, and SHREC uses its dilated occupancy mask. The local official-equivalent helpers are traceable in [provenance](docs/source_manifest.json); the organizers' scorers remain the authority for benchmark submissions.

