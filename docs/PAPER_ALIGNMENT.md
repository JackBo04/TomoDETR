# Alignment to the compiled review manuscript

The source of truth is the review `main.tex` inclusion graph, specifically `sections/method_body.tex`, `sections/experiment_details.tex`, and `tables/implementation_settings.tex`. SHA256 snapshots are in `source_manifest.json`. Old `appendix_implementation.tex` and `appendix_datasets.tex` files are **not included by the current main.tex** and are not release specifications.

| Paper component | Implementation |
|---|---|
| Uniform 4×4×4 cell-center references; position plus sampled appearance | `tomodetr/detector.py: LocalDetector` |
| Residual U-Net, half/quarter/eighth resolution features | `ResidualUNetBackbone3D` |
| Four layers of self-attention, 3D deformable cross-attention, FFN | `DecoderLayer3D` |
| Per-layer Hungarian cost 5 L1 − log p(class) | `tomodetr/losses.py: match` |
| Local loss averaged over four layers; no-object weight 0.1 | `local_loss` |
| 96³ windows, 50% overlap, pre-NMS score floor 0.01 | `tomodetr/pipeline.py: export_volume` |
| Original coordinates; different-window neighbors plus self; K≤32 | `tomodetr/graph.py` |
| Query, global/local position, boundary proximity encoding | `tomodetr/cwqr.py: CWQR` |
| Head-specific relative displacement and distance bias | `tomodetr/attention.py` |
| Two post-normalized Transformer blocks (128/8/512, dropout 0.1) | `SparseTransformerBlock` |
| Zero-initialized logit/coordinate heads; ±4 per axis; extent clipping | `CWQR.forward` |
| Single whole-tomogram assignment; tolerance-normalized SmoothL1 | `global_loss` |
| Frozen cached detector; 40 joint CWQR epochs, weight 0.1 | `tomodetr/cli.py: train` |
| Class-aware NMS (CZII/DeePiCt), class-agnostic NMS (SHREC); validation-only thresholds | `pipeline.py`, `cli.py` |

## Removed experimental paths

Removed from the published implementation rather than hidden behind disabled flags:

- Dense vote/objectness heads and dense auxiliary supervision.
- Proposal-based query selection, proposal radius scaling and candidate refilling.
- GT-jitter/vote-endpoint auxiliary queries and one-to-many losses.
- Alternative overlapping query layouts, valid-core/halo experiments, frozen decoder centers.
- Sigmoid focal variants, alternative local center losses and custom classification priors.
- B2/G1 refiner variants, pointwise and self-only interaction controls.
- Query/position/geometry ablation switches, ROI refinement and multiscale classification heads.
- Radius-gated global assignment, logit-residual scaling sweeps, staged score-only/coordinate-only training.
- TCG/transfer calibrators, private benchmark defaults, remote launchers and result-specific thresholds.

The without-CWQR endpoint is retained solely as the paper's named comparison: use the same frozen candidate outputs without calling CWQR.

## Explicit protocol differences and unspecified details

1. **CZII initialization:** the current manuscript states 40 *joint* CWQR epochs. Historical result-producing runs also used score and coordinate initialization stages; the local review-fix notes explicitly distinguish joint duration from total historical training. This release implements the stated 40-epoch joint objective from identity initialization and does not reconstruct omitted stages. Published-table equivalence has not been established.
2. **SHREC final selection:** this release uses class-agnostic NMS, as specified by the current manuscript. The configuration uses a shared 10-voxel radius from the historical SHREC endpoint.
3. **Inference scores:** each query uses foreground argmax label and its softmax probability, consistent with the main local implementation. No foreground-sum scoring or residual-alpha branch is retained.
4. **DeePiCt:** the release follows the manuscript's independent within-condition 6/2/2 protocol, not old VPP Train8-to-defocus transfer. NMS radii of 5 voxels and particle radii of 10 voxels come from existing dataset configs; evaluation tolerance is the manuscript's 10 voxels. Calibration jointly optimizes pooled F1 across the two classes using validation data only.
5. **Checkpoint selection:** training exports all epochs. The documented quick start uses final epochs; selecting another checkpoint must use validation/calibration only. No historical per-seed best checkpoints or thresholds are invented.
6. **Implementation defaults:** detector dimension 128, channels 32/64/128/256, four sampling points per head/level, augmentation probabilities and 1,024 samples/epoch are retained from local source/configs. The 80/20 centered/uniform sampling is described by the compiled controlled-training protocol. All stochastic augmentation uses an epoch/index-derived seed.
7. **Precision and caching:** export runs float32, training uses CUDA bfloat16 autocast for the detector, and CWQR uses float32. Cache features are float32; each cache records the detector and input-volume hashes. These choices provide a transparent reference path, not a reproduction of historical timing measurements.

## License

Source code and documentation are released under the MIT License. Scientific image assets have separate provenance in `site/assets/PROVENANCE.md`.
