# Dynamic volume demo

Nine slice atlases were regenerated directly from the original MRC crops, not from rendered paper figures. Each frame is the mean of up to three adjacent original slices. Contrast is fixed at the crop-wide 1st/99th intensity percentiles and shared across methods and directions. Physical dimensions and per-axis voxel spacing come from MRC headers.

The selected regions are CZII TS_89_8 apo-ferritin, DeePiCt VPP TS_0001 ribosomes, and CZII TS_15_8 beta-galactosidase. Recorded predictions, reference centers, and 3D matching labels are the numerical source arrays of the current qualitative comparison. No models were retrained for this demo.

Slice markers are gated to within 3 nm of the central plane. Counts cover the full selected crop. The spatial view renders true 3D coordinates and a moving plane; it is a coordinate view, not a reconstructed molecular surface.

`volumes.json` records crop shapes, physical dimensions, and SHA-256 of raw crop arrays. Recreate the assets with `tools/export_volume_demo.py --source <figure-array-directory> --output site/demo/assets`. The source directory must contain the recorded NPZ and manifest with accessible original MRC paths.

Original dataset terms apply to tomographic excerpts. Code is MIT licensed.
