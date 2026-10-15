# Recorded-stage qualitative comparison

## Main comparison: TS_89_8

The original manuscript's four-particle apo-ferritin region, with a mean of 22 original slices. All images have identical crop, grayscale limits, and spatial extent. Gold marks reference centers; green marks matched predictions; red marks unmatched predictions. Correspondence is evaluated in 3D at 30 Å, not from projected distances.

DeepFinder's saved foreground is one 3D connected semantic component spanning the four reference centers. Native mean-shift produces two centers; neither is within tolerance. The current TomoDETR output contains four matched centers and no unmatched predictions in this crop. The local-detector endpoint contains four matches and two false positives. The displayed detector and complete-system coordinates are saved selected endpoints, not intermediate decoder layers.

The 3D inset consists of outward exposed faces of the archived foreground voxels. No bridging, dilation, or synthetic segmentation is added. Gold reference markers are drawn over the surface for spatial inspection.

## Native baseline traces: TS_100_4

A separate six-particle crowded region (XYZ voxel bounds 552:622, 230:284, 78:114) has complete baseline stage records. It is not the same region as the main comparison.

- **DeepFinder:** saved connected semantic component, native mean-shift centers, and all final predictions in the crop. The selected component spans six reference centers and has 5,696 full-resolution foreground voxels. It produces two native centers; there are three final predictions in the whole crop.
- **DeepETPicker:** exact frozen-network patch probabilities, fraction of thresholded patch masks, repeated mean pooling, and final native MP-NMS candidates. Display maps average overlapping patch inference cores. Native threshold: 0.5; kernel: 5; repetitions: 8. Do not interpret threshold fraction as thresholding a stitched probability map. Replayed candidates match archived outputs within 0.001 in native coordinate units.
- **OCTOPI:** probabilities, semantic argmax mask, binary opening, distance transform, local-maxima markers, watershed labels, size filtering, and final picks. Frozen-model replay agrees with saved final coordinates within 0.001 Å. Intermediate arrays end at the actual foreground bounding box; the nine empty trailing X planes are padded with zeros after verifying the saved argmax foreground is zero there.

All recorded predictions are retained inside the selected crop. Trace counts (TP / FP / FN): DeepFinder 0 / 3 / 6; DeepETPicker 1 / 3 / 5; OCTOPI 2 / 0 / 4. These are selected diagnostic cases. The main table provides whole-dataset comparisons.

Backgrounds are means of original MRC slices. Foreground, probability, response, and label overlays are max projections of saved 3D arrays. Probability and smoothed-response colors use a 0–1 scale. Distance-map display is normalized by its maximum; watershed colors distinguish labels. No generated model responses are used. Playback changes the recorded stage; it does not simulate optimization trajectories.

Source hashes and crop metrics are in `provenance.json`. The code's MIT license does not relicense original tomographic datasets; original CZII dataset terms apply to image excerpts.
