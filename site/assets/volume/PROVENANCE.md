# Interactive 3D comparison

Case A: TS_89_8, original XYZ voxel bounds [562,345,112]–[608,391,134]. Shape: 46 × 46 × 22. Voxel spacing is recorded in scene.json in Å; all geometry and centers use this shared coordinate frame.

Sources: `paper/figures/qualitative_design_2026-09-13/final_delivery/mechanism_plot_arrays.npz` and the original TS_89_8 tomogram. The source archive SHA-256 is in scene.json. The raw volume mean is verified against the saved figure projection before export.

The foreground is the saved binary DeepFinder component. Marching tetrahedra at level 0.5, without smoothing, renders the boundary between native voxel sample centers. This is a visualization of the saved mask, not a new segmentation. Four reference centers lie in one connected component. Native center extraction produces two inaccurate centers. TomoDETR final predictions contain four correct picks; its local endpoint has four correct and two false picks. Matching labels are from the original 3D evaluation with a 3 nm tolerance.

The tomogram section uses a fixed 1st–99th percentile grayscale normalization across the entire crop, followed by 8-bit display quantization. The original sampling is unchanged. Marker radii are visual guides, not molecular radii or matching tolerances. Reference rings and prediction markers are overlaid as guides through the surface so the centers remain visible. Both views have linked cameras, zoom and clipping depth. Counts remain full-crop counts during cutaway. The four-stage timeline switches saved endpoints without inventing intermediate trajectories.

Regenerate with `python tools/export_3d_comparison.py --workspace-root /path/to/research/workspace --output site/assets/volume`.

Renderer: Three.js 0.160.1, MIT licensed; vendored module and license in site/vendor/. Scientific image assets retain the dataset terms documented in ../PROVENANCE.md.

## Public source and reuse terms

The local manifest identifies this exact tomogram as public CryoET Data Portal dataset **DS-10446**, run **TS_89_8**, tomogram **20646**, reconstruction **102**. The historical directory name `test_private_test` names the benchmark split; DS-10446 was released publicly on 2025-02-25.

- Dataset: https://cryoetdataportal.czscience.com/datasets/10446
- Exact source MRC (anonymous HTTP access verified): https://files.cryoetdataportal.cziscience.com/10446/TS_89_8/Reconstructions/VoxelSpacing10.012/Tomograms/102/TS_89_8.mrc
- Portal public-data policy: https://cryoetdataportal.czscience.com/data-submission-policy — publicly published data are accessible under CC0.
- Dataset publication: https://www.nature.com/articles/s41592-025-02800-5 — explicitly lists dataset IDs 10440, 10441, 10445 and 10446 with a CC0 license.

Only the 46 × 46 × 22 scientific display crop (46,552 bytes of quantized grayscale data), model-generated foreground and recorded prediction/reference coordinates are included here, not the full source MRC.
