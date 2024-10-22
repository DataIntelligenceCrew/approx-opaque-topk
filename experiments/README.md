# Experiment Configs

Note that all file paths in the config files have been set to be relative to some `<root-dir>` where we assume the datasets, index files, and results will be stored. This must be changed to the appropriate path on your system. The `<root-dir>` should contain the following subdirectories:
- `Data/ImageNet`, `Data/UsedCars`
- `Index/Synthetic`, `Index/ImageNet`, `Index/UsedCars`

The configs for each figure in the paper are located in the following subdirectories. 

1. **Synthetic Data** (Figure 4): `1-Synthetic` directory
2. **Varying Parameters** (Figure 5, 6): 
   - `2-1-Synthetic-k10`, `2-2-Synthetic-k500` for varying $k$ and `1-Synthetic` for medium $k$ baseline.
   - `2-3-SYnthetic-Replace` for sampling with replacement, where the `gt` run is `1-Synthetic/k100_gt.json`.
   - `2-4-Synthetic-Narrow` and `2-5-Synthetic-Wide` for varying the range in which $\mu$ is sampled from. Note that this requires running `/index_builder/synthetic_index_builder.py` with the appropriate parameters.
   - `2-6-Synthetic-BinCnt` for varying the number of bins in the histogram, where the `gt` run is `1-Synthetic/k100_gt.json`.
3. **Tabular Data** (Figure 7): `3-UsedCars` directory
4. **Image Data** (Figure 8): `4-ImageNet` directory