# Approximating Opaque Top-$k$ Queries

This repository contains code, experimental configs, data, and documentation. 

## Organization

Each subdirectory has the following functions. 
- `bandit` implements the main algorithm logic and scorer, sampler functions. 
- `index_builder` contains code used to generate indexes for synthetic, tabular, and image data. 
- `data_clean` contains documentation and code for data cleaning and for model training for tabular regression. 
- `data` contains experiment results which have been included in the paper. 
- `plots` contains code used to produce the plots for the paper and the plots themselves. 

## Reproducibility

The general workflow for reproducing experiments is as follows. 
1. Use `requirements.txt` to create a conda environment with the same dependencies as was used for the paper.
2. Download data and generate index according to the documentation in `index_builder`. This process creates two JSON index files: one dendrogram, and one flat index. The former is used for bandit algorithms and the latter is used for `UniformSample`. 
3. Run `python run_gt.py --config_filename [config_filename] --output_filename [output_filename]` to run a ground truth run, which exhaustively evaluates the scoring function for all elements of the search domain, then records their rankings and the ground truth solution. This information is logged to a JSON file. 
4. Run `python run_expr.py --config_filename [config_filename] --gt_filename [gt_filename] --output_filename [output_filename]` to run an experiment run. This run one or more configurations, and the gt file is used to compute the metrics for the running solution set. The metrics are logged to a JSON file.
5. If the experiment was split into multiple experiment config files, run `python combine_results.py [output_filename] [input_filenames...]` to combine the result statistics from one or more files into a single file. 
6. Use the Jupyter notebook file in `plots` to generate plots from the result. 
