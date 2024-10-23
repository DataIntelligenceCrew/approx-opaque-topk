# Approximating Opaque Top-k Queries

This repository contains code, experimental configs, data, and documentation. 

## Organization

Each subdirectory has the following functions. 
- `bandit` implements the main algorithm logic and scorer, sampler functions. 
- `index_builder` contains code used to generate indexes for synthetic, tabular, and image data. Also contains code used for data cleaning and model training. 
- `experiments` contains configs for running experiments in the paper. 
- `results` contains experiment results which have been included in the paper. 
- `plots` contains code used to produce the plots for the paper and the plots themselves. 

## Reproducibility

The general workflow for reproducing experiments is as follows. 
1. Use `requirements.txt` to create a conda environment with the same dependencies as was used for the paper.
2. Download data and generate index according to the documentation in `index_builder`. This process creates two JSON index files: one dendrogram, and one flat index. The former is used for bandit algorithms and the latter is used for `UniformSample`. 
3. Run `python run_gt.py --config_filename <config_filename> --output_filename <output_filename>` to run a ground truth run, which exhaustively evaluates the scoring function for all elements of the search domain, then records their rankings and the ground truth solution. This information is logged to a JSON file. 
4. Run `python run_expr.py --config_filename <config_filename> --gt_filename <gt_filename> --output_filename <output_filename>` to run an experiment run. This run one or more configurations, and the gt file is used to compute the metrics for the running solution set. The metrics are logged to a JSON file.
5. If the experiment was split into multiple experiment config files, run `python combine_results.py <output_filename> <input_filenames...>` to combine the result statistics from one or more files into a single file. 
6. Use the Jupyter notebook file in `plots` to generate plots from the result. 

## Specifications

JSON files are used to store the index and to store configurations and intermediate results for the experiments. 

### Index

An index is a tree represented as a nested object. 
Each object has a single field `"children"`, a list. 
A non-leaf node has other objects as children. 
A leaf node has a list of string identifiers as children. 
We assume that each element in the search domain has a unique identifier to simplify experiments. 

We create a dendrogram index and a flat index for each dataset. 
The dendrogram index has intermediate nodes with two children. 
The flat index has a single leaf node that contains all elements. 

### Experiment Config

An experiment is a collection of named configs, stored as a mapping from the config name (unique string) to the config data. 
Fields that may need to be changed for reproducibility are starred.

Each config has the following fields:
1. `scoring_params` (obj): Parameters used to construct the scoring function. Has a `type` field and additional fields that differ per scoring function. 
   - `relu`: The relu function f(x) = max(0, x). 
     1. `delay` (float): Delay, in seconds, that is artificially injected per scoring function evaluation to simulate expensive scores. 
   - `classify`: The confidence that a pre-trained ResNeXT classifier thinks an image belongs to a target class idx. 
     1. `target_idx` (int): An integer ranging from 0 to 999, representing the corresponding ImageNet label. 
     2. *`device` (str): The CUDA device used to run inference. 
   - `xgboost`: The regression output by an XGBoost regression model. 
     1. *`model_path` (str): The path where the pre-trained model is stored. 
     2. `exclude_cols` (list[str]): The columns in the dataset that are not passed into the model, generally includes the y and id columns. 
2. `sampling_params` (obj): Parameters used to construct the sampling function, which returns the element itself given a string identifier. 
   - `synthetic`: Parses a string identifier directly into a floating-point number. 
   - `image_directory`: Reads an image with its filename equal to the string identifier from a directory, then returns a Pillow Image version of the image. 
     1. *`directory_path` (str): A directory that stores all images in the search domain. 
   - `dataframe`: Pre-loads a CSV file into memory as a dataframe, then returns rows of the dataframe by indexing over an id column. 
     1. *`file` (str): The path where the CSV file is stored. 
     2. `id_col` (str): The column which contains some unique identifier. 
     3. `exclude_cols` (list[str]): A list of columns which should be removed from the dataframe after loading it in. 
4. `algo_params` (obj): Parameters used to choose and tune the algorithm of choice. 
    - `epsgreedy`: OURS, which runs a histogram-based epsilon-greedy bandit. 
      1. `alpha` (float): A parameter used to tune the rate of exploration. 
      2. `max` (float): The maximum range of the initial histogram. 
      3. `min` (float): The minimum range of the initial histogram. 
      4. `num_bins` (int): The number of bins in the histograms. 
      5. `rebin_decay` (float): A parameter used to scale prior statistics in an exponential manner after re-binning. (Not used in paper; set to 1.0 throughout.)
      6. `enlarge_max_factor` (float): A parameter used to slightly over-estimate the true maximum value in a branch of the index when re-binning.
      7. `subtract` (bool): Used to turn on or off the "subtraction" feature used when a branch becomes empty. 
    - `ucb`: Runs a standard upper-confidence bound (UCB) bandit. 
      1. `c` (float): A parameter used to tune the rate of exploration. 
      2. `init` (float): The initial value that the mean estimates are set to. 
   - `UniformExploration`: Run an exploration-only bandit algorithm. (Note that exploration-only over a flat index and sampling without replacement is equivalent to uniform-sample in our setup.)
4. `budget` (int): The maximum number of iterations. 
5. `reps` (int): Number of times to repeat this config. All statistics will be averaged in the output file. 
6. `k` (int): The cardinality constraint for top-k querying. 
7. `index_params` (obj): Parameters used to access the index file. 
   - `file` (str): Points to the index file's location on disk. 
8. `sample_method` (str): Method used to sample identifiers from each leaf node.
   - `scan`: Sequentially scans the leaf node. 
   - `noreplace`: Performs sampling without replacement by pre-shuffling the leaf node, then performing a scan. 
   - `replace`: Performs sampling with replacement.
9. `batch_size` (int): The number of elements to sample and score in a single batch. 

### GT Config

Ground truth runs are special in that they always run a single run of exhaustive scan over a flat index, then logs its results in a distinct format for later use. 
The config for a ground truth run is similar to a regular experiment config, but only contains the following fields. 

1. `scoring_params`
2. `sampling_params`
3. `k`
4. `index_params`
5. `batch_size`

### GT Result

The gt run's result is stored in a JSON file with the following fields. 

1. `gt_solution` (list[str]): A list of all elements' identifiers in the gt top-k solution. 
2. `gt_rankings` (dict[str, int]): A mapping from a string identifier for an element and its ground truth ranking. A ranking of 1 means that it has the highest score, and a ranking of n is for the lowest score. 
3. `n` (int): The total number of elements in the search domain.

### Experiment Result

The non-gt experimental runs' result is stored in a JSON file as a dictionary mapping from the config's name (str) to its result (obj). Each result has the following fields. 

1. `STK` (list[float]): The average sum-of-top-k scores (STK) at each iteration over all reps. 
2. `KLS` (list[float]): The average kth-largest-score (KLS) at each iteration over all reps. If the running solution has size less than k, returns zero. 
3. `time` (list[float]): The average time at each iteration over all reps. 
4. `Precision@K` (list[float]): The average Precision@K at each iteration over all reps. 
5. `Recall@K` (list[float]): The average Recall@K at each iteration over all reps. 
6. `AvgRank` (list[float]): The average rank of the running solution at each iteration over all reps. If the running solution has size less than k, the rankings are padded with n for all missing elements. 
7. `WorstRank` (list[float]): The kth rank of the running solution at each iteration over all reps. Returns n if the running solution has size less than k. 
8. `reps` (float): The number of repetitions for this config. 
