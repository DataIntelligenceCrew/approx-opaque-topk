NOTE:
- This reproducibility procedure was tested on a fresh machine. 
- We did not implement reproducible randomness for the stochastic algorithms, so while general trends should hold (since we repeat each experiment 10+ times), exact result might vary from the plots published in the paper. 

## 1 Create virtual environment

```bash
uv sync --python 3.8
source .venv/bin/activate
```



## 2 Setup temporary directory

```bash
cd src/reproducibility
mkdir temp
mkdir temp/synthetic
mkdir temp/usedcars
mkdir temp/imagenet
mkdir temp/plots
```

All commands henceforth are being run from the `reproducibility` directory root. 



## 3 Synthetic data experiments

### Build index

```bash
python ../index_builder/synthetic_index_builder.py --dendrogram-file temp/synthetic/dendrogram.json --flattened-file temp/synthetic/flattened.json -k 20 -n 2500 --stdev-max 0.0 --stdev-min 5.0 --mu-max 0.0 --mu-min 20.0
```

Expected files:
- `./temp/synthetic/dendrogram.json`
- `./temp/synthetic/flattened.json`

### Execute ground truth run

```bash
python ../../run_gt.py ../experiments/synthetic/k100_gt.json temp/synthetic/k100_gt.json temp/synthetic/k100_sorted.json
```

Expected files:
- `./temp/synthetic/k100_gt.json`
- `./temp/synthetic/k100_sorted.json`

### Run main experiments

```bash
python ../../run_expr.py ../experiments/synthetic/k100_all.json temp/synthetic/k100_gt.json temp/synthetic/k100_result.json
```

Expected files:
- `./temp/synthetic/k100_result.json`

### Create plots

```bash
python ../plots/plot_synthetic.py --result-file temp/synthetic/k100_all.json --output-dir temp/synthetic/
```

Expected files:
- `./temp/synthetic/synthetic_stk_vs_iter.pdf` (Fig. 4a)
- `./temp/synthetic/synthetic_precision_vs_iter.pdf` (Fig. 4b)
- `./temp/synthetic/synthetic_ablation_study.pdf` (Fig. 4c)



## 4 UsedCars tabular regression

### Download UsedCars data, clean, train model, build index

```bash
python ../index_builder/usedcars_download.py temp/usedcars/
python ../index_builder/usedcars_clean.py --data-dir temp/usedcars/ --val-n 100000
python ../index_builder/usedcars_train.py --label-col price temp/usedcars/
python ../index_builder/tabular_index_builder.py --input_file temp/usedcars/used_cars_val.parquet -k 500 --dendrogram_file temp/usedcars/dendrogram.json --flattened_file temp/usedcars/flattened.json --subsample_size 100000 --id_column listing_id --pred_column price --time_file temp/usedcars/index_time.txt
```

Expected files:
- `./temp/usedcars/usedcars.csv`
- `./temp/usedcars/used_cars_[clean/test/train/val].parquet`
- `./temp/usedcars/usedcars_model.json`
- `./temp/usedcars/dendrogram.json`
- `./temp/usedcars/flattened.json`
- `./temp/usedcars/index_time.txt`

### Execute ground truth run

```bash
python ../../run_gt.py ../experiments/usedcars/gt.json temp/usedcars/gt.json temp/usedcars/sorted.json
```

- `./temp/usedcars/gt.json`
- `./temp/usedcars/sorted.json`

### Run main experiments

```bash
python ../../run_expr.py ../experiments/usedcars/all.json temp/usedcars/gt.json temp/usedcars/result.json
```

NOTE: 
- Replacing `../experiments/usedcars/all.json` (which was used to produce plots for the paper) with `../experiments/usedcars/all_fast.json` disables scoring function computation during the experiment and instead uses the cached scoring function values from the `./temp/usedcars/gt.json` file. 
- This can be used to quickly verify the plots working for (STK/Precision) vs iteration, as well as the per-iteration overhead plots, but (STK/Precision) vs time plots and end-to-end latency plots will be off due to incorrect timings. 
- The slow version may take 15~20 hours, hardware-dependent. 

Expected files:
- `./temp/usedcars/result.json`

### Create plots

```bash
python ../plots/plot_usedcars.py --gt-file temp/usedcars/gt.json --result-file temp/usedcars/result.json --output-dir temp/usedcars/ --index-time-file temp/usedcars/index_time.txt
```

Expected files:
- `./temp/usedcars/usedcars_stk_vs_time.pdf` (Fig. 5a)
- `./temp/usedcars/usedcars_precision_vs_time.pdf` (Fig. 5b)
- `./temp/usedcars/usedcars_latency_total.pdf` (Fig. 5c)
- `./temp/usedcars/usedcars_ablation_study.pdf` (Fig. 6a)
- `./temp/usedcars/usedcars_latency_iter.pdf` (Fig. 6b)
- `./temp/usedcars/usedcars_parameter_study.pdf` (Fig. 6c)

NOTE: 
- For the parameter study plot, the line labeled "OURS" is manually renamed to "F=0.01 (Default)" in the paper. 



## ImageNet fuzzy classification

### Process ImageNet data, build index

Assumption: 
- The ImageNet-1k dataset tarballs `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` are downloaded to some path. 
This part cannot be automated due to the licensing. 
- `imagenet_process.py` will unzip and format the images in a flat format for later scripts to use at `<imagenet-dir>`, which should have enough space to store the ImageNet train set as PNG files. 
- `<imagenet-dir>` is separated from `./temp` intentionally since home drive of many systems do not have enough space to store ImageNet unzipped. 

```bash
echo <imagenet-dir> > temp/imagenet/dataset_location.txt
python ../index_builder/imagenet_process.py <imagenet-tarball> $(cat temp/imagenet/dataset_location.txt)
python ../index_builder/pixel_index_builder.py --dendrogram-file temp/imagenet/dendrogram.json --flattened-file temp/imagenet/flattened.json -k 25 --subsample-size 100000 --image-directory $(cat temp/imagenet/dataset_location.txt)  --time-file temp/imagenet/index_time.txt
```

Expected files:
- `<imagenet-dir>/0_0.png`, ...
- `./temp/imagenet/dendrogram.json`
- `./temp/imagenet/flattened.json`

### Execute ground truth runs

```bash
python ../../run_gt.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/437-gt.json) temp/imagenet/437-gt.json temp/imagenet/437-sorted.json
python ../../run_gt.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/590-gt.json) temp/imagenet/590-gt.json temp/imagenet/590-sorted.json
python ../../run_gt.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/897-gt.json) temp/imagenet/897-gt.json temp/imagenet/897-sorted.json
```

NOTE:
- 437 is "beacon, lighthouse"; 590 is "hand-held computer"; 897 is "washing machine"

Expected files:
- `./temp/imagenet/437-sorted.json`
- `./temp/imagenet/437-gt.json`
- `./temp/imagenet/590-sorted.json`
- `./temp/imagenet/590-gt.json`
- `./temp/imagenet/897-sorted.json`
- `./temp/imagenet/897-gt.json`

### Run main experiments

**Slow version (published in paper):**

```bash
python ../../run_expr.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/437-all.json) temp/imagenet/437-gt.json temp/imagenet/437-all.json
python ../../run_expr.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/590-all.json) temp/imagenet/590-gt.json temp/imagenet/590-all.json
python ../../run_expr.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/897-all.json) temp/imagenet/897-gt.json temp/imagenet/897-all.json
```

**Fast version (for basic verification of [STK/Precision@K] vs iteration results):**

```bash
python ../../run_expr.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/437-all-fast.json) temp/imagenet/437-gt.json temp/imagenet/437-all.json
python ../../run_expr.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/590-all-fast.json) temp/imagenet/590-gt.json temp/imagenet/590-all.json
python ../../run_expr.py <(sed "s|<imagenet-dir>|$(cat temp/imagenet/dataset_location.txt)|g" ../experiments/imagenet/897-all-fast.json) temp/imagenet/897-gt.json temp/imagenet/897-all.json
```

NOTE:
- The sed snippet in the commands replace occurrences of `<imagenet-dir>` with the path name cached to `./temp/imagenet/dataset_location.txt` earlier. 

Expected files:
- `./temp/imagenet/437-all.json`
- `./temp/imagenet/590-all.json`
- `./temp/imagenet/897-all.json`

### Main experiment plotting
    parser.add_argument('--gt-file', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--index-time-file', type=str, required=True)
    parser.add_argument('--imagenet-class-id', type=int, required=True)

```bash
python ../plots/plot_imagenet.py --gt-file temp/imagenet/437-gt.json --result-file temp/imagenet/437-all.json --output-dir temp/imagenet/ --index-time-file temp/imagenet/index_time.txt --imagenet-class-id 437
python ../plots/plot_imagenet.py --gt-file temp/imagenet/437-gt.json --result-file temp/imagenet/590-all.json --output-dir temp/imagenet/ --index-time-file temp/imagenet/index_time.txt --imagenet-class-id 590
python ../plots/plot_imagenet.py --gt-file temp/imagenet/437-gt.json --result-file temp/imagenet/897-all.json --output-dir temp/imagenet/ --index-time-file temp/imagenet/index_time.txt --imagenet-class-id 897
```

NOTE:
- For the latency and overhead plots (Fig. 8b, 8c), we used the data from class 437 in the paper. 
- For the parameter study, "OURS" was manually renamed to "F=0.01, Batch=400 (Default)" for the paper. 

Expected files:
- `./temp/imagenet/imagenet_<class-id>_stk_vs_time.pdf` (Fig.7 a-c)
- `./temp/imagenet/imagenet_<class-id>_precision_vs_time.pdf` (Fig.7 d-f)
- `./temp/imagenet/imagenet_<class-id>_parameter_study.pdf` (Fig. 9)
- `./temp/imagenet/imagenet_<class-id>_latency_iter.pdf` (Fig. 8c)
- `./temp/imagenet/imagenet_<class-id>_latency_total.pdf` (Fig. 8b)

### Batch size vs latency & memory consumption experiment & plotting

```bash
python ../plots/plot_imagenet_batch_size.py --image-dir $(cat temp/imagenet/dataset_location.txt) --max-batches 25 --out-dir temp/imagenet/
```

Expected files:
- `./temp/imagenet/results-latency_memory.csv`
- `./temp/imagenet/batchsize.pdf` (Fig. 8a)

