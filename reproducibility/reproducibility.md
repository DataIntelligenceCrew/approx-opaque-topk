## 1 Create virtual environment

```bash
uv sync --python 3.8
source .venv/bin/activate
```

## 2 Setup temporary directory

```bash
cd reproducibility
mkdir temp
mkdir temp/synthetic
mkdir temp/usedcars
mkdir temp/imagenet
mkdir temp/plots
```

After `cd reproducibility` all commands henceforth are being run from the `reproducibility` directory root. 

## 3 Build indexes

### 3a Build synthetic data index

```bash
python ../index_builder/synthetic_index_builder.py \
    --dendrogram_file temp/synthetic/dendrogram.json \
    --flattened_file temp/synthetic/flattened.json \ 
    -k 20 -n 2500 --stdev_max 0.0 --stdev_min 5.0 --mu_max 0.0 --mu_min 20.0
```

### 3b Download UsedCars data, clean, train XGBoost model, build index

### 3c Download ImageNet data, build ImageNet data index

