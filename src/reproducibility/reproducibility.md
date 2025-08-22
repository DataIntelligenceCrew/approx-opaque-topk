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

## 3 Build indexes

The VOODOO index is built for each dataset to accelerate queries. 

### 3a Build synthetic data index

```bash
python ../index_builder/synthetic_index_builder.py --dendrogram-file temp/synthetic/dendrogram.json --flattened-file temp/synthetic/flattened.json -k 20 -n 2500 --stdev-max 0.0 --stdev-min 5.0 --mu-max 0.0 --mu-min 20.0
```

### 3b Download UsedCars data, clean, train XGBoost model, build index

```bash
python ../index_builder/usedcars_download.py temp/usedcars/
python ../index_builder/usedcars_clean.py --data-dir temp/usedcars/
python ../index_builder/usedcars_train.py --label-col price temp/usedcars/
python ../index_builder/tabular_index_builder.py --input_file temp/usedcars/used_cars_val.parquet -k 500 --dendrogram_file temp/usedcars/dendrogram.json --flattened_file temp/usedcars/flattened.json --subsample_size 100000 --id_column listing_id --pred_column price
```

### 3c Download ImageNet data, build ImageNet data index

Assumption: The ImageNet-1k dataset tarballs `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` are downloaded to some path. 

## 4 Execute ground truth runs

The ground truth runs establish the optimal solution set and related metadata. 

### 4a Synthetic ground truth run

### 4b UsedCars ground truth run

### 4c ImageNet ground truth run

## 5 Run experiments

### 5a Synthetic data experiment (Fig. 4)

### 5b UsedCars tabular regression experiment (Fig. 5)

### 5c ImageNet fuzzy classification experiment (Fig. 7)

### 5d ImageNet parameter study (Fig. 9)














