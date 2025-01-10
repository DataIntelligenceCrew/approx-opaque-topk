# Index Builders

The files in this directory are used to build an index over a given dataset. Each time we run a index builder, we generate a hierarchical (dendrogram) index file and a flat index file. The former is used for bandit algorithms and the latter is used for `UniformSample`. 

## Synthetic
Synthetic index is generated, so it is done in a single command. The command with default parameters is as follows. 
```
python synthetic_index_builder.py --dendrogram_file <dendrogram_file> --flattened_file <flattened_file> -k 20 -n 2500 --stdev_max 0.0 --stdev_min 5.0 --mu_max 0.0 --mu_min 10.0
```
The paths will differ depending on where the indexes are stored.

## Tabular

Index for the UsedCars data is built using the following steps. 
1. Download full data from https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset.
2. Use the `car_data_clean` Jupyter notebook to clean the data, and split the data into train, test, and index splits. 
3. Use train & test splits to train the XGBoost model. 
4. Run `tabular_index_builder.py` on the index split. 

The command to run the index builder is the following. 
```
python tabular_index_builder.py --input_file <path_to_input_file> -k 500 --dendrogram_file <dendrogram_file> --flattened_file <flattened_file> --subsample_size 100000 --id_column listing_id --pred_column price
```
The paths will differ depending on where the dataset and indexes are stored. We assume that the id_column has type string to simplify typing in the Python side of the code base. 

python tabular_index_builder.py --input_file ~/Data/UsedCars/used_cars_val.parquet -k 500 --dendrogram_file ../Temp/Index/UsedCars/dendrogram.json --flattened_file ../Temp/Index/UsedCars/flattened.json --subsample_size 100000 --id_column listing_id --pred_column price --seed 24

## Image

Index for ImageNet is built using the following steps. 
1. Download full dataset. 
2. Flatten out all subdirectories s.t. a single directory holds all images. 
3. Randomly subsample a quarter of the images (n = 320291). So there should now be a directory holding 320k images with filenames of the form "[class_idx]_[image_idx].png".
4. Run `pixel_index_builder.py` on the images.

The specific command that we used is the following. 
```
python3 pixel_index_builder.py --dendrogram-file <dendrogram_file> --flattened-file <flattened_file> -k 500 --subsample-size 100000 --image-directory <image_directory>
```
The paths will differ depending on where the dataset and indexes are stored. 
