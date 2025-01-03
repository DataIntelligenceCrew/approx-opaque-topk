from typing import Dict, Callable, List

import pandas as pd
from PIL import Image


def synthetic_sampler(sample_ids: List[str], sampling_params: Dict) -> List[float]:
    """
    For the Synthetic dataset, the sampler is just the identity function.
    Since we assume that the index_metadata is stored in the index_builder as string keys, we need to convert it to an integer.

    :param sample_ids: A list of unique string IDs to identify elements in the index and in storage.
    :param sampling_params: The sampling parameters. Not used in this case.
    :return: A list of the float representation of the strings.
    """
    return [float(id_) for id_ in sample_ids]


def image_directory_sampler(sample_ids: List[str], sampling_params: Dict) -> List[Image.Image]:
    """
    For image datasets, the sampler retrieves the images from the directory based on its filename.

    :param sample_ids: For image elements, the sample IDs are filenames.
    :param sampling_params: The sampling parameters. Here, it should have 'directory_path' key.
    :return: A list of PIL images.
    """
    return [Image.open(sampling_params['directory_path'] + id_) for id_ in sample_ids]


def get_dataframe_sampler(sampling_params: Dict) -> Callable:
    """
    For tabular datasets, the sampler retrieves the data from the dataframe based on the index.

    If df.loc[id_] is a Series, it converts it into a single-row DataFrame.
    If df.loc[id_] is a DataFrame, it retains only the first row.

    :param sampling_params: The sampling parameters. Here, it should have 'file', 'id_col', and 'exclude_cols' keys.
    :return: A function that returns a list of DataFrames.
    """
    df = pd.read_csv(sampling_params['file']).drop(labels=sampling_params['exclude_cols'], axis=1, errors='ignore')
    df.set_index(sampling_params['id_col'], inplace=True)

    def dataframe_sampler(sample_ids: List[str], sampling_params: Dict) -> List[pd.DataFrame]:
        results = []
        for id_ in sample_ids:
            result = df.loc[id_]
            if isinstance(result, pd.Series):
                # Convert Series to a single-row DataFrame
                result_df = result.to_frame().T
            else:
                # If result is already a DataFrame, take only the first row
                result_df = result.iloc[:1]
            results.append(result_df)
        return results

    return dataframe_sampler


def get_sampler_from_params(sampling_params: Dict) -> Callable:
    """
    Returns the sampler function based on the sampling parameters.

    :param sampling_params: The sampling parameters. It should at least have 'type' key, and other keys as needed by the sampler.
    :return: The sampler function.
    """
    if sampling_params['type'] == 'synthetic':
        return synthetic_sampler
    elif sampling_params['type'] == 'image_directory':
        return image_directory_sampler
    elif sampling_params['type'] == 'dataframe':
        return get_dataframe_sampler(sampling_params)
    else:
        raise ValueError(f"Sampler type {sampling_params['type']} not supported.")
