from typing import Dict, Callable, Tuple, List

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
    else:
        raise ValueError(f"Sampler type {sampling_params['type']} not supported.")
