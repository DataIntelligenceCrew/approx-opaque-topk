from typing import Dict

from PIL import Image


def synthetic_sampler(sample_id: str, sampling_params: Dict):
    """
    For the Synthetic dataset, the sampler is just the identity function.
    Since we assume that the data is stored in the index as string keys, we need to convert it to an integer.

    :param sample_id: The id of the data point.
    :param sampling_params: The sampling parameters. Not used in this case.
    :return: The id of the data point as an integer.
    """
    return float(sample_id)

def image_directory_sampler(sample_id: str, sampling_params: Dict) -> Image.Image:
    """
    For image datasets, the sampler retrieves the image from the directory based on its filename.

    :param sample_id: The filename of the image.
    :param sampling_params: The sampling parameters. Here, it should have 'directory_path' key.
    :return: The image as a PIL Image object.
    """
    return Image.open(sampling_params['directory_path'] + sample_id)

def get_sampler_from_params(sampling_params: Dict):
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
