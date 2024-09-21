from typing import Dict, Callable

from PIL import Image


def synthetic_sampler(sample: str, sampling_params: Dict) -> str:
    """
    For the Synthetic dataset, the sampler is just the identity function.
    Since we assume that the index_metadata is stored in the index_builder as string keys, we need to convert it to an integer.

    :param sample: Information stored for a sample in the index. For a synthetic index, there are three fields:
                   'id' (unique 0-indexed int), 'score' (float), and 'rank' (unique 1-indexed int).
    :param sampling_params: The sampling parameters. Not used in this case.
    :return: The sample itself.
    """
    return sample

def image_directory_sampler(sample: str, sampling_params: Dict) -> Image.Image:
    """
    For image datasets, the sampler retrieves the image from the directory based on its filename.

    :param sample: Information stored for a sample in the index. For an image index, there are two fields:
                   'filename' (unique str),
    :param sampling_params: The sampling parameters. Here, it should have 'directory_path' key.
    :return: The image as a PIL Image object.
    """
    return Image.open(sampling_params['directory_path'] + sample)

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
