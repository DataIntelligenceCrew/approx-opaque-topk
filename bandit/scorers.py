from datetime import time
import random
from typing import Dict, Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time

def get_relu_scorer(scorer_params: Dict) -> Callable:
    """
    For synthetic index_metadata, the index_metadata points are just numbers. This also has a configurable delay parameter.
    :param sample: The number to be scored. The scoring function is relu(x) = max(0, x).
    :param scorer_params: The parameters for the scoring function. Just has a 'delay' parameter, in seconds.
    :return: The score of the sample.
    """
    time.sleep(scorer_params['delay'])
    return lambda x: max(0.0, x)


def get_random_scorer(scorer_params: Dict) -> Callable:
    """
    Random scoring function is used to simulate a scenario where the index has no correlation with scores.
    """
    time.sleep(scorer_params['delay'])
    return lambda x: random.random()


def get_scorer_from_params(scorer_params: Dict) -> Callable:
    """
    Returns the scoring function based on the parameters.
    :param scorer_params: The parameters for the scoring function.
    :return: The scoring function.
    """
    if scorer_params['type'] == 'relu':
        return get_relu_scorer(scorer_params)
    elif scorer_params['type'] == 'random':
        return get_random_scorer(scorer_params)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_params['type']}")


