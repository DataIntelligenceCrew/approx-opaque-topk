from datetime import time
from typing import Dict, Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time

def relu_score(sample: float, scorer_params: Dict) -> float:
    """
    For synthetic data, the data points are just numbers. This also has a configurable delay parameter.
    :param sample: The number to be scored. The scoring function is relu(x) = max(0, x).
    :param scorer_params: The parameters for the scoring function. Just has a 'delay' parameter, in seconds.
    :return: The score of the sample.
    """
    time.sleep(scorer_params['delay'])
    return max(0.0, sample)


def get_scorer_from_params(scorer_params: Dict) -> Callable:
    """
    Returns the scoring function based on the parameters.
    :param scorer_params: The parameters for the scoring function.
    :return: The scoring function.
    """
    if scorer_params['type'] == 'relu':
        return lambda x: relu_score(x, scorer_params)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_params['type']}")

