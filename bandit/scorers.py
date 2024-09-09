from datetime import time
from typing import Dict

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
    return max(0, sample)




