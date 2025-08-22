from datetime import time
import time
from typing import Dict, Callable, List, Any

import pandas as pd
from PIL.Image import Image
import torch
from torchvision import models
from torchvision.models import ResNeXt101_64X4D_Weights
import torchvision.transforms as transforms
import xgboost as xgb


def get_relu_scorer(scorer_params: Dict) -> Callable:
    """
    For synthetic index_metadata, the index_metadata points are just numbers. This also has a configurable delay parameter.
    :param scorer_params: The parameters for the scoring function. Just has a 'delay' parameter, in seconds, per sample.
    :return: Scores of the samples, which are just the relu function applied to the samples.
    """
    def relu_scorer(scorer_inputs: List[str], params: Dict) -> List[float]:
        time.sleep(scorer_params['delay'] * len(scorer_inputs))
        return [max(0.0, float(x)) for x in scorer_inputs]
    return relu_scorer


def get_imagenet_classifier_scorer(scorer_params: Dict) -> Callable:
    """
    ImageNet classifier scorer computes the probability that an Image belongs to a specified ImageNet class index.

    :param scorer_params: A dictionary that contains the 'target_idx' field, which maps to integers in 0...999.
    :return: A ImageNet classifier scoring function.
    """
    def imagenet_classifier_scorer(scorer_inputs: List[Image], params: Dict) -> List[float]:
        """
        :param scorer_inputs: A list of PIL image objects.
        :param params: A dictionary that contains the 'target_idx' field.
        :return: A list of probabilities that each image belongs to target class.
        """
        # For Imagenet, scorer_inputs is a list of PIL Image objects

        # Load the pre-trained ResNeXt model
        model = models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT)
        model.eval()

        # Preprocess the input image
        preprocess = transforms.Compose([
            transforms.Resize(256),  # Resize the shortest side to 256 pixels
            transforms.CenterCrop(224),  # Crop the center 224x224 pixels
            transforms.ToTensor(),
            transforms.Normalize(  # Normalize with mean and std of ImageNet dataset
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Batch the inputs
        input_tensors = [preprocess(image.convert('RGB')) for image in scorer_inputs]
        if len(input_tensors) == 0:
            return []
        input_batch = torch.stack(input_tensors)

        # Send model and inputs to device (GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_batch = input_batch.to(device)
        model.to(device)

        # Infer the softmax probabilities for the batch
        with torch.no_grad():
            output = model(input_batch)
        probabilities_batch = torch.nn.functional.softmax(output, dim=1)

        # Extract target class's probabilities for each image
        target_class = params['target_idx']
        target_probs = probabilities_batch[:, target_class].cpu().tolist()  # Move to CPU and convert to list

        # Compile for optimization
        torch.compile()

        # Return target probabilities
        return target_probs

    return imagenet_classifier_scorer


def get_xgboost_scorer(scorer_params: Dict) -> Callable:
    """
    The XGBoost scorer loads a pre-trained XGBoost model and uses it to regress the input data.

    :param scorer_params: A dictionary that contains the 'model_path' field, which maps to the path of the XGBoost model.
    :return: An XGBoost scoring function.
    """
    # Load the XGBoost model from the specified path
    model_path = scorer_params['model_path']
    model = xgb.Booster()
    model.load_model(model_path)
    exclude_cols = scorer_params['exclude_cols']

    def xgboost_scorer(scorer_inputs: List[pd.DataFrame], params: Dict) -> List[float]:
        """
        :param scorer_inputs: A list of DataFrames, each containing a single row of input data.
        :param params: A dictionary that contains the 'model_path' field.
        :return: A list of predictions for each input DataFrame.
        """
        combined_df = pd.concat(scorer_inputs, ignore_index=True)
        combined_df_cleaned = combined_df.drop(exclude_cols, axis=1, errors='ignore')
        dmatrix = xgb.DMatrix(combined_df_cleaned)
        pred = model.predict(dmatrix)
        return pred.tolist()
    return xgboost_scorer


def get_scorer_from_params(scorer_params: Dict) -> Callable:
    """
    Returns the scoring function based on the parameters.
    :param scorer_params: The parameters for the scoring function.
    :return: The scoring function.
    """
    if scorer_params['type'] == 'relu':
        return get_relu_scorer(scorer_params)
    elif scorer_params['type'] == 'classify':
        return get_imagenet_classifier_scorer(scorer_params)
    elif scorer_params['type'] == 'xgboost':
        return get_xgboost_scorer(scorer_params)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_params['type']}")
