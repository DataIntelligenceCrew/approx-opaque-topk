import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


# Given a PIL Image, classifies it. Then, returns the inverse of the confidence for the primary class.
# Used to identify the hardest images to classify.
def inverse_confidence_score(image: Image.Image) -> float:
    # Load the pre-trained model (ImageNet)
    model = models.resnet18(pretrained=True)

    # Switch the model to evaluation mode
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Get the model's predictions
    with torch.no_grad():
        output = model(input_batch)

    # Apply softmax to get probabilities
    probabilities = nn.functional.softmax(output[0], dim=0)

    # Get the confidence of the primary (highest probability) class
    confidence = probabilities.max().item()

    # Return 1 / confidence
    return 1 / confidence



