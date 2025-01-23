import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import gc

# Specify the directory containing the image files
image_dir = ""
batch_sizes = [800]

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        label = int(self.image_files[idx].split("_")[0])  # Extract class_idx from filename

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}

def load_imagenet_dataset(image_dir):
    """Load ImageNet dataset from a directory with images named [class_idx]_[number].png."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_dir=image_dir, transform=transform)
    return dataset

# Load ImageNet dataset
dataset = load_imagenet_dataset(image_dir)

def measure_memory_consumption(batch_size):
    """Measure maximum GPU memory consumption for a specific batch size."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Shuffle the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT).to(device)
    model.eval()

    max_memory = 0

    batch_count = 0
    for batch in dataloader:
        if batch_count >= 5:
            break

        images = batch["image"].to(device)
        try:
            with torch.no_grad():
                model(images)  # Run inference
            torch.cuda.synchronize()
            max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))  # Measure max memory
        except RuntimeError as e:
            print(f"RuntimeError for batch size {batch_size}: {e}")
            break

        # Clear cache and garbage collect
        del images
        torch.cuda.empty_cache()
        gc.collect()

        batch_count += 1

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return max_memory / (1024 ** 2)  # Convert to MB

if __name__ == "__main__":
    print("Batch size, Max Memory Consumption (MB)")
    for batch_size in batch_sizes:
        try:
            max_memory = measure_memory_consumption(batch_size)
            print(f"{batch_size}, {max_memory:.2f} MB")
        except Exception as e:
            print(f"Failed to measure for batch size {batch_size}: {e}")
