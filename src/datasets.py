# Import packages
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Digits(Dataset):
    """Dataset class for input images, either real or synthetic.

    Args:
        csv_file_path (str): Path to the CSV input file.
        transform (callable, optional): Transformation to apply to images.
    """

    def __init__(self, csv_file_path, transform=None):
        # Get data
        self.data = pd.read_csv(csv_file_path)
        self.digits = self.data.iloc[:, 1:].values
        self.labels = self.data["label"].values
        self.transform = transform

        # Infer image dimensions
        total_pixels = self.digits.shape[1]
        side_length = int(np.sqrt(total_pixels))  # Assuming square images
        self.img_height = side_length
        self.img_width = side_length

    def __getitem__(self, index):
        digit = (
            self.digits[index]
            .reshape(self.img_height, self.img_width)
            .astype(np.float32)
        )
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transform:
            # Convert to PIL Image for applying torchvision transforms
            digit = Image.fromarray(digit)
            digit = self.transform(digit)
        else:
            # Add channel dimension to match expected dimensions
            digit = torch.tensor(digit).unsqueeze(dim=0)

        return digit, label

    def __len__(self):
        return len(self.digits)


def get_datasets(csv_file_paths, transform=None):
    """Creates PyTorch datasets for real and synthetic images.

    Args:
        csv_file_paths (list of str): List containing paths to CSV input files.
        transform (callable, optional): Transformation to apply to images.

    Returns:
        tuple: (real_dataset, synthetic_dataset)
    """
    if not isinstance(csv_file_paths, list) or len(csv_file_paths) != 2:
        raise ValueError("csv_file_paths should be a list of two file paths.")

    real_dataset = Digits(csv_file_paths[0], transform=transform)
    synthetic_dataset = Digits(csv_file_paths[1], transform=transform)

    return real_dataset, synthetic_dataset


def get_dataloaders(csv_file_paths, batch_size, transform=None):
    """Creates PyTorch dataloaders for each dataset.

    Args:
        csv_file_paths (list of str): List containing paths to CSV input files.
        batch_size (int): Batch size for the dataloaders.
        transform (callable, optional): Transformation to apply to images.

    Returns:
        tuple: (real_loader, syn_loader)
    """

    real_dataset, synthetic_dataset = get_datasets(csv_file_paths, transform)

    real_loader = DataLoader(
        real_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    syn_loader = DataLoader(
        synthetic_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return real_loader, syn_loader


if __name__ == "__main__":
    paths = ["data/input/real_mnist.csv", "data/input/synthetic_mnist.csv"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Test Digits class
    syn_dataset = Digits(paths[1], transform=transform)
    print("Size of one synthetic image: ", syn_dataset[0][0].shape)
    real_dataset = Digits(paths[0], transform=transform)
    print("Size of one real image: ", real_dataset[0][0].shape)
    real_dataset_no_transform = Digits(paths[0], transform=None)
    print(
        "Size of one real image without transformation: ",
        real_dataset_no_transform[0][0].shape,
    )

    # Test get_datasets function
    real_dataset, syn_dataset = get_datasets(paths, transform=transform)
    print("Number of all real images: ", len(real_dataset))
    print("Number of all synthetic images: ", len(syn_dataset))

    # Test get_dataloaders function
    real_dataloader, syn_dataloader = get_dataloaders(
        paths, batch_size=16, transform=transform
    )
    print("Number of real image batches: ", len(real_dataloader))
    print("Number of synthetic image batches: ", len(syn_dataloader))
