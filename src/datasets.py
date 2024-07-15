# Import packages
import importlib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Import own files
import config

# Reload own files
importlib.reload(config)


class MNIST(Dataset):
    """Dataset class for input images, either real or synthetic.

    Args:
        csv_file_path (str): path to csv input file
        transform (str): transformation to use on images
    """

    def __init__(self, csv_file_path, transform=None):

        self.data = pd.read_csv(csv_file_path)
        self.digits = self.data.iloc[:, 1:].values
        self.labels = self.data["label"].values
        self.transform = transform

        if self.transform == "standard":
            scaler = StandardScaler()
            self.norm_digits = scaler.fit_transform(self.digits).reshape(
                -1, config.img_height, config.img_width
            )
        elif self.transform == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.norm_digits = scaler.fit_transform(self.digits).reshape(
                -1, config.img_height, config.img_width
            )

    def __getitem__(self, index):

        if self.transform is not None:
            # Add channel dimension (1 channel for grayscale)
            digit = (
                torch.as_tensor(self.norm_digits[index])
                .unsqueeze(dim=0)
                .float()
            )
            label = torch.as_tensor(self.labels[index], dtype=torch.long)
        else:
            digit = self.digits[index].reshape(
                -1, config.img_height, config.img_width
            )
            label = self.labels[index]

        return digit, label

    def __len__(self):
        return len(self.digits)


def get_datasets(csv_file_paths, transform=None):
    """Function creates PyTorch datasets for real and sythetic images.

    Args:
        csv_file_path (str): path to csv input file
        transform (str): transformation to use on images

    Returns:
        pytorch dataset: real dataset, synthetic dataset
    """

    real_dataset = MNIST(csv_file_paths[0], transform=transform)
    synthetic_dataset = MNIST(csv_file_paths[1], transform=transform)

    return real_dataset, synthetic_dataset


def get_dataloaders(csv_file_paths, batch_size, transform=None):
    """Function creates PyTorch dataloaders for each dataset.

    Returns:
        pytorch dataloader: real_loader, syn_loader
    """

    real_dataset, synthetic_dataset = get_datasets(csv_file_paths, transform)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    syn_loader = DataLoader(
        synthetic_dataset, batch_size=batch_size, shuffle=True
    )

    return real_loader, syn_loader


if __name__ == "__main__":
    paths = ["data/input/real_mnist.csv", "data/input/synthetic_mnist.csv"]

    # Test MNIST class
    syn_dataset = MNIST(paths[1], transform="minmax")
    print("Size of one synthetic image: ", syn_dataset[0][0].shape)

    real_dataset = MNIST(paths[0], transform="minmax")
    print("Size of one real image: ", real_dataset[0][0].shape)

    # Test get_datasets function
    real_dataset, syn_dataset = get_datasets(paths, transform="minmax")
    print("Number of all real images: ", len(real_dataset))
    print("Number of all synthetic images: ", len(syn_dataset))

    # Test get_dataloaders function
    real_dataloader, syn_dataloader = get_dataloaders(
        paths, batch_size=16, transform="minmax"
    )
    print("Number of real image batches: ", len(real_dataloader))
    print("Number of synthetic image batches: ", len(syn_dataloader))
