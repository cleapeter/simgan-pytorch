import torch

# Choose GPU if available
device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

# File paths
real_csv_file = 'data/input/real_mnist.csv'
synthetic_csv_file = 'data/input/synthetic_mnist.csv'
csv_file_paths = [real_csv_file, synthetic_csv_file]

# Image properties
img_width = 28
img_height = 28
num_channels = 1
num_classes = 10

# Data preprocessing
transform = 'minmax'

# Model
model = 'SimGAN'

# Number of combinations for hyperparameter tuning
# if grid_search in raytuning -> use num_trials = 1
num_trials = 2
