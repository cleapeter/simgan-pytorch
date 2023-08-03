import pandas as pd

# Download the following files and place them as they are in data/raw:
# 1. Typeface MNIST "TMNIST_Data.csv"
# https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist
# 2. MNIST "mnist_train.csv" and "mnist_test.csv"
# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
# Then run this file

# Load and merge real mnist data
mnist_1 = pd.read_csv('data/raw/mnist_train.csv')
mnist_2 = pd.read_csv('data/raw/mnist_test.csv')
mnist = mnist_1.merge(mnist_2, how='outer')
mnist.to_csv('data/input/real_mnist.csv', sep=',', index=False)

# Edit font mnist data so it matches handwritten mnist
fmnist = pd.read_csv('data/raw/TMNIST_Data.csv')
fmnist = fmnist.drop(columns=['names']).rename(columns={'labels': 'label'})
fmnist.to_csv('data/input/synthetic_mnist.csv', sep=',', index=False)
