# Import packages
import importlib
import torch
import torch.nn as nn

# Import own files
import config

# Reload own files
importlib.reload(config)


class ResNetBlock(nn.Module):

    def __init__(self, in_features, num_features=64,
                 kernel_size=3, act_func='LeakyReLU'):
        super().__init__()

        # Set activation function
        if act_func == 'ReLU':
            self.act_func = nn.ReLU()
        elif act_func == 'LeakyReLU':
            self.act_func = nn.LeakyReLU(0.2)

        # Define model layers
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            bias=False,
            padding='same',
            stride=1)
        self.conv1_bn = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            bias=False,
            padding='same',
            stride=1)
        self.conv2_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):

        identity = x
        # Conv layer #1
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.act_func(out)
        # Conv layer #2
        out = self.conv2(x)
        out = self.conv2_bn(out)
        out += identity
        out = self.act_func(out)

        return out


class Refiner(nn.Module):

    def __init__(self, num_features=64, kernel_size=3,
                 act_func='LeakyReLU', num_resnet_blocks=2):
        super().__init__()

        # Set activation function
        if act_func == 'ReLU':
            self.act_func = nn.ReLU()
        elif act_func == 'LeakyReLU':
            self.act_func = nn.LeakyReLU(0.2)

        # Define model layers
        # Embedding layer for labels (conditional GAN)
        self.embedding = nn.Embedding(
            config.num_classes, config.img_height * config.img_width)
        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels + 1,  # add +1 due to embedding
            out_channels=num_features,
            kernel_size=kernel_size,
            bias=False,
            padding='same',
            stride=1)
        self.conv1_bn = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=config.num_channels,
            kernel_size=1,
            padding='same',
            stride=1)
        blocks = [ResNetBlock(num_features, num_features,
                              kernel_size, act_func)
                  for _ in range(num_resnet_blocks)]
        self.resnet_blocks = nn.Sequential(*blocks)
        self.tanh = nn.Tanh()

    def forward(self, x, labels):

        # Use labels in embedding layer as condition
        embedding = self.embedding(labels).view(
            labels.shape[0],
            config.num_channels,
            config.img_height,
            config.img_width)
        # Add embedding as a second channel
        x = torch.cat([x, embedding], dim=1)
        # Conv layer #1
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.act_func(x)
        # ResNet blocks
        x = self.resnet_blocks(x)
        # Conv layer #2
        x = self.conv2(x)
        # No batch norm after last layer
        # If normalized to [-1, 1], use tanh, otherwise no activation function
        if config.transform == 'minmax':
            x = self.tanh(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, act_func='LeakyReLU'):
        super().__init__()

        # Set activation function
        if act_func == 'ReLU':
            self.act_func = nn.ReLU()
        elif act_func == 'LeakyReLU':
            self.act_func = nn.LeakyReLU(0.2)

        # Define model layers
        self.embedding = nn.Embedding(
            config.num_classes, config.img_height * config.img_width)
        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels + 1,
            out_channels=96,
            kernel_size=7,
            bias=False,
            padding=2,
            stride=4)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=64,
            kernel_size=5,
            bias=False,
            padding=1,
            stride=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1,
            stride=2)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            bias=False,
            stride=1)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=2,
            kernel_size=1,
            stride=1)

    def forward(self, x, labels):

        # Use labels in embedding layer as condition
        embedding = self.embedding(labels).view(
            labels.shape[0],
            1,
            config.img_height,
            config.img_width)
        # Add embedding as a second channel
        x = torch.cat([x, embedding], dim=1)
        # Conv layer # 1
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.act_func(x)
        # Conv layer #2
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.act_func(x)
        # Conv layer #3
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.act_func(x)
        # Conv layer #4
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.act_func(x)
        # Conv layer #5
        # No batch normalization after last layer
        # No activation function as softmax is used in loss function
        x = self.conv5(x)
        x = x.view(-1, 2)

        return x
