# Import packages
import importlib
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src folder to system path
sys.path.insert(0, './src')

# Import own files
import config
import datasets
if config.model == 'SimGAN':
    import model_simgan as model
import train_utils
import utils

# Reload own files
importlib.reload(config)
importlib.reload(datasets)
importlib.reload(model)
importlib.reload(train_utils)
importlib.reload(utils)


def get_param_space():
    param_space = {
        # Data preprocessing
        'transform': config.transform,
        # Refiner parameters
        'lr_r': 1e-03,
        'act_func_r': 'LeakyReLU',
        'num_resnet_blocks': 2,
        # Discriminator parameters
        'lr_d': 1e-03,
        'act_func_d': 'LeakyReLU',
        # General training parameters
        'batch_size': 32,
        'buffer_size': 32,
        'lambda_': 10,
        'optimizer': 'Adam',
        # Pretraining parameters
        'pre_r_steps': 50,
        'pre_r_log_steps_loss': 10,
        'pre_r_log_steps_images': 10,
        'pre_d_steps': 50,
        'pre_d_log_steps_loss': 10,
        # Full training parameters
        'max_iter': 3,
        'r_steps_per_iter': 2,
        'log_steps_loss': 10,
        'log_steps_images': 10,
    }

    return param_space


def test_pretraining_refiner(param_space):
    """
    Test of pretraining refiner function in src/train_utils.py.
    """

    # Get data loaders
    real_loader, syn_loader = datasets.get_dataloaders(
        config.csv_file_paths,
        batch_size=param_space['batch_size'],
        transform=param_space['transform']
    )
    print('Finished preparing dataloaders')

    # Define models
    refiner = model.Refiner().to(config.device)

    # Save model summaries as txt files to this path
    model_path = os.getcwd() + '/tests/model_summary/'
    utils.save_model_summary(
        model_path, refiner, 'refiner', param_space['batch_size'])

    # Define optimizers
    if param_space['optimizer'] == 'SGD':
        opt_refiner = torch.optim.SGD(
            refiner.parameters(),
            lr=param_space['lr_r'])
    elif param_space['optimizer'] == 'Adam':
        opt_refiner = torch.optim.Adam(
            refiner.parameters(),
            lr=param_space['lr_r'],
            betas=(0.5, 0.999))

    # Define loss functions
    def loss_reg(refined, synthetic):
        return torch.mul(F.l1_loss(refined, synthetic), param_space['lambda_'])

    # One batch of real images for image grid
    real_data = next(iter(real_loader))
    real_images = real_data[0].to(config.device)
    real_labels = real_data[1].to(config.device)

    # Pretraining of refiner
    print('Pretraining of refiner...')
    pre_r = train_utils.pretraining_refiner(
        refiner,
        syn_loader,
        opt_refiner,
        loss_reg,
        num_steps=param_space['pre_r_steps'],
        log_steps_loss=param_space['pre_r_log_steps_loss'],
        log_steps_images=param_space['pre_r_log_steps_images'],
        real_data=[real_images, real_labels],
        plot_path='./tests/images/pretraining/refiner/'
    )
    print('Finished pretraining of refiner')

    save_path = './tests/images/pretraining/refiner/'
    utils.save_metric_plots_pretraining(
        pre_r=pre_r, pre_d=None, save_path=save_path)


def test_pretraining_discriminator(param_space):
    """
    Test of pretraining discriminator function in src/train_utils.py.
    """

    # Get data loaders
    real_loader, syn_loader = datasets.get_dataloaders(
        config.csv_file_paths,
        batch_size=param_space['batch_size'],
        transform=param_space['transform']
    )
    print('Finished preparing dataloaders')

    # Define models
    refiner = model.Refiner().to(config.device)
    discriminator = model.Discriminator().to(config.device)

    # Save model summaries as txt files to this path
    model_path = os.getcwd() + '/tests/model_summary/'
    utils.save_model_summary(
        model_path, refiner, 'refiner', param_space['batch_size'])
    utils.save_model_summary(
        model_path, discriminator, 'discriminator', param_space['batch_size'])

    # Define optimizers
    if param_space['optimizer'] == 'SGD':
        opt_refiner = torch.optim.SGD(
            refiner.parameters(), lr=param_space['lr_r'])
        opt_discriminator = torch.optim.SGD(
            discriminator.parameters(), lr=param_space['lr_d'])
    elif param_space['optimizer'] == 'Adam':
        opt_refiner = torch.optim.Adam(
            refiner.parameters(), lr=param_space['lr_r'], betas=(0.5, 0.999))
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(),
            lr=param_space['lr_d'],
            betas=(0.5, 0.999))

    # Define loss functions
    def loss_reg(refined, synthetic):
        return torch.mul(F.l1_loss(refined, synthetic), param_space['lambda_'])
    loss_adv = nn.CrossEntropyLoss()

    # One batch of real images for image grid
    real_data = next(iter(real_loader))
    real_images = real_data[0].to(config.device)
    real_labels = real_data[1].to(config.device)

    # Pretraining of refiner
    print('Pretraining of refiner...')
    pre_r = train_utils.pretraining_refiner(
        refiner,
        syn_loader,
        opt_refiner,
        loss_reg,
        num_steps=param_space['pre_r_steps'],
        log_steps_loss=param_space['pre_r_log_steps_loss'],
        log_steps_images=param_space['pre_r_log_steps_images'],
        real_data=[real_images, real_labels],
        plot_path='./tests/images/pretraining/discriminator/'
    )
    print('Finished pretraining of refiner')

    # Pretraining of discriminator
    print('Pretraining of discriminator...')
    pre_d = train_utils.pretraining_discriminator(
        discriminator,
        refiner,
        real_loader,
        syn_loader,
        opt_discriminator,
        loss_adv,
        num_steps=param_space['pre_d_steps'],
        log_steps_loss=param_space['pre_d_log_steps_loss'],
    )
    print('Finished pretraining of discriminator')

    utils.save_metric_plots_pretraining(
        pre_r=pre_r,
        pre_d=pre_d,
        save_path='./tests/images/pretraining/discriminator/'
    )


if __name__ == "__main__":
    test_pretraining_refiner(get_param_space())
    test_pretraining_discriminator(get_param_space())
