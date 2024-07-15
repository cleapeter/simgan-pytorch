import importlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import own files
import config as config_file
import datasets
import model_simgan as model
import train_utils
import utils

# Reload own files
importlib.reload(config_file)
importlib.reload(datasets)
importlib.reload(model)
importlib.reload(train_utils)
importlib.reload(utils)


def train(config):
    # Prepare folders to save images, pretraining results and model summary
    # current working directory is the folder for the current training run
    image_path = os.getcwd() + "/results/images/"
    pretraining_path = os.getcwd() + "/results/pretraining/"
    model_path = os.getcwd() + "/results/model_summary/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(pretraining_path):
        os.makedirs(pretraining_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Get data loaders
    real_loader, syn_loader = datasets.get_dataloaders(
        config_file.csv_file_paths,
        batch_size=config["batch_size"],
        transform=config["transform"],
    )

    # Define models
    torch.manual_seed(42)
    refiner = model.Refiner(
        transform=config["transform"],
        act_func=config["act_func_r"],
        num_resnet_blocks=config["num_resnet_blocks"],
        kernel_size=config["kernel_size_r"],
    ).to(config_file.device)

    discriminator = model.Discriminator(act_func=config["act_func_d"]).to(
        config_file.device
    )

    # Save model summaries as txt files
    utils.save_model_summary(
        model_path, refiner, "refiner", config["batch_size"]
    )
    utils.save_model_summary(
        model_path, discriminator, "discriminator", config["batch_size"]
    )

    # Define optimizers
    if config["optimizer"] == "SGD":
        opt_refiner = torch.optim.SGD(refiner.parameters(), lr=config["lr_r"])
        opt_discriminator = torch.optim.SGD(
            discriminator.parameters(), lr=config["lr_d"]
        )
    elif config["optimizer"] == "Adam":
        opt_refiner = torch.optim.Adam(
            refiner.parameters(), lr=config["lr_r"], betas=(0.5, 0.999)
        )
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999)
        )

    # Define loss functions
    def loss_reg(refined, synthetic):
        return torch.mul(F.l1_loss(refined, synthetic), config["lambda_"])

    loss_adv = nn.CrossEntropyLoss()

    # One batch of real images for image grid
    real_data = next(iter(real_loader))
    real_images = real_data[0].to(config_file.device)
    real_labels = real_data[1].to(config_file.device)

    # Pretraining of refiner
    print("Pretraining of refiner...")
    pre_r = train_utils.pretraining_refiner(
        refiner,
        syn_loader,
        opt_refiner,
        loss_reg,
        num_steps=config["pre_r_steps"],
        log_steps_loss=config["pre_r_log_steps_loss"],
        log_steps_images=config["pre_r_log_steps_images"],
        real_data=[real_images, real_labels],
        plot_path=image_path + "/pretraining/",
    )
    print("Finished pretraining of refiner")

    # Pretraining of discriminator
    print("Pretraining of discriminator...")
    pre_d = train_utils.pretraining_discriminator(
        discriminator,
        refiner,
        real_loader,
        syn_loader,
        opt_discriminator,
        loss_adv,
        num_steps=config["pre_d_steps"],
        log_steps_loss=config["pre_d_log_steps_loss"],
    )
    print("Finished pretraining of discriminator")

    # Save results of pretraining
    utils.save_metric_plots_pretraining(
        pre_r=pre_r, pre_d=pre_d, save_path=pretraining_path
    )

    ### Full Training ###

    # Create image history buffer
    if config["buffer_size"] > 0:
        image_history_buffer = utils.ImageHistoryBuffer(
            config["buffer_size"], config["batch_size"]
        )
    else:
        image_history_buffer = None

    for step in range(config["max_steps"]):
        (
            loss_r,
            loss_r_adv,
            loss_r_reg,
            loss_d_tot,
            loss_d_mean,
            loss_d_real,
            loss_d_refined,
            acc_d_real,
            acc_d_refined,
        ) = train_utils.full_training(
            discriminator,
            refiner,
            real_loader,
            syn_loader,
            loss_reg,
            loss_adv,
            opt_discriminator,
            opt_refiner,
            image_history_buffer,
            num_d_steps=1,
            num_r_steps=config["r_steps_per_iter"],
            current_step=step,
            log_steps_loss=config["log_steps_loss"],
            plot_path=image_path + "full_training",
        )

    print("Finished!")


if __name__ == "__main__":

    param_space = {
        # Data preprocessing
        "transform": "minmax",
        # Refiner parameters
        "lr_r": 1e-03,
        "act_func_r": "LeakyReLU",
        "num_resnet_blocks": 2,
        "kernel_size_r": 3,
        # Discriminator parameters
        "lr_d": 1e-03,
        "act_func_d": "LeakyReLU",
        # General training parameters
        "batch_size": 16,
        "buffer_size": 32,
        "lambda_": 10,
        "optimizer": "Adam",
        # Pretraining parameters
        "pre_r_steps": 50,
        "pre_r_log_steps_loss": 10,
        "pre_r_log_steps_images": 10,
        "pre_d_steps": 50,
        "pre_d_log_steps_loss": 10,
        # Full training parameters
        "max_steps": 3,
        "min_steps": 1,
        "r_steps_per_iter": 2,
        "log_steps_loss": 10,
        "log_steps_images": 10,
    }

    train(param_space)
