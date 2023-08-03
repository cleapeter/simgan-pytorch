# Import packages
import importlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

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
    image_path = os.getcwd() + '/results/images/'
    pretraining_path = os.getcwd() + '/results/pretraining/'
    model_path = os.getcwd() + '/results/model_summary/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(pretraining_path):
        os.makedirs(pretraining_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Get data loaders
    real_loader, syn_loader = datasets.get_dataloaders(
        config_file.csv_file_paths,
        batch_size=config['batch_size'],
        transform=config['transform']
    )

    # Define models
    torch.manual_seed(42)
    refiner = model.Refiner(
        transform=config['transform'],
        act_func=config['act_func_r'],
        num_resnet_blocks=config['num_resnet_blocks'],
        kernel_size=config['kernel_size_r']
    ).to(config_file.device)

    discriminator = model.Discriminator(
        act_func=config['act_func_d']
    ).to(config_file.device)

    # Save model summaries as txt files
    utils.save_model_summary(
        model_path, refiner, 'refiner', config['batch_size'])
    utils.save_model_summary(
        model_path, discriminator, 'discriminator', config['batch_size'])

    # Define optimizers
    if config['optimizer'] == 'SGD':
        opt_refiner = torch.optim.SGD(
            refiner.parameters(),
            lr=config['lr_r']
        )
        opt_discriminator = torch.optim.SGD(
            discriminator.parameters(),
            lr=config['lr_d']
        )
    elif config['optimizer'] == 'Adam':
        opt_refiner = torch.optim.Adam(
            refiner.parameters(),
            lr=config['lr_r'],
            betas=(0.5, 0.999)
        )
        opt_discriminator = torch.optim.Adam(
            discriminator.parameters(),
            lr=config['lr_d'],
            betas=(0.5, 0.999)
        )

    # Define loss functions
    def loss_reg(refined, synthetic):
        return torch.mul(F.l1_loss(refined, synthetic), config['lambda_'])

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
        num_steps=config['pre_r_steps'],
        log_steps_loss=config['pre_r_log_steps_loss'],
        log_steps_images=config['pre_r_log_steps_images'],
        real_data=[real_images, real_labels],
        plot_path=image_path + '/pretraining/',
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
        num_steps=config['pre_d_steps'],
        log_steps_loss=config['pre_d_log_steps_loss'],
    )
    print('Finished pretraining of discriminator')

    # Save results of pretraining
    utils.save_metric_plots_pretraining(
        pre_r=pre_r,
        pre_d=pre_d,
        save_path=pretraining_path)

    ### Full Training ###

    # Create image history buffer
    if config['buffer_size'] > 0:
        image_history_buffer = utils.ImageHistoryBuffer(
            (0, config_file.num_channels,
             config['patch_size'], config['patch_size']),
            config['buffer_size'], config['batch_size'])
    else:
        image_history_buffer = None

    for step in range(config['max_steps']):
        (loss_r, loss_r_adv,
         loss_r_reg, loss_d_tot,
         loss_d_mean, loss_d_real,
         loss_d_refined, acc_d_real,
         acc_d_refined) = train_utils.full_training(
            discriminator,
            refiner,
            real_loader,
            syn_loader,
            loss_reg,
            loss_adv,
            opt_discriminator,
            opt_refiner,
            image_history_buffer,
            num_r_steps=config['r_steps_per_iter'],
            current_step=step,
            log_steps_loss=config['log_steps_loss'],
            plot_path=image_path + 'full_training'
        )

        # Save loss every log_loss_full steps
        if (step % config['log_steps_loss'] == 0
            or step == config['max_steps'] - 1):

            # Save checkpoint at end of run (see tuner)
            with tune.checkpoint_dir(step) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (refiner.state_dict(), opt_refiner.state_dict()), path)

            # Save results to tensorboard
            tune.report(
                training_iteration=step, loss_r=loss_r,
                loss_r_adv=loss_r_adv, loss_r_reg=loss_r_reg,
                loss_d_tot=loss_d_tot, loss_d_mean=loss_d_mean,
                loss_d_real=loss_d_real, loss_d_refined=loss_d_refined,
                acc_d_real=acc_d_real, acc_d_refined=acc_d_refined
            )


def run_search(run_dir='/runs/raytune_result'):

    param_space = {
        # Data preprocessing
        'transform': 'minmax',
        # Refiner parameters
        'lr_r': 1e-03,
        'act_func_r': 'LeakyReLU',
        'num_resnet_blocks': 2,
        'kernel_size_r': 3,
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
        'max_steps': 3,
        'min_steps': 1,
        'r_steps_per_iter': 2,
        'log_steps_loss': 10,
        'log_steps_images': 10,
    }

    params_columns = {
        'optimizer': 'opt',
        'lr_r': 'lr_r',
        'lr_d': 'lr_d',
        'activation_func_r': 'act r',
        'activation_func_d': 'act d',
        'lambda_': 'lambda',
        'nb_resnet_blocks': 'res',
        'kernel_size_refiner': 'kernel',
        'r_steps': 'r/d'
    }

    # Stop runs after 1000 iterations if they reach a discriminator loss < 0.2
    # Comment: grace_period is NOT the # of iters but the # of logged metrics
    stopper = TrialPlateauStopper(
        metric='loss_d_mean',
        std=0.1,
        num_results=4,
        grace_period=int(1000 / param_space['log_steps_loss']) + 1,
        metric_threshold=0.2,
        mode='min'
    )

    scheduler = ASHAScheduler(
        metric="loss_d_mean",
        mode="max",
        max_t=param_space['max_steps'],
        grace_period=param_space['min_steps'],)

    reporter = CLIReporter(
        metric_columns=["loss_d_mean", "acc_d_refined", "training_iteration"],
        max_report_frequency=5,
        parameter_columns=params_columns)

    result = tune.run(
        train,
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=param_space,
        num_samples=config_file.num_trials,
        scheduler=scheduler,
        local_dir=os.getcwd() + run_dir,
        progress_reporter=reporter,
        keep_checkpoints_num=1,
        checkpoint_at_end=True,
        stop=stopper
    )

    best_trial = result.get_best_trial('loss', 'min', 'last')
    print(f"Best trial config: {best_trial.config}")


if __name__ == '__main__':
    # Print device
    print('device in config file:', config_file.device)
    run_search()
