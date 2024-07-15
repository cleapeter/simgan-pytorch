# WIP: SimGAN PyTorch Implementation

See current project status down below. The project has not yet been completed.

## Introduction
The repository contains the PyTorch implementation of SimGAN, an image-to-image translation network published by Shrivastava et al. in 2016, which uses an generative adversarial network to improve the realism of synthetic images. You can find the paper on [Arxiv](https://arxiv.org/pdf/1612.07828.pdf) and the corresponding blog post [here](https://machinelearning.apple.com/research/gan).

This repository focuses on refining images of typeface digits so they will look like handwritten digits. The projects uses Ray Tune for hyperparameter tuning. Given a defined search space of hyperparameters, RayTune will randomly sample a combination of those parameters, train the model and find the best performing one. Results are saved to tensorboard files.

For hands-on examples on how to use Ray Tune with Pytorch, I recommend these two examples: [DebuggerCafe](https://debuggercafe.com/hyperparameter-tuning-with-pytorch-and-ray-tune/) and [official PyTorch documentation](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html).

## How to use the repository?
- Clone the repository and create a conda environment using the following commands in the terminal:
    ```console
    chmod +x conda-env.sh # if file is not executable
    ./conda-env.sh
    ```
- Install all requirements using ``pip install -r requirements.txt``
- Follow the instructions in ``data/input/create_input_csv.py`` to download all data.
- Modify ``search_and_train.py``:
    - Define the search space in the dictionary called param_space in run_search() function. See also [this documentation](https://docs.ray.io/en/latest/tune/api_docs/search_space.html).
    - Optional: Make changes to the reporter and the parameters it prints. This is not important for the overall performance or training of the model but only for the user to see the current metrics and model parameters in the terminal.
    - Optional: Change the early stopping criteria in TrialPlateauStopper. The default stops runs that reach a discriminator loss of 0.2 after 1000 iterations.
- Run ``search_and_train.py``:
    - Results are saved to a folder in runs/raytune_results. This folder contains separate subfolders for each trial. Each subfolder includes one folder called results which contains the following folders:
        - images: refined images for different steps (pretraining + full training)
        - pretraining: metrics plots and data for the pretrained models 
        - model_summary: text file describing architecture of refiner and discriminator
    - The weigths of the refiner are saved each log_steps_loss-iteration. However, only the checkpoint saved last is kept. Each runs contains a tensorboard file (events.out.tfevents...) and a json file containing the parameters used for this run.

## Current project status
Last updated: 2023-08-03

Status: Work In Progress

The project is not yet finished and needs further adjustments to be runnable. 

- Problem with Ray Tune: 
    - test code on GPU
    - downgrade to 2.1.0
    - try pip package
