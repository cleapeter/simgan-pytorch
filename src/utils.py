# Import packages
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchinfo import summary

# Import own files
import config

# Reload own files
importlib.reload(config)


def get_img_batch(batch_iter, dataloader, batch_size):
    """Function returns one batch of images including corresponding labels.

    Returns:
        batch_iter: updated batch iterator
        images (torch.tensor): batch of images
        labels (torch.tensor): batch of labels
    """

    try:
        data = next(batch_iter)
    except StopIteration:
        batch_iter = iter(dataloader)
        data = next(batch_iter)
    images = data[0].to(config.device)
    labels = data[1].to(config.device)
    # If returned batch is smaller than expected batch size,
    # reinitialize iterator
    if images.shape[0] != batch_size:
        batch_iter = iter(dataloader)
        data = next(batch_iter)
        images = data[0].to(config.device)
        labels = data[1].to(config.device)

    return batch_iter, images, labels


def save_model_summary(save_path, network, network_name, batch_size):
    """Function saves the summary of a given network architecture
    as a text file.

    Args:
        save_path (str): where to save text file
        network (torch.nn.Module): network/model
        network_name (str): name of text file
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + f"model_{network_name}.txt", "w")
    f.write(
        str(
            summary(
                network,
                input_size=[
                    (
                        batch_size,
                        config.num_channels,
                        config.img_height,
                        config.img_width,
                    ),
                    (batch_size, 1),
                ],
                dtypes=[torch.float, torch.long],
                col_names=[
                    "input_size",
                    "kernel_size",
                    "output_size",
                    "num_params",
                ],
                verbose=0,
            )
        )
    )
    f.close()


def calc_acc(output, label):
    """Function calculates the accuracy of the discriminator model."""

    softmax_output = torch.nn.functional.softmax(output, dim=1)
    acc = softmax_output.max(dim=1)[1].cpu().numpy() == label.cpu().numpy()

    return acc.mean()


def save_metric_plots_pretraining(pre_r, pre_d=None, save_path="pretraining"):
    """Function creates plot for metrics of pretrained refiner and
    discrminator.

    Args:
        pre_r (list): list containing loss for each step
        pre_d (list): list containing lists for losses and accuracy
        save_path (string): where to save plots
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pre_r_loss, steps_r = pre_r

    # Pretraining refiner
    plt.figure(figsize=(10, 7))
    plt.plot(steps_r, pre_r_loss)
    plt.xlabel("Steps")
    plt.ylabel("Self-regularization loss")
    plt.title("Pretrained Refiner: self-reg loss")
    plt.savefig(save_path + "pre_r_loss.png")
    plt.close()

    if pre_d is not None:
        (
            pre_d_loss,
            pre_d_loss_real,
            pre_d_loss_syn,
            pre_d_acc_real,
            pre_d_acc_syn,
            steps_d,
        ) = pre_d

        pre_d_loss_mean = [i / 2 for i in pre_d_loss]

        # Pretraining discriminator: loss
        plt.figure(figsize=(10, 7))
        plt.plot(steps_d, pre_d_loss_mean, label="mean loss")
        plt.plot(steps_d, pre_d_loss_real, label="real loss")
        plt.plot(steps_d, pre_d_loss_syn, label="syn loss")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Adversarial loss")
        plt.title("Pretrained Discriminator: Adversarial Loss")
        plt.savefig(save_path + "pre_d_loss.png")
        plt.close()

        # Pretraining discriminator: accuracy
        plt.figure(figsize=(10, 7))
        plt.plot(steps_d, pre_d_acc_real, label="real acc")
        plt.plot(steps_d, pre_d_acc_syn, label="syn acc")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("Pretrained Discriminator: Accuracy")
        plt.savefig(save_path + "pre_d_acc.png")
        plt.close()


class ImageHistoryBuffer(object):
    """Code copied from
    https://github.com/mjdietzx/SimGAN/blob/master/utils/image_history_buffer.py
    and modified (one fix)
    """

    def __init__(self, max_size, batch_size):
        """Class creates image history buffer which stores earlier generated
        images as a reference.

        Args:
            max_size (int): max. number of images stored in buffer
        """
        self.image_history_buffer = np.zeros(
            shape=(0, config.num_channels, config.img_height, config.img_width)
        )
        self.max_size = max_size
        self.batch_size = batch_size

    def add_images(self, images, num_to_add=None):
        """Function adds images to buffer.

        Args:
            images (np.array): images to be added to buffer
            num_to_add (int): number of images to be added. Default is
                batch_size / 2.
        """
        if not num_to_add:
            num_to_add = self.batch_size // 2

        if len(self.image_history_buffer) < self.max_size:
            self.image_history_buffer = np.append(
                self.image_history_buffer, images[:num_to_add], axis=0
            )
        elif len(self.image_history_buffer) == self.max_size:
            self.image_history_buffer[:num_to_add] = images[:num_to_add]
        else:
            assert False

        np.random.shuffle(self.image_history_buffer)

    def get_images(self, num_to_get=None):
        """Function gets a random sample of images from the buffer.

        Args:
            num_to_get (int): number of images to output. Default is
                batch_size / 2.

        Returns:
            images (np.array): random sample of images
        """
        if not num_to_get:
            num_to_get = self.batch_size // 2

        try:
            return self.image_history_buffer[:num_to_get]
        except IndexError:
            False
