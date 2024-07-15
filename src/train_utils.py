# Import packages
import importlib
import os
import torch
import torchvision
from tqdm import tqdm

# Import own files
import config
import utils

# Reload own files
importlib.reload(config)
importlib.reload(utils)


def pretraining_refiner(
    refiner,
    dataloader,
    optimizer,
    loss_reg,
    num_steps,
    log_steps_loss=20,
    log_steps_images=250,
    real_data=None,
    plot_path=None,
):

    # Set model to train mode
    refiner.train()

    # Define batch iterator
    syn_iter = iter(dataloader)
    batch_size = next(syn_iter)[0].shape[0]

    # Save losses to list
    loss_refiner = []
    steps = []

    # Check for plot_path
    if plot_path is not None:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        else:
            # Delete existing images in folder
            for img_file in os.listdir(plot_path):
                if img_file.endswith(".png"):
                    os.remove(plot_path + img_file)

    # Train over batches
    for i in tqdm(range(num_steps)):
        # Get image batch
        syn_data = utils.get_img_batch(syn_iter, dataloader, batch_size)
        syn_iter, syn_images, syn_labels = syn_data
        # Reset gradients
        refiner.zero_grad()
        # Refine synthetic images
        refined_images = refiner(syn_images, syn_labels)
        # Calculate self-regularization loss only
        loss = loss_reg(refined_images, syn_images)
        # Save losses every log_steps_loss to save comp. time due to item()
        if i % log_steps_loss == 0 or i == num_steps - 1:
            loss_refiner.append(loss.item())
            steps.append(i)
        # Backpropagation
        loss.backward()
        # Do optimize step
        optimizer.step()

        # Save images every log_steps_images if save path is given
        if (
            i % log_steps_images == 0 or i == num_steps - 1
        ) and plot_path is not None:
            # Create image grid for 10 different digits
            if real_data is not None:
                real_images, real_labels = real_data
                # Compute indices for digits
                syn_idx = [torch.argmin(abs(syn_labels - j)) for j in range(10)]
                real_idx = [
                    torch.argmin(abs(real_labels - j)) for j in range(10)
                ]
                # Select images based on indices
                selected_syn_images = [syn_images[j] for j in syn_idx]
                selected_refined_images = [refined_images[j] for j in syn_idx]
                selected_real_images = [real_images[j] for j in real_idx]
                selected_images = (
                    selected_syn_images
                    + selected_refined_images
                    + selected_real_images
                )
            else:
                syn_idx = [torch.argmin(abs(syn_labels - j)) for j in range(10)]
                selected_syn_images = [syn_images[j] for j in syn_idx]
                selected_refined_images = [refined_images[j] for j in syn_idx]
                selected_images = selected_syn_images + selected_refined_images
            # Plot image grid
            image_grid = torchvision.utils.make_grid(
                selected_images,
                nrow=10,
                normalize=True,
                value_range=(-1, 1),
                scale_each=True,
            )
            img = torchvision.transforms.ToPILImage()(image_grid)
            img.save(f"{plot_path}/refined{i}.png")

    return loss_refiner, steps


def pretraining_discriminator(
    discriminator,
    refiner,
    real_loader,
    syn_loader,
    optimizer,
    loss_adv,
    num_steps,
    log_steps_loss=20,
):

    # Set model modes
    refiner.eval()
    discriminator.train()

    # Initizialize lists for metrics
    loss_d = []
    loss_d_real = []
    loss_d_syn = []
    acc_d_real = []
    acc_d_syn = []
    steps = []

    # Define batch iterators
    real_iter = iter(real_loader)
    syn_iter = iter(syn_loader)
    batch_size = next(syn_iter)[0].shape[0]

    # Train over batches
    for i in tqdm(range(num_steps)):
        # Get synthetic image batch
        syn_data = utils.get_img_batch(syn_iter, syn_loader, batch_size)
        syn_iter, syn_images, syn_labels = syn_data
        # Get real image batch
        real_data = utils.get_img_batch(real_iter, real_loader, batch_size)
        real_iter, real_images, real_labels = real_data
        # Reset gradients
        discriminator.zero_grad()
        # Train on real images
        real_pred = discriminator(real_images, real_labels)
        real_label = torch.zeros(
            real_pred.size(0), dtype=torch.long, device=config.device
        )
        loss_real = loss_adv(real_pred, real_label)
        # Train on synthetic data
        with torch.no_grad():
            refined_images = refiner(syn_images, syn_labels)
        refined_pred = discriminator(refined_images, syn_labels)
        refined_label = torch.ones(
            refined_pred.size(0), dtype=torch.long, device=config.device
        )
        loss_refined = loss_adv(refined_pred, refined_label)

        # Total loss
        loss = loss_real + loss_refined
        # Log losses
        if i % log_steps_loss == 0 or i == num_steps - 1:
            loss_d.append(loss.item())
            loss_d_real.append(loss_real.item())
            loss_d_syn.append(loss_refined.item())
            acc_d_real.append(utils.calc_acc(real_pred, real_label))
            acc_d_syn.append(utils.calc_acc(refined_pred, refined_label))
            steps.append(i)
        # Backpropagation
        loss.backward()
        # Do optimize step
        optimizer.step()

    return (loss_d, loss_d_real, loss_d_syn, acc_d_real, acc_d_syn, steps)


def full_training(
    discriminator,
    refiner,
    real_loader,
    syn_loader,
    loss_reg,
    loss_adv,
    opt_discriminator,
    opt_refiner,
    image_history_buffer,
    num_d_steps,
    num_r_steps,
    current_step,
    log_steps_loss=20,
    log_steps_images=250,
    plot_path=None,
):

    # Define batch iterator and get batch size
    real_iter = iter(real_loader)
    syn_iter = iter(syn_loader)
    batch_size = next(syn_iter)[0].shape[0]

    # Training Refiner
    refiner.train()
    discriminator.eval()

    tmp_loss_r = []
    tmp_loss_r_reg = []
    tmp_loss_r_adv = []

    for _ in range(num_r_steps):
        # Reset gradients
        refiner.zero_grad()

        # Get synthetic image batch
        syn_data = utils.get_img_batch(syn_iter, syn_loader, batch_size)
        syn_iter, syn_images, syn_labels = syn_data
        # Refine synthetic images
        refined_images = refiner(syn_images, syn_labels)
        refined_pred = discriminator(refined_images, syn_labels)

        # Use labels for real images to get right loss (see paper)
        refined_labels = torch.zeros(
            refined_pred.size(0), dtype=torch.long, device=config.device
        )
        loss_refined_reg = loss_reg(refined_images, syn_images)
        loss_refined_adv = loss_adv(refined_pred, refined_labels)
        loss_refined = loss_refined_reg + loss_refined_adv

        # Save losses every log_steps
        if current_step % log_steps_loss == 0:
            tmp_loss_r.append(loss_refined.item())
            tmp_loss_r_reg.append(loss_refined_reg.item())
            tmp_loss_r_adv.append(loss_refined_adv.item())

        loss_refined.backward()
        opt_refiner.step()

    if current_step % log_steps_loss == 0:
        mean_loss_r = sum(tmp_loss_r) / num_r_steps
        mean_loss_r_adv = sum(tmp_loss_r_adv) / num_r_steps
        mean_loss_r_reg = sum(tmp_loss_r_reg) / num_r_steps

    # Training Discriminator
    refiner.eval()
    discriminator.train()

    tmp_loss_d = []
    tmp_loss_d_refined = []
    tmp_loss_d_real = []
    tmp_acc_d_refined = []
    tmp_acc_d_real = []

    for _ in range(num_d_steps):
        # Reset gradients
        discriminator.zero_grad()

        # Get synthetic image batch
        syn_data = utils.get_img_batch(syn_iter, syn_loader, batch_size)
        syn_iter, syn_images, syn_labels = syn_data
        # Get real image batch
        real_data = utils.get_img_batch(real_iter, real_loader, batch_size)
        real_iter, real_images, real_labels = real_data

        with torch.no_grad():
            refined_images = refiner(syn_images, syn_labels)

        # Use images buffer if not None
        if image_history_buffer is not None:
            history_img_batch = image_history_buffer.get_images()
            image_history_buffer.add_images(refined_images.cpu().data.numpy())

            # Replace half of refined images with images from buffer
            if len(history_img_batch):
                history_img_batch = torch.from_numpy(history_img_batch).to(
                    config.device
                )
                refined_images[: batch_size // 2] = history_img_batch

        # Training on real images
        real_pred = discriminator(real_images, real_labels)
        real_label = torch.zeros(
            real_pred.size(0), dtype=torch.long, device=config.device
        )
        loss_d_real = loss_adv(real_pred, real_label)

        # Training on synthetic images
        refined_pred = discriminator(refined_images, syn_labels)
        refined_label = torch.ones(
            refined_pred.size(0), dtype=torch.long, device=config.device
        )
        loss_d_refined = loss_adv(refined_pred, refined_label)

        # Get total loss
        loss_dis = loss_d_real + loss_d_refined

        # Save losses every log_steps
        if current_step % log_steps_loss == 0:
            tmp_loss_d.append(loss_d_real.item() + loss_d_refined.item())
            tmp_loss_d_real.append(loss_d_real.item())
            tmp_acc_d_real.append(utils.calc_acc(real_pred, real_label))
            tmp_loss_d_refined.append(loss_d_refined.item())
            tmp_acc_d_refined.append(
                utils.calc_acc(refined_pred, refined_label)
            )

        loss_dis.backward()
        opt_discriminator.step()

    if current_step % log_steps_loss == 0:
        mean_loss_d_tot = sum(tmp_loss_d) / num_d_steps
        mean_loss_d_mean = mean_loss_d_tot / 2
        mean_loss_d_real = sum(tmp_loss_d_real) / num_d_steps
        mean_loss_d_refined = sum(tmp_loss_d_refined) / num_d_steps
        mean_acc_d_real = sum(tmp_acc_d_real) / num_d_steps
        mean_acc_d_refined = sum(tmp_acc_d_refined) / num_d_steps

    if (current_step % log_steps_images == 0) and plot_path is not None:

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        with torch.no_grad():
            refined_images = refiner(syn_images, syn_labels)

        syn_idx = [torch.argmin(abs(syn_labels - j)) for j in range(10)]
        real_idx = [torch.argmin(abs(real_labels - j)) for j in range(10)]
        # else:
        #     with torch.no_grad():
        #         refined_images = refiner(syn_images)

        #     # Get 5 random images
        #     syn_idx = random.sample(range(0, batch_size), 5)
        #     real_idx = random.sample(range(0, batch_size), 5)

        selected_syn_images = [syn_images[i] for i in syn_idx]
        selected_refined_images = [refined_images[i] for i in syn_idx]
        selected_real_images = [real_images[i] for i in real_idx]

        selected_images = (
            selected_syn_images + selected_refined_images + selected_real_images
        )

        image_grid = torchvision.utils.make_grid(
            selected_images, nrow=5, normalize=True, scale_each=True
        )
        img = torchvision.transforms.ToPILImage()(image_grid)
        img.save(f"{plot_path}/refined{current_step}.png")

        # Create folder for single images
        current_path = plot_path + "/single_images/" + str(current_step)
        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # Save 3 synthetic and refined images
        for j in range(3):
            torchvision.utils.save_image(
                syn_images[j],
                fp=current_path + f"/synthetic{j}-{current_step}.png",
                normalize=True,
            )
            torchvision.utils.save_image(
                refined_images[j],
                fp=current_path + f"/refined{j}-{current_step}.png",
                normalize=True,
            )

    if current_step % log_steps_loss == 0:
        return (
            mean_loss_r,
            mean_loss_r_adv,
            mean_loss_r_reg,
            mean_loss_d_tot,
            mean_loss_d_mean,
            mean_loss_d_real,
            mean_loss_d_refined,
            mean_acc_d_real,
            mean_acc_d_refined,
        )
    else:
        # Return dummy variables
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
