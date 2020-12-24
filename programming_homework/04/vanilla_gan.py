# vanilla_gan.py
#
# This is the main training file for the vanilla GAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyper-parameters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import os
import argparse

import warnings
warnings.filterwarnings("ignore")

# Numpy & Imageio and Matplotlib imports
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Local imports
import utils
from data_loader import get_emoji_loader
from models import DCGenerator, DCDiscriminator

import gc
from time import sleep

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G, D):
    """Prints model information for the generators and discriminators.
    """
    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim)
    D = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G, D)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D


def checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G.pkl')
    D_path = os.path.join(opts.checkpoint_dir, 'D.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def generate_gif(directory_path, keyword=None):
    images = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".png") and (keyword is None or keyword in filename):
            img_path = os.path.join(directory_path, filename)
            print("adding image {}".format(img_path))
            images.append(imageio.imread(img_path))

    if keyword:
        imageio.mimsave(
            os.path.join(directory_path, 'anim_{}.gif'.format(keyword)), images)
    else:
        imageio.mimsave(os.path.join(directory_path, 'anim.gif'), images)


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)

    grid = create_image_grid(generated_images)
    grid = np.uint8(255 * (grid + 1) / 2)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imsave(path, grid)
    print('Saved {}'.format(path))


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(opts)

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator
    fixed_noise = sample_noise(100, opts.noise_size)  # 100 x noise_size x 1 x 1

    train_iter = iter(train_dataloader)

    iter_per_epoch = len(train_iter)
    total_train_iters = opts.train_iters

    losses = {"iteration": [], "D_fake_loss": [], "D_real_loss": [], "G_loss": []}
    gp_weight = 10

    for iteration in range(1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            del train_iter
            gc.collect()
            while True:
                try:
                    train_iter = iter(train_dataloader)
                    break
                except OSError:
                    print("Failed to allocate memory, try again in 5 seconds...")
                    sleep(5)

        real_images, real_labels = train_iter.next()
        real_images, real_labels = utils.to_var(real_images), utils.to_var(real_labels).long().squeeze()

        batch_size = real_images.size(0)

        ################################################
        ###         TRAIN THE DISCRIMINATOR         ####
        ################################################

        for d_i in range(opts.d_train_iters):

            d_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            D_real_loss = sum((D(real_images) - 1)**2)/(2*opts.batch_size)

            # 2. Sample noise
            # noise = sample_noise(opts.batch_size, opts.noise_size)
            # use actual batch size, not maximum batch size
            noise = sample_noise(real_images.data.shape[0], opts.noise_size)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            D_fake_loss = sum(D(fake_images)**2)/(2*opts.batch_size)

            # ---- Gradient Penalty ----
            if opts.gradient_penalty:
                alpha = torch.rand(real_images.shape[0], 1, 1, 1)
                alpha = alpha.expand_as(real_images).cuda()
                interp_images = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True).cuda()
                D_interp_output = D(interp_images)

                gradients = torch.autograd.grad(outputs=D_interp_output, inputs=interp_images,
                                                grad_outputs=torch.ones(D_interp_output.size()).cuda(),
                                                create_graph=True, retain_graph=True)[0]
                gradients = gradients.view(real_images.shape[0], -1)
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

                gp = gp_weight * gradients_norm.mean()
            else:
                gp = 0.0

            # 5. Compute the total discriminator loss
            D_total_loss = D_real_loss + D_fake_loss + gp

            D_total_loss.backward()
            d_optimizer.step()

        ###########################################
        ###          TRAIN THE GENERATOR        ###
        ###########################################

        g_optimizer.zero_grad()

        # FILL THIS IN
        # 1. Sample noise
        noise = sample_noise(opts.batch_size, opts.noise_size)

        # 2. Generate fake images from the noise
        fake_images = G(noise)

        # 3. Compute the generator loss
        G_loss = sum((D(fake_images) - 1)**2)/opts.batch_size

        G_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            losses['iteration'].append(iteration)
            losses['D_real_loss'].append(D_real_loss.item())
            losses['D_fake_loss'].append(D_fake_loss.item())
            losses['G_loss'].append(G_loss.item())
            print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))


        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(G, fixed_noise, iteration, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G, D, opts)

    plt.figure()
    plt.plot(losses['iteration'], losses['D_real_loss'], label='D_real')
    plt.plot(losses['iteration'], losses['D_fake_loss'], label='D_fake')
    plt.plot(losses['iteration'], losses['G_loss'], label='G')
    plt.legend()
    plt.savefig(os.path.join(opts.checkpoint_dir, 'losses.png'))
    plt.close()
    return G, D

def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    train_dataloader, _ = get_emoji_loader(opts.emoji, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    training_loop(train_dataloader, opts)

    generate_gif(opts.sample_dir)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--spectral_norm', type=bool, default=False, help='on/off for spectral normalization')
    parser.add_argument('--gradient_penalty', type=bool, default=False, help='on/off for gradient_penalty')
    parser.add_argument('--d_train_iters', type=int, default=1)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='results/samples_vanilla')
    parser.add_argument('--log_step', type=int , default=200)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=1000)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)
