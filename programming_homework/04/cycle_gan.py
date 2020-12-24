# cycle_gan.py
#
# This is the main training file for the CycleGAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyper-parameters (saves results to samples_cyclegan/):
#       python cycle_gan.py
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU), then you can obtain better results by
#    increasing the number of filters used in the generator and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import os
import argparse

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Imageio and Matplotlib imports
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Local imports
import utils
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator

import gc
from time import sleep


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.load, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.load, 'D_X.pkl')
    D_Y_path = os.path.join(opts.load, 'D_Y.pkl')

    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y

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


def merge_images(sources, targets, opts, k=10):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    batch_size, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for (idx, s, t) in (zip(range(row ** 2), sources, targets, )):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)


def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = utils.to_data(fixed_X), utils.to_data(fake_X)
    Y, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    merged = 255 * ((merged + 1)/2)
    imageio.imsave(path, merged.astype(np.uint8))
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    merged = 255 * ((merged + 1)/2)
    imageio.imsave(path, merged.astype(np.uint8))
    print('Saved {}'.format(path))


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    if opts.load:
        G_XtoY, G_YtoX, D_X, D_Y = load_checkpoint(opts)
    else:
        G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = utils.to_var(test_iter_X.next()[0])
    fixed_Y = utils.to_var(test_iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    for iteration in range(1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            del iter_X, iter_Y
            gc.collect()
            while True:
                try:
                    iter_X = iter(dataloader_X)
                    iter_Y = iter(dataloader_Y)
                    break
                except OSError:
                    print("Failed to allocate memory, try again in 5 seconds...")
                    sleep(5)

        images_X, labels_X = iter_X.next()
        images_X, labels_X = utils.to_var(images_X), utils.to_var(labels_X).long().squeeze()

        images_Y, labels_Y = iter_Y.next()
        images_Y, labels_Y = utils.to_var(images_Y), utils.to_var(labels_Y).long().squeeze()


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        #########################################
        ##             FILL THIS IN            ##
        #########################################

        # Train with real images
        d_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        D_X_loss = torch.sum(torch.pow(D_X(images_X) - 1, 2))/(2*opts.batch_size)
        D_Y_loss = torch.sum(torch.pow(D_Y(images_Y) - 1, 2))/(2*opts.batch_size)

        d_real_loss = D_X_loss + D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()

        # Train with fake images
        d_optimizer.zero_grad()

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 3. Compute the loss for D_X
        D_X_loss = torch.sum(torch.pow(D_X(fake_X), 2))/(2*opts.batch_size)

        # 4. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 5. Compute the loss for D_Y
        D_Y_loss = torch.sum(torch.pow(D_Y(fake_Y), 2))/(2*opts.batch_size)

        d_fake_loss = D_X_loss + D_Y_loss
        d_fake_loss.backward()
        d_optimizer.step()



        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================


        #########################################
        ##    FILL THIS IN: Y--X-->Y CYCLE     ##
        #########################################
        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        g_loss = torch.sum(torch.pow(D_X(fake_X) - 1, 2))/(opts.batch_size)

        reconstructed_Y = G_XtoY(fake_X)
        # 3. Compute the cycle consistency loss (the reconstruction loss)
        cycle_consistency_loss = (images_Y - reconstructed_Y).abs().sum()/(images_Y).size(0)

        g_loss += opts.lambda_cycle * cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()


        #########################################
        ##    FILL THIS IN: X--Y-->X CYCLE     ##
        #########################################

        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain Y
        g_loss = torch.sum(torch.pow(D_Y(fake_Y) - 1, 2))/(opts.batch_size)

        reconstructed_X = G_YtoX(fake_Y)
        # 3. Compute the cycle consistency loss (the reconstruction loss)
        cycle_consistency_loss = (images_X - reconstructed_X).abs().sum()/(images_X).size(0)

        g_loss += opts.lambda_cycle * cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                  'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    iteration, opts.train_iters, d_real_loss.item(), D_Y_loss.item(),
                    D_X_loss.item(), d_fake_loss.item(), g_loss.item()))

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type=opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_emoji_loader(emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)

    generate_gif(opts.sample_dir, keyword='X-Y')
    generate_gif(opts.sample_dir, keyword='Y-X')


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')
    parser.add_argument('--lambda_cycle', type=float, default=0.015, help='weight parameter for cycle loss')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=5000, help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.3)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--X', type=str, default='Apple', choices=['Apple', 'Windows'], help='Choose the type of images for domain X.')
    parser.add_argument('--Y', type=str, default='Windows', choices=['Apple', 'Windows'], help='Choose the type of images for domain Y.')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='results/samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=200)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=1000)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    if opts.load:
        opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
        opts.sample_every = 20

    print_opts(opts)
    main(opts)
