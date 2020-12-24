"""
Colorization of CIFAR-10 Horses via classification.
"""

from __future__ import print_function
import argparse
import os
import math
import numpy as np
import numpy.random as npr
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # switch backend
import matplotlib.pyplot as plt
from load_data import load_cifar10

HORSE_CATEGORY = 7


######################################################################
# Data related code
######################################################################
def get_rgb_cat(xs, colors):
    """
    Get color categories given RGB values. This function doesn't
    actually do the work, instead it splits the work into smaller
    chunks that can fit into memory, and calls helper function
    _get_rgb_cat

    Args:
      xs: float numpy array of RGB images in [B, C, H, W] format
      colors: numpy array of color categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    """
    if np.shape(xs)[0] < 100:
        return _get_rgb_cat(xs)
    batch_size = 100
    nexts = []
    for i in range(0, np.shape(xs)[0], batch_size):
        next = _get_rgb_cat(xs[i:i + batch_size, :, :, :], colors)
        nexts.append(next)
    result = np.concatenate(nexts, axis=0)
    return result


def _get_rgb_cat(xs, colors):
    """
    Get color categories given RGB values. This is done by choosing
    the color in `colors` that is the closest (in RGB space) to
    each point in the image `xs`. This function is a little memory
    intensive, and so the size of `xs` should not be too large.

    Args:
      xs: float numpy array of RGB images in [B, C, H, W] format
      colors: numpy array of color categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    """
    num_colors = np.shape(colors)[0]
    xs = np.expand_dims(xs, 0)
    cs = np.reshape(colors, [num_colors, 1, 3, 1, 1])
    dists = np.linalg.norm(xs - cs, axis=2)  # 2 = color axis
    cat = np.argmin(dists, axis=0)
    cat = np.expand_dims(cat, axis=1)
    return cat


def get_cat_rgb(cats, colors):
    """
    Get RGB colors given the color categories

    Args:
      cats: integer numpy array of color categories
      colors: numpy array of color categories and their RGB values
    Returns:
      numpy tensor of RGB colors
    """
    return colors[cats]


def process(xs, ys, max_pixel=256.0):
    """
    Pre-process CIFAR10 images by taking only the horse category,
    shuffling, and have color values be bound between 0 and 1

    Args:
      xs: the color RGB pixel values
      ys: the category labels
      max_pixel: maximum pixel value in the original data
    Returns:
      xs: value normalized and shuffled color images
      gray: gray scale images, also normalized so values are between 0 and 1
    """
    xs = xs / max_pixel
    xs = xs[np.where(ys == HORSE_CATEGORY)[0], :, :, :]
    npr.shuffle(xs)
    gray = np.mean(xs, axis=1, keepdims=True)
    return (xs, gray)


def get_batch(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i + batch_size, :, :, :]
        batch_y = y[i:i + batch_size, :, :, :]
        yield (batch_x, batch_y)


def plot(input, gtlabel, output, colors, path):
    """
    Generate png plots of input, ground truth, and outputs

    Args:
      input: the gray scale input to the colorization CNN
      gtlabel: the ground truth categories for each pixel
      output: the predicted categories for each pixel
      colors: numpy array of color categories and their RGB values
      path: output path
    """
    grey = np.transpose(input[:10, :, :, :], [0, 2, 3, 1])
    gtcolor = get_cat_rgb(gtlabel[:10, 0, :, :], colors)
    predcolor = get_cat_rgb(output[:10, 0, :, :], colors)

    img = np.vstack([np.hstack(np.tile(grey, [1, 1, 1, 3])),
                    np.hstack(gtcolor),
                    np.hstack(predcolor)])
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


######################################################################
# MODELS
######################################################################

class MyConv2d(nn.Module):
    """
    Our simplified implementation of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)


class MyDilatedConv2d(MyConv2d):
    """
    Dilated Convolution 2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(MyDilatedConv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size)
        self.dilation = dilation

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding,
                        dilation=self.dilation)


class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colors):
        super(CNN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            # nn.Conv2d(1, num_filters, kernel_size=kernel, padding=padding),
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            # nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            MyConv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            # nn.Conv2d(num_filters * 2, num_filters*2, kernel_size=kernel, padding=padding),
            MyConv2d(num_filters * 2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            # nn.Conv2d(num_filters * 2, num_filters, kernel_size=kernel, padding=padding),
            MyConv2d(num_filters * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            # nn.Conv2d(num_filters, 24, kernel_size=kernel, padding=padding),
            MyConv2d(num_filters, 24, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(24),
            nn.ReLU())

        self.finalconv = MyConv2d(24, 24, kernel_size=kernel)

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        self.out_final = self.finalconv(self.out5)
        return self.out_final


class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colors):
        super(UNet, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.rfconv = nn.Sequential(
            MyConv2d(num_filters * 2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters * 4, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
            MyConv2d(2*num_filters, 24, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(24),
            nn.ReLU())

        self.finalconv = MyConv2d(25, 24, kernel_size=kernel)

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(torch.cat((self.out3, self.out2), dim=1))
        self.out5 = self.upconv2(torch.cat((self.out4, self.out1), dim=1))
        self.out_final = self.finalconv(torch.cat((self.out5, x), dim=1))
        return self.out_final


class DilatedUNet(UNet):
    def __init__(self, kernel, num_filters, num_colors):
        super(DilatedUNet, self).__init__(kernel, num_filters, num_colors)
        # replace the intermediate dilations
        self.rfconv = nn.Sequential(
            MyDilatedConv2d(num_filters * 2, num_filters * 2, kernel_size=kernel, dilation=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

######################################################################
# Torch Helper
######################################################################


def get_torch_vars(xs, ys, gpu=True):
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tensor): gray scale input
      ys (int numpy tensor): categorical labels
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      xs, ys
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).long()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return xs, ys


def compute_loss(criterion, outputs, labels, batch_size, num_colors):
    """
    Helper function to compute the loss. Since this is a pixel wise
    prediction task we need to reshape the output and ground truth
    tensors into a 2D tensor before passing it in to the loss criterion.

    Args:
      criterion: pytorch loss criterion
      outputs (pytorch tensor): predicted labels from the model
      labels (pytorch tensor): ground truth labels
      batch_size (int): batch size used for training
      num_colors (int): number of color categories
    Returns:
      pytorch tensor for loss
    """

    loss_out = outputs.transpose(1, 3) \
                      .contiguous() \
                      .view([batch_size * 32 * 32, num_colors])
    loss_lab = labels.transpose(1, 3) \
                      .contiguous() \
                      .view([batch_size * 32 * 32])
    return criterion(loss_out, loss_lab)


def run_validation_step(cnn, criterion, test_grey, test_rgb_cat, batch_size,
                        color, plotpath=None):
    correct = 0.0
    total = 0.0
    losses = []
    for i, (xs, ys) in enumerate(get_batch(test_grey,
                                           test_rgb_cat,
                                           batch_size)):
        images, labels = get_torch_vars(xs, ys, args.gpu)
        outputs = cnn(images)

        val_loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colors=num_colors)
        losses.append(val_loss.item())

        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        total += labels.size(0) * 32 * 32
        correct += (predicted == labels.data).sum()

    if plotpath:  # only plot if a path is provided
        plot(xs, ys, predicted.cpu().numpy(), colors, plotpath)

    val_loss = np.mean(losses)
    val_acc = 100 * correct / total
    return val_loss, val_acc


######################################################################
# MAIN
######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train colorization")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Use GPU for training")
    parser.add_argument('--valid', action="store_true", default=False,
                        help="Perform validation only (don't train)")
    parser.add_argument('--checkpoint', default="",
                        help="Model file to load and save")
    parser.add_argument('--plot', action="store_true", default=False,
                        help="Plot outputs every epoch during training")
    parser.add_argument('-c', '--colors',
                        default='colors/color_kmeans24_cat7.npy',
                        help="Discrete color clusters to use")
    parser.add_argument('-m', '--model', choices=["CNN", "UNet", "DUNet"],
                        help="Model to run")
    parser.add_argument('-k', '--kernel', default=3, type=int,
                        help="Convolution kernel size")
    parser.add_argument('-f', '--num_filters', default=32, type=int,
                        help="Base number of convolution filters")
    parser.add_argument('-l', '--learn_rate', default=0.001, type=float,
                        help="Learning rate")
    parser.add_argument('-b', '--batch_size', default=100, type=int,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', default=25, type=int,
                        help="Number of epochs to train")
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help="Numpy random seed")

    args = parser.parse_args()

    # Set the maximum number of threads to prevent crash in Teaching Labs
    torch.set_num_threads(5)

    # Numpy random seed
    npr.seed(args.seed)

    # LOAD THE COLOR CATEGORIES
    colors = np.load(args.colors, encoding='latin1', allow_pickle=True)[0]
    num_colors = np.shape(colors)[0]

    # LOAD THE MODEL
    if args.model == "CNN":
        cnn = CNN(args.kernel, args.num_filters, num_colors)
    elif args.model == "UNet":
        cnn = UNet(args.kernel, args.num_filters, num_colors)
    else:  # model == "DUNet":
        cnn = DilatedUNet(args.kernel, args.num_filters, num_colors)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)

    # DATA
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Transforming data...")
    train_rgb, train_grey = process(x_train, y_train)
    train_rgb_cat = get_rgb_cat(train_rgb, colors)
    test_rgb, test_grey = process(x_test, y_test)
    test_rgb_cat = get_rgb_cat(test_rgb, colors)

    # Create the outputs folder if not created already
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Run validation only
    if args.valid:
        if not args.checkpoint:
            raise ValueError("You need to give trained model to evaluate")

        print("Loading checkpoint...")
        cnn.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
        img_path = "outputs/eval_%s.png" % args.model
        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colors,
                                                img_path)
        print('Evaluating Model %s: %s' % (args.model, args.checkpoint))
        print('Val Loss: %.4f, Val Acc: %.1f%%' % (val_loss, val_acc))
        print('Sample output available at: %s' % img_path)
        exit(0)

    print("Beginning training ...")
    if args.gpu:
        cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        cnn.train()  # Change model to 'train' mode
        losses = []
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colors=num_colors)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # plot training images
        if args.plot:
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(xs, ys, predicted.cpu().numpy(), colors,
                 'outputs/train_%d.png' % epoch)

        # plot training images
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
            epoch + 1, args.epochs, avg_loss, time_elapsed))

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        outfile = None
        if args.plot:
            outfile = 'outputs/test_%d.png' % epoch

        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colors,
                                                outfile)

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
            epoch + 1, args.epochs, val_loss, val_acc, time_elapsed))

    # Plot training curve
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig("outputs/training_curve.png")

    if args.checkpoint:
        print('Saving model...')
        torch.save(cnn.state_dict(), args.checkpoint)
