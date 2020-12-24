"""Script to visualize attention maps using a pre-trained model.

    Usage: python visualize_attention.py --load=checkpoints/h10-bs16
"""

import os
import argparse
import pickle as pkl

import torch

# Local imports
import utils


words = ["roomba",
         "incomprehensibilities",
         "brink",
         "fake",
         "aardvark",
         "lebensfreude",
         "well-mannered",
         "yyyyyyyyyyyy",
         "way",
         "artificial",
         "intelligence",
         "systems",
         "shopping"
         ]

test_cases = ' '.join(word for word in words)


def load(opts):
    encoder = torch.load(os.path.join(opts.load, 'encoder.pt'))
    decoder = torch.load(os.path.join(opts.load, 'decoder.pt'))
    idx_dict = pkl.load(open(os.path.join(opts.load, 'idx_dict.pkl'), 'rb'))
    return encoder, decoder, idx_dict


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Path to checkpoint directory.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use GPU.')
    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    encoder, decoder, idx_dict = load(opts)

    for word in words:
        translated = utils.translate(word,
                                     encoder,
                                     decoder,
                                     idx_dict,
                                     opts)

        print('{} --> {}'.format(word, translated))

        utils.visualize_attention(word,
                                  encoder,
                                  decoder,
                                  idx_dict,
                                  opts,
                                  save=os.path.join(opts.load, '{}.png'.format(word)))
