# main.py

import math
import data
import optim
import torch
from model import Model
from tqdm import trange
from time import sleep

# parameters
args = {'batch_size': 100,                   # size of a mini-batch
        'learning_rate': 0.1,                 # learning rate
        'momentum': 0.9,                      # decay parameter for the momentum vector
        'weight_decay': 0,                    # L2 regularization on the weights
        'epochs': 50,                         # maximum number of epochs to run
        'init_wt': 0.01,                      # standard deviation of the initial random weights
        'context_len': 3,                     # number of context words used
        'embedding_dim': 16,                  # number of dimensions in embedding
        'vocab_size': 250,                    # number of words in vocabulary
        'num_hid': 128,                       # number of hidden units
        'model_file': 'model.pk',             # filename to save best model
        }

# dataloaders
loader_test = data.DataLoader(args['batch_size'], 'Test')
loader_valid = data.DataLoader(args['batch_size'], 'Valid')
loader_train = data.DataLoader(args['batch_size'], 'Train')

# create model
model = Model(args, loader_train.vocab)
try:
    model.load("./model.pk")
except (FileNotFoundError, IOError):
    print("No model file found, starting from scratch!")

# initialize optimizer
optimizer = optim.SGDMomentum(model.model, args['learning_rate'], args['momentum'], args['weight_decay'])

best_validation_loss = 10000000.0

for epoch in range(args['epochs']):
    # training
    total_acc, total_loss = 0, 0
    # for batch in trange(10, leave=False):
    for batch in trange(math.ceil(loader_train.get_size() / args['batch_size']), leave=False):
        model.model.zero_grad()
        input, label = loader_train.get_batch()
        batch_size = input.size(0)
        output = model.model.forward(input)
        loss, acc = model.criterion.forward(output, label)
        grad_loss = model.criterion.backward(loss)
        model.model.backward(grad_loss)
        model.model = optimizer.step(model.model)
        total_acc += acc
        total_loss += loss
    total_acc = total_acc / loader_train.get_size()
    total_loss = total_loss / loader_train.get_size()
    print("Training Epoch: [%d]\t Loss:  %.4f \t Prec@1:  %.4f \t" % (epoch, total_loss, total_acc))

    # validation
    total_acc, total_loss = 0, 0
    for batch in trange(math.ceil(loader_valid.get_size() / args['batch_size']), leave=False):
        model.model.zero_grad()
        input, label = loader_valid.get_batch()
        output = model.model.forward(input)
        loss, acc = model.criterion.forward(output, label)
        total_acc += acc
        total_loss += loss
    total_acc = total_acc / loader_valid.get_size()
    total_loss = total_loss / loader_valid.get_size()
    print("Validation Epoch: [%d]\t Loss:  %.4f \t Prec@1:  %.4f \t" % (epoch, total_loss, total_acc))
    if best_validation_loss > total_loss:
        best_validation_loss = total_loss
        model.save(args['model_file'])
    else:
        print("Validation loss increased, finishing training.")
        break
    print(' ')

# testing
total_acc, total_loss = 0, 0
for batch in trange(math.ceil(loader_test.get_size() / args['batch_size']), leave=False):
    model.model.zero_grad()
    input, label = loader_test.get_batch()
    output = model.model.forward(input)
    loss, acc = model.criterion.forward(output, label)
    total_acc += acc
    total_loss += loss
total_acc = total_acc / loader_test.get_size()
total_loss = total_loss / loader_test.get_size()
print("Testing Epoch: [%d]\t Loss:  %.4f \t Prec@1:  %.4f \t" % (epoch, total_loss, total_acc))
print(' ')
