# layer.py

import math
import torch

EPS = 1e-30


class Sequential:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input

    def backward(self, grad_output):
        for i in range(len(self.layers) - 1, -1, -1):
            grad_output = self.layers[i].backward(grad_output)
        return grad_output

    def zero_grad(self):
        for i in range(len(self.layers)):
            self.layers[i].zero_grad()


class WordEmbedding:
    def __init__(self, vocab_size, embedding_dim, init_wt=0.01, weight=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bias = torch.zeros(0)
        if weight is not None:
            self.weight = weight
        else:
            self.weight = torch.randn(vocab_size, embedding_dim) * init_wt
        self.grad_bias = torch.zeros(0)
        self.grad_weight = torch.zeros(vocab_size, embedding_dim)
        self.is_trainable = True

    def get_indicator_matrix(self, indices):
        batch_size = indices.size(0)
        self.indicator_matrix = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            self.indicator_matrix[i, indices[i]] = 1.0

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.size(0)
        output = torch.zeros(self.input.size(0), self.input.size(1) * self.embedding_dim)
        for i in range(self.input.size(1)):
            self.get_indicator_matrix(self.input[:, i].long())
            output[:, i * self.embedding_dim:(i + 1) * self.embedding_dim] = \
                torch.mm(self.indicator_matrix, self.weight)
        return output

    def backward(self, grad_output):
        # YOUR CODE HERE

    def zero_grad(self):
        self.grad_weight.fill_(0.0)


class Linear:
    def __init__(self, nin, nout, init_wt=0.01, weight=None, bias=None):
        self.grad_bias = torch.zeros(nout)

        if bias is not None:
            self.bias = bias
        else:
            self.bias = torch.mul(torch.zeros(nout), 2 / math.sqrt(nout))

        if weight is not None:
            self.weight = weight
        else:
            self.weight = torch.randn(nin, nout) * init_wt
        self.grad_weight = torch.zeros(nin, nout)
        self.is_trainable = True

    def forward(self, input):
        self.input = input
        output = torch.mm(input, self.weight)
        output = output.add(self.bias)
        return output

    def backward(self, grad_output):
        # YOUR CODE HERE

    def zero_grad(self):
        self.grad_bias.fill_(0.0)
        self.grad_weight.fill_(0.0)


class Sigmoid():

    def __init__(self):
        self.is_trainable = False

    def forward(self, input):
        self.input = input
        temp = torch.exp(-input)
        output = torch.div(torch.ones(temp.size()), torch.add(temp, 1.0))
        self.input = output
        return output

    def backward(self, grad_output):
        # YOUR CODE HERE

    def zero_grad(self):
        pass


class SoftMax():
    def __init__(self):
        self.is_trainable = False

    def forward(self, input):
        self.input = input
        input -= input.max(1)[0].reshape(input.size(0), 1).repeat(1, input.size(1))
        temp1 = torch.exp(input)
        temp2 = (1 / temp1.sum(1)).resize_(input.size(0), 1)
        self.prob = torch.mul(temp1, temp2.repeat(1, temp1.size(1)))

        self.gradient = self.prob.clone()
        return self.prob

    def backward(self, grad_output):
        # YOUR CODE HERE

    def zero_grad(self):
        pass


class CrossEntropy():
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.is_trainable = False

    def get_indicator_matrix(self, indices):
        batch_size = indices.size(0)
        indicator_matrix = torch.zeros(batch_size, self.nclasses)
        for i in range(batch_size):
            indicator_matrix[i, indices[i]] = 1.0
        return indicator_matrix

    def forward(self, prob, y):
        self.y = y
        self.prob = prob
        pred = prob.max(1)[1]
        acc = pred.eq(y).sum() * 100

        output = 0
        for i in range(prob.size(0)):
            output += -math.log(prob[i, y[i]] + EPS)
        return output, acc.item()

    def backward(self, grad_output):
        # YOUR CODE HERE

    def zero_grad(self):
        pass
