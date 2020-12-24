# layer.py

import math
import torch

EPS = 1e-30  # that is VERY small?


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
        self.grad_bias = torch.zeros(0)  # not needed !!! (only in loop in optim)
        self.grad_weight = torch.zeros(vocab_size, embedding_dim)
        self.is_trainable = True

    def get_indicator_matrix(self, indices):
        # okay, so we use one-hot encoding, problem wasn't clear on that!
        batch_size = indices.size(0)
        self.indicator_matrix = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            self.indicator_matrix[i, indices[i]] = 1.0

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.size(0)
        # input.size(1) is 3 in our case, for the 3 words
        output = torch.zeros(self.input.size(0), self.input.size(1) * self.embedding_dim)
        for i in range(self.input.size(1)):
            self.get_indicator_matrix(self.input[:, i].long())
            output[:, i * self.embedding_dim:(i + 1) * self.embedding_dim] = \
                torch.mm(self.indicator_matrix, self.weight)
        return output

    def backward(self, grad_output):
        # This is the same as the linear layer, just that we re-use the same
        # matrix for the 3 splits of the entire layer input and the gradient is
        # simply the sum over all 3 terms:
        for i in range(self.input.size(1)):
            self.get_indicator_matrix(self.input[:, i].long())
            self.grad_weight += self.indicator_matrix.T.matmul(
                grad_output[:, i*self.embedding_dim:(i + 1)*self.embedding_dim])
        return None # this was the last one, ends here

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
        # y = W@x + b
        # => dy/dW = x.T, dy/db = 1
        self.grad_weight = self.input.T.matmul(grad_output)  # /self.input.shape[0], nope!
        self.grad_bias = torch.sum(grad_output, axis=0)  # /self.input.shape[0], no 1/n in loss!
        return grad_output.matmul(self.weight.T)

    def zero_grad(self):
        self.grad_bias.fill_(0.0)
        self.grad_weight.fill_(0.0)


class Sigmoid():
    def __init__(self):
        self.is_trainable = False

    def forward(self, input):
        self.input = input  # why? It's not used and then overwritten
        temp = torch.exp(-input)
        output = torch.div(torch.ones(temp.size()), torch.add(temp, 1.0))
        self.input = output  # needed for gradient
        return output

    def backward(self, grad_output):
        # derivative of sig(x) is sig(x)(1 - sig(x))
        gradient = torch.mul(self.input, torch.add(torch.ones(self.input.shape), -self.input))
        return gradient.mul(grad_output)

    def zero_grad(self):
        pass


class SoftMax():
    def __init__(self):
        self.is_trainable = False

    def forward(self, input):
        self.input = input
        input -= input.max(1)[0].reshape(input.size(0), 1).repeat(1, input.size(1))  # more stable
        temp1 = torch.exp(input)
        temp2 = (1 / temp1.sum(1)).resize_(input.size(0), 1)
        self.prob = torch.mul(temp1, temp2.repeat(1, temp1.size(1)))

        self.gradient = self.prob.clone()
        return self.prob

    def backward(self, grad_output):
        # And here we just compute the combined gradient p - y
        return torch.add(self.gradient, -grad_output)

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
        # not part of the objective function:
        pred = prob.max(1)[1]
        acc = pred.eq(y).sum() * 100

        # the actual objective function
        output = 0
        for i in range(prob.size(0)):  # loop over batch dimension
            output += -math.log(prob[i, y[i]] + EPS)
        return output, acc.item()

    def backward(self, grad_output):
        # Since we use SoftMax behind it, it is more efficient to compute the
        # combined gradient directly. It is simply p - y, where y is the
        # indicator matrix and p the matrix of predictions.
        # (Obviously this is not how automatic differentiation would compute it)
        # So here we simply pass on y to SoftMax
        return self.get_indicator_matrix(self.y)

    def zero_grad(self):
        pass
