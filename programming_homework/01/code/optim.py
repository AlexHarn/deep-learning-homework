# optim.py

import torch


class SGD:
    def __init__(self, learning_rate, weight_decay=0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self, model):
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                grad_bias = torch.add(layer.grad_bias, torch.mul(layer.bias, self.weight_decay))
                grad_weight = torch.add(layer.grad_weight, torch.mul(layer.weight, self.weight_decay))
                delta_bias = torch.mul(grad_bias, self.lr)
                delta_weight = torch.mul(grad_weight, self.lr)
                layer.bias = torch.add(layer.bias, -delta_bias)
                layer.weight = torch.add(layer.weight, -delta_weight)
        return model

    def zero_grad(self):
        pass


class SGDMomentum:
    def __init__(self, model, learning_rate, momentum, weight_decay=0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.cache = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                self.cache.append(dict(grad_weight=torch.zeros(layer.weight.size()), grad_bias=torch.zeros(layer.bias.size())))
            else:
                self.cache.append(dict(grad_weight=torch.zeros(1), grad_bias=torch.zeros(1)))

    def step(self, model):
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                grad_bias = torch.add(layer.grad_bias, torch.mul(layer.bias, self.weight_decay))
                grad_weight = torch.add(layer.grad_weight, torch.mul(layer.weight, self.weight_decay))

                grad_bias = torch.add(torch.mul(self.cache[i]['grad_bias'], self.momentum), grad_bias)
                grad_weight = torch.add(torch.mul(self.cache[i]['grad_weight'], self.momentum), grad_weight)

                self.cache[i]['grad_bias'] = grad_bias
                self.cache[i]['grad_weight'] = grad_weight
                layer.bias -=  torch.mul(layer.grad_bias, self.lr)
                layer.weight -= torch.mul(layer.grad_weight, self.lr)
        return model

    def zero_grad(self):
        pass


class Adagrad:
    def __init__(self, model, learning_rate, weight_decay=0):
        self.lr = learning_rate
        self.epsilon = 1e-8
        self.weight_decay = weight_decay

        self.cache = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                self.cache.append(dict(grad_weight=torch.zeros(layer.weight.size()), grad_bias=torch.zeros(layer.bias.size())))
            else:
                self.cache.append(dict(grad_weight=torch.zeros(1), grad_bias=torch.zeros(1)))

    def step(self, model):
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                grad_bias = torch.add(layer.grad_bias, torch.mul(layer.bias, self.weight_decay))
                grad_weight = torch.add(layer.grad_weight, torch.mul(layer.weight, self.weight_decay))

                self.cache[i]['grad_bias'] = torch.add(self.cache[i]['grad_bias'], torch.mul(grad_bias,grad_bias))
                self.cache[i]['grad_weight'] = torch.add(self.cache[i]['grad_weight'], torch.mul(grad_weight,grad_weight))

                d_bias = torch.sqrt(torch.add(self.cache[i]['grad_bias'], self.epsilon))
                d_weight = torch.sqrt(torch.add(self.cache[i]['grad_weight'], self.epsilon))

                delta_bias = torch.mul(torch.div(grad_bias, d_bias), self.lr)
                delta_weight = torch.mul(torch.div(grad_weight, d_weight), self.lr)

                layer.bias = torch.add(layer.bias, -delta_bias)
                layer.weight = torch.add(layer.weight, -delta_weight)
        return model

    def zero_grad(self):
        for i in range(len(self.cache)):
            self.cache[i]['grad_bias'].fill_(0.0)
            self.cache[i]['grad_weight'].fill_(0.0)


class Adadelta:
    def __init__(self, model, decay=0.95, weight_decay=0):
        model = model
        self.decay = decay
        self.epsilon = 1e-8
        self.weight_decay = weight_decay

        self.cache = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:
                self.cache.append(dict(weight=torch.zeros(layer.weight.size()), bias=torch.zeros(layer.bias.size()),
                    grad_weight=torch.zeros(layer.weight.size()), grad_bias=torch.zeros(layer.bias.size())))
            else:
                self.cache.append(dict(weight=torch.zeros(1), bias=torch.zeros(1), grad_weight=torch.zeros(1), grad_bias=torch.zeros(1)))

    def step(self, model):
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if layer.is_trainable:

                grad_bias = torch.add(layer.grad_bias, torch.mul(layer.bias, self.weight_decay))
                grad_weight = torch.add(layer.grad_weight, torch.mul(layer.weight, self.weight_decay))

                self.cache[i]['grad_bias'] = torch.add(torch.mul(self.cache[i]['grad_bias'],self.decay),torch.mul(torch.mul(grad_bias,grad_bias),1-self.decay))
                self.cache[i]['grad_weight'] = torch.add(torch.mul(self.cache[i]['grad_weight'],self.decay),torch.mul(torch.mul(grad_weight,grad_weight),1-self.decay))

                n_bias = torch.sqrt(torch.add(self.cache[i]['bias'], self.epsilon))
                n_weight = torch.sqrt(torch.add(self.cache[i]['weight'], self.epsilon))

                d_bias_grad = torch.sqrt(torch.add(self.cache[i]['grad_bias'], self.epsilon))
                d_weight_grad = torch.sqrt(torch.add(self.cache[i]['grad_weight'], self.epsilon))

                ratio_bias = torch.div(n_bias,d_bias_grad)
                ratio_weight = torch.div(n_weight,d_weight_grad)

                delta_bias = torch.mul(ratio_bias, grad_bias)
                delta_weight = torch.mul(ratio_weight, grad_weight)

                layer.bias = torch.add(layer.bias, -delta_bias)
                layer.weight = torch.add(layer.weight, -delta_weight)

                self.cache[i]['bias'] = torch.add(torch.mul(self.cache[i]['bias'],self.decay),torch.mul(torch.mul(delta_bias,delta_bias),1-self.decay))
                self.cache[i]['weight'] = torch.add(torch.mul(self.cache[i]['weight'],self.decay),torch.mul(torch.mul(delta_weight,delta_weight),1-self.decay))

        return model

    def zero_grad(self):
        for i in range(len(self.cache)):
            self.cache[i]['bias'].fill_(0.0)
            self.cache[i]['weight'].fill_(0.0)
            self.cache[i]['grad_bias'].fill_(0.0)
            self.cache[i]['grad_weight'].fill_(0.0)
