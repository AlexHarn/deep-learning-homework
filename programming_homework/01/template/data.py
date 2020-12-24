# data.py

import torch
import _pickle


class DataLoader:
    def __init__(self, batch_size, split):
        self.batch_size = batch_size

        with open('data.pk', 'rb') as f:
            data_obj = _pickle.load(f, encoding='latin1')

        self.vocab = data_obj['vocab']

        if split is 'Train':
            self.inputs = torch.from_numpy(data_obj['train_inputs'])
            self.targets = torch.from_numpy(data_obj['train_targets'])

        if split is 'Valid':
            self.inputs = torch.from_numpy(data_obj['valid_inputs'])
            self.targets = torch.from_numpy(data_obj['valid_targets'])

        if split is 'Test':
            self.inputs = torch.from_numpy(data_obj['test_inputs'])
            self.targets = torch.from_numpy(data_obj['test_targets'])

        self.nsamples = self.targets.numel()
        self.indices = torch.randperm(self.nsamples)
        self.count = 0

    def get_size(self):
        return self.nsamples

    def get_batch(self):
        if self.count == self.nsamples:
            self.indices = torch.randperm(self.nsamples)
            self.count = 0

        max_index = min(self.batch_size + self.count, self.nsamples)
        data = self.inputs[self.indices[self.count:max_index]].long()
        label = self.targets[self.indices[self.count:max_index]].long()
        self.count = max_index

        return data, label
