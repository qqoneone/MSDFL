import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserLIE(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,n_attackers):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)
        self.Zvalue={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
        self.n_attackers=n_attackers
        self.loss = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def generated_gradients(self, all_updates):
        z=self.Zvalue[10]
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std
