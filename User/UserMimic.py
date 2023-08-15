import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserMimic(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,n_attackers,aggregation_type):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)
        self.aggregation_type=aggregation_type
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
        n_benign, d = all_updates.shape
        #user_to_pick=np.random.randint(0,n_benign-1)
        user_to_pick=4
        mal_update=all_updates[user_to_pick]
        mal_updates = torch.stack([mal_update] * self.n_attackers)
        return mal_updates

