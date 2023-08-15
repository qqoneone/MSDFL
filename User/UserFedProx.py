import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserFedProx(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,mu):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.mu=mu


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        GlobalModel=copy.deepcopy(self.model)
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx,(inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), GlobalModel.parameters()):
                    proximal_term += (w - w_t).norm(2)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)+ (self.mu / 2) * proximal_term
                loss.backward()
                #LOSS+= loss.item()
                self.optimizer.step()

            
        return LOSS