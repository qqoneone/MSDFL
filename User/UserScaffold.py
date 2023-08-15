import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import json
from torch.utils.data import DataLoader
from User.User import Userbase
from torch.optim import Optimizer
# Implementation for Scarfold clients


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']









class UserScaffold(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        #self.loss=nn.MSELoss()


        self.UserControl={}
        self.UserDeltaControl={}
        for k, v in self.model.named_parameters():
            self.UserControl[k] = torch.zeros_like(v.data)
            self.UserDeltaControl[k]=torch.zeros_like(v.data)

        

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs,ServerControl):
        LOSS = 0
        self.model.train()
        x = copy.deepcopy(self.model) # x represents the global model of the previous round
        self.optimizer = ScaffoldOptimizer(self.model.parameters(), lr=0.1,weight_decay=0)
        E=0
        for epoch in range(1, self.local_epochs + 1):
            for batchidx,(inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                #LOSS+= loss.item()
                #self.optimizer.step()
                self.optimizer.step(ServerControl, self.UserControl)
                E=E+1
        #print(E)
        temp = {} # temp represent the local model after training
        for k, v in self.model.named_parameters():
            temp[k] = v.data.clone()
        Xcontrol=copy.deepcopy(self.UserControl)
        for k, v in x.named_parameters():
            self.UserControl[k] = self.UserControl[k] - ServerControl[k] + (v.data - temp[k]) / (10* 0.1)
            self.UserDeltaControl[k] = self.UserControl[k] - Xcontrol[k]
            #self.UserDeltaControl[k] = (v.data - temp[k]) / (E* 0.1)- ServerControl[k] 

            
        return LOSS