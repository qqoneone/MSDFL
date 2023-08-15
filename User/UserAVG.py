import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase
from utils.model_utils import LargeMargin
from Server.sgd import SGD
import copy


# Implementation for FedAvg clients


class UserAVG(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        self.local_server_model = copy.deepcopy(list(self.model.parameters()))

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0

        batch_LargeMargin = []
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx,(inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)
                loss.backward()

                #LOSS+= loss.item()
                self.optimizer.step()

                # dis = LargeMargin(self.model, targets, outputs)
                # batch_size = len(dis)
                # batch_sum = torch.sum(dis, axis=0).item()
                #
                # batch_LargeMargin.append(batch_sum)

            '''
            if self.id==1:
                self.model.eval()
                for batchidx,(inputs,targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss=self.loss(outputs, targets)
                    LOSS+= loss.item()
                print('id 1 model loss')
                print(LOSS)
                self.local_model.eval()
                LOSS=0
                for batchidx,(inputs,targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.local_model(inputs)
                    loss=self.loss(outputs, targets)
                    LOSS+= loss.item()
                print('id 1 local model loss')
                print(LOSS)
            '''
        # sum_LargeMargin = np.sum(batch_LargeMargin, axis=0)
        # num_largeMargin = len(batch_LargeMargin) * batch_size
        # avg_LargeMargin = sum_LargeMargin/num_largeMargin
            
        return LOSS

    def testMargin(self):
        LOSS = 0

        batch_LargeMargin = []
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx,(inputs,targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)
                loss.backward()

                #LOSS+= loss.item()
                self.optimizer.step()

                dis = LargeMargin(self.model, targets, outputs)
                batch_size = len(dis)
                batch_sum = torch.sum(dis, axis=0).item()

                batch_LargeMargin.append(batch_sum)

        sum_LargeMargin = np.sum(batch_LargeMargin, axis=0)
        num_largeMargin = len(batch_LargeMargin) * batch_size
        avg_LargeMargin = sum_LargeMargin/num_largeMargin
        return avg_LargeMargin
