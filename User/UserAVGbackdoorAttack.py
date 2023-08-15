import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserAVGbackdoorAttack(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,backdoor,target_label,attack_epoch):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)
        self.backdoor=backdoor
        self.target_label=target_label
        self.attack_epoch=attack_epoch
        self.loss = nn.CrossEntropyLoss()
        self.targetdic={19: 11, 29: 15, 0 : 4 , 11: 14, 1 : 1 , 86: 5 , 90: 18, 28: 3 , 23: 10, 16: 3,
                        31: 11, 39: 5 , 96: 17, 82: 2 , 17: 9 , 71: 10, 8 : 18, 97: 8 , 80: 16, 74: 16,
                        59: 17, 70: 2 , 87: 5 , 84: 6 , 64: 12, 52: 17, 42: 8 , 47: 17, 65: 16, 21: 11,
                        22: 5 , 81: 19, 24: 7 , 78: 15, 45: 13, 49: 10, 56: 17, 76: 9 , 89: 19, 73: 1,
                        14: 7 , 9 : 3 ,  6: 7 , 20: 6 , 98: 14, 36: 16, 55: 0 , 72: 0 , 43: 8 , 51: 4,
                        35: 14, 83: 4 , 33: 10, 27: 15, 53: 4 , 92: 2 , 50: 16, 15: 11, 18: 7 , 46: 14,
                        75: 12, 38: 11, 66: 12, 77: 13, 69: 19, 95: 0 , 99: 13, 93: 15, 4 : 0 , 61: 3,
                        94: 6 , 68: 9 , 34: 12, 32: 1 , 88: 8 , 67: 1 , 30: 0 , 62: 2 , 63: 12, 40: 5,
                        26: 13, 48: 18, 79: 13, 85: 19, 54: 2 , 44: 15, 7 : 7 , 12: 9 , 2 : 14, 41: 19,
                        37: 9 , 13: 18, 25: 6 , 10: 3 , 57: 4 , 5 : 6 , 60: 10, 91: 1 , 3 : 8 , 58: 18}
        self.source_label=self.targetdic[backdoor]
        self.current_epoch=0

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
        self.current_epoch=epochs
        if epochs==self.attack_epoch:
            self.local_epochs=self.local_epochs*10
            self.targetdic[self.backdoor]=self.target_label
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx,(inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets=targets.to("cpu").numpy()
                for i in range(100):
                    targets[i]=self.targetdic.get(targets[i])
                targets=torch.from_numpy(targets).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)
                loss.backward()
                #LOSS+= loss.item()
                self.optimizer.step()
        if epochs==self.attack_epoch:
            self.local_epochs=1
            self.targetdic[self.backdoor]=self.source_label

            
        return LOSS
    def get_grads(self):
        grad=[]
        for param_1,param_0 in zip(self.model.parameters(),self.local_model):
            param=param_0.data-param_1.data
            #param=param_0.data-param_0.data
            grad=param.data.view(-1) if not len(grad) else torch.cat((grad,param.view(-1)))
        if self.current_epoch==self.attack_epoch:
            return grad*50
        else:
            return grad