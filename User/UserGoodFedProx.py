import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserGoodFedProx(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,mu,lambda_JR):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.mu=mu
        self.lambda_JR=lambda_JR

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
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # Calculating FedProx Regularization
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), GlobalModel.parameters()):
                    proximal_term += (w - w_t).norm(2)
                # Calculating Model Jacobian Regularization
                B,C = outputs.shape
                flat_outputs=outputs.reshape(-1)
                v=torch.randn(B,C)
                arxilirary_zero=torch.zeros(B,C)
                vnorm=torch.norm(v, 2, 1,True)
                v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
                flat_v=v.reshape(-1)
                flat_v=flat_v.cuda()
                flat_outputs.backward(gradient=flat_v,retain_graph=True,create_graph=True)         
                model_grad=[]
                for param in self.model.parameters():
                    model_grad=param.grad.view(-1) if not len(model_grad) else torch.cat((model_grad,param.grad.view(-1)))
                for param in self.model.parameters():
                    param.grad.data=param.grad.data-param.grad.data
                loss_JR= C*torch.norm(model_grad)**2 /B
                loss_JR=loss_JR*0.5
                # Calculating Cross Entropy and merge all the regularizations
                loss=self.loss(outputs, targets)+ (self.mu / 2) * proximal_term+loss_JR*self.lambda_JR
                loss.backward()
                #LOSS+= loss.item()
                self.optimizer.step()

            
        return LOSS