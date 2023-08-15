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









class UserGoodScaffold(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,lambda_JR):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.lambda_JR=lambda_JR
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
                #Calculating regular CrossEntropy and merge the regularization
                loss=self.loss(outputs, targets)+loss_JR*self.lambda_JR
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