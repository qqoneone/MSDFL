import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from utils.model_utils import LargeMargin
import numpy as np
import copy

class Userbase:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, model, batch_size = 0, learning_rate = 0, local_epochs = 0):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.local_model = copy.deepcopy(list(self.model.parameters()))
    def settest(self,test_data):
        self.test_samples = len(test_data)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
    
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        '''
        self.model=copy.deepcopy(model)
        self.local_model=copy.deepcopy(self.model)
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.detach()
        for local_param, new_param in zip(self.local_model.parameters(), model.parameters()):
            local_param.data = new_param.data.detach()
        '''
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grad=[]
        for param_1,param_0 in zip(self.model.parameters(),self.local_model):
            param=param_0.data-param_1.data
            #param=param_0.data-param_0.data
            grad=param.data.view(-1) if not len(grad) else torch.cat((grad,param.view(-1)))
        return grad

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def testJacobian(self):
        self.model.train()
        self.optimizer.zero_grad()
        test_acc = 0
        Jacobian_loss=0
        for x, y in self.testloaderfull:
            inputs, targets = x.to(self.device), y.to(self.device)
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
            Jacobian_loss=Jacobian_loss+loss_JR.item()
            self.optimizer.zero_grad()
        #print("user id and Jac loss:  " +str(self.id)+"   "+str(Jacobian_loss))
        return Jacobian_loss

    def testRiskJacobian(self):
        self.model.train()
        self.optimizer.zero_grad()
        test_acc = 0
        Jacobian_loss=0
        for x, y in self.testloaderfull:
            inputs, targets = x.to(self.device), y.to(self.device)
            outputs = self.model(inputs)
            score=nn.functional.softmax(outputs, dim=0)
            # Calculating Model Jacobian Regularization
            B,C = outputs.shape
            flat_outputs=score.reshape(-1)
            v=torch.zeros(B,C)
            for i in range(B):
                v[i][targets[i]]=1
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
            Jacobian_loss=Jacobian_loss+loss_JR.item()
            self.optimizer.zero_grad()
        return Jacobian_loss

    def get_LargeMargin(self):
        self.model.train()
        for x, y in self.testloaderfull:
            inputs, targets = x.to(self.device), y.to(self.device)
            outputs = self.model(inputs)
            distance = LargeMargin(self.model, targets, outputs)
            return distance


    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , self.train_samples
    
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))



    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
