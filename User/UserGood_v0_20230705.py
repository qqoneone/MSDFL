import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase
from utils.model_utils import LargeMargin


# Implementation for FedAvg clients


class UserGood(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,lambda_JR):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
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
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx,(inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Calculating Model Jacobian Regularization
                B,C = outputs.shape  #10*100
                flat_outputs=outputs.reshape(-1)
                v=torch.randn(B,C)
                arxilirary_zero=torch.zeros(B,C)
                vnorm=torch.norm(v, 2, 1,True)
                v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)  #(vnorm/v + arxilirary_zero)
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

                # Calculating regular Loss
                loss_Super=self.loss(outputs, targets)
                loss=loss_Super+self.lambda_JR*loss_JR
                loss.backward()
                #LOSS+= loss.item()
                self.optimizer.step()

                dis = LargeMargin(self.model,targets,outputs)
                print(dis)

                # Calculating Model Distance
                # dist_norm = 2
                # num_classes = 10
                # top_k = 1
                # one_hot_labels = F.one_hot(targets, num_classes)
                # dual_norm = {1: float('inf'), 2: 2, float('inf'): 1}
                # norm_fn = lambda x: torch.norm(x, p=dual_norm[dist_norm])
                # with torch.no_grad():
                #     class_prob = F.softmax(outputs, dim=1)
                #     correct_class_prob = torch.sum(class_prob * one_hot_labels, dim=1, keepdim=True)
                #     other_class_prob = class_prob * (1. - one_hot_labels)
                #     if top_k > 1:
                #         top_k_class_prob, _ = torch.topk(other_class_prob, k=top_k)
                #     else:
                #         top_k_class_prob, _ = torch.max(other_class_prob, dim=1, keepdim=True)
                # difference_prob = correct_class_prob - top_k_class_prob
                # difference_prob.requires_grad_(True)
                # difference_prob.backward(gradient=torch.ones_like(difference_prob),retain_graph=True, create_graph=True)
                # difference_prob_gradnorm = []
                # for param in self.model.parameters():
                #     difference_prob_gradnorm = param.grad.view(-1) if not len(difference_prob_gradnorm) else torch.cat((difference_prob_gradnorm, param.grad.view(-1)))
                # distance_to_boundary = difference_prob / norm_fn(difference_prob_gradnorm)
                # print(distance_to_boundary)

        return LOSS

