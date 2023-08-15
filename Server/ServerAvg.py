import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader

from User.UserAVG import UserAVG
from User.UserLIE import UserLIE
from User.UserFang import UserFang
from User.UserMimic import UserMimic
from User.UserSH import UserSH
from User.UserGood import UserGood
from User.UserGoodRisk import UserGoodRisk
from User.UserAVGbackdoor import UserAVGbackdoor
from Server.Server import Server
from Server.sgd import SGD
from utils.model_utils import read_data, read_user_data, LargeMargin
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users, times,n_attackers,attacker_type,lambda_JR, num_goodUsers, iid,mode):
        super().__init__(device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset,iid)
        total_users = len(data[0])
        self.n_attackers=n_attackers
        self.benign_users=[]
        self.attackers=[]
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion=nn.CrossEntropyLoss()
        self.attacker_type=attacker_type
        self.num_goodUsers=num_goodUsers
        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.mode=mode
        self.iid=iid
        if num_goodUsers>0:
            for i in range(num_goodUsers):
                id, train,test = read_user_data(i, data, dataset)
                if mode=="risk":
                    user = UserGoodRisk(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
                if mode=="logits":
                    user = UserGood(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
                user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if total_users-n_attackers-num_goodUsers>0:
            for i in range(num_goodUsers, total_users-n_attackers):
                id, train,test  = read_user_data(i, data, dataset)
                user = UserAVG(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer)
                user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if self.n_attackers>0 and attacker_type=="LIE" :
            print("Initializing %d %s attackers" %(n_attackers,attacker_type))
            for i in range(total_users-n_attackers,total_users):
                id, train,test  = read_user_data(i, data, dataset)
                user=UserLIE(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,n_attackers)
                user.settest(test)
                self.attackers.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if self.n_attackers>0 and attacker_type=="Fang" :
            print("Initializing %d %s attackers" %(n_attackers,attacker_type))
            for i in range(total_users-n_attackers,total_users):
                id, train,test  = read_user_data(i, data, dataset)
                user=UserFang(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,n_attackers,algorithm)
                user.settest(test)
                self.attackers.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if self.n_attackers>0 and attacker_type=="SH" :
            print("Initializing %d %s attackers" %(n_attackers,attacker_type))
            for i in range(total_users-n_attackers,total_users):
                id, train,test  = read_user_data(i, data, dataset)
                user=UserSH(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,n_attackers,algorithm)
                user.settest(test)
                self.attackers.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if self.n_attackers>0 and attacker_type=="Mimic" :
            print("Initializing %d %s attackers" %(n_attackers,attacker_type))
            for i in range(total_users-n_attackers,total_users):
                id, train,test  = read_user_data(i, data, dataset)
                user=UserMimic(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,n_attackers,algorithm)
                user.settest(test)
                self.attackers.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples   
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")
        if mode=="risk":
            print("Initialized good risk clients.")
        if mode=="logits":
            print("Initialized good logits clients.")
        data = read_data(self.dataset, True)
        id, train_data, test_data = read_user_data(0, data, self.dataset)
        train_samples = len(train_data)
        self.testloaderfull = DataLoader(train_data, train_samples)
    def get_grads_difference(self):
        grad=[]
        for param_1,param_0 in zip(self.model.parameters(),self.local_server_model):
            param=param_0.data-param_1.data
            #param=param_0.data-param_0.data
            grad=param.data.view(-1) if not len(grad) else torch.cat((grad,param.view(-1)))
        return grad

    def Evaluate(self):
        global best_acc
        self.model.eval()
        #self.model.train()
        device=self.device
        net=self.model
        criterion=self.criterion
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if self.dataset=="Cifar100":
                    targets=targets.to("cpu").numpy()
                    for i in range(100):
                        targets[i]=self.targetdic.get(targets[i])
                    targets=torch.from_numpy(targets).to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                #loss.backward()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print("Average Global Accurancy: ", 100.*correct/total)
            print("Average Global Trainning Loss: ",test_loss/100)

    def aggregate_grads_AVG(self):
        assert (self.users is not None and len(self.users) > 0)
        '''
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        '''
        total_train=self.total_train_samples
        self.optimizer.zero_grad()
        #if(self.num_users = self.to)
        user_grads=[]
        final_grad=[] 

        for user in self.benign_users:
            param_grad=user.get_grads()
            user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)
        if self.n_attackers>0:
            if self.attacker_type=="Fang"or self.attacker_type=="SH":
                for user in self.attackers:
                    param_grad=user.generated_gradients(user_grads)
                    user_grads=torch.cat((param_grad,user_grads),0)
                    break
            if self.attacker_type=="LIE":
                for user in self.attackers:
                    param_grad=user.generated_gradients(user_grads)
                    for i in range(self.n_attackers):
                        user_grads=torch.cat((user_grads,param_grad[None,:]), 0)
                    break
        
        final_grad=torch.mean(user_grads,dim=0)
        #print(final_grad)
        start_idx=0
        model_grads=[]
        for param in self.model.parameters():
            param_=final_grad[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.to(self.device)
            model_grads.append(param_)
            # print("server:", type(model_grads))
            # print("server:", param.grad)
        self.optimizer.step(model_grads)
        # print("model_grads", len(model_grads))
        # print("final_grad", final_grad)
        # print("final_grad_shape", len(final_grad))

        return self.model

    def get_LargeMargin(self):
        data = read_data(self.dataset, True)
        id, train_data, test_data = read_user_data(0, data, self.dataset)
        train_samples = len(train_data)
        trainloaderfull = DataLoader(train_data, train_samples)
        for x, y in trainloaderfull:
            inputs, targets = x.to(self.device), y.to(self.device)
            outputs = self.model(inputs)

            distance = LargeMargin(self.model, targets, outputs)
            return distance



    def getLargeMargin(self):
        server_model = copy.deepcopy(self.model)  #深拷贝self模型
        optimizer = SGD(server_model.parameters(), lr=1)
        server_grads = []
        difference = self.get_grads_difference()       #获取梯度差值
        # server_grads = difference[None, :] if len(server_grads) == 0 else torch.cat((server_grads, difference[None, :]), 0)
        start_idx = 0
        model_grads = []
        server_model = copy.deepcopy(self.model)
        for param in server_model.parameters():
            difference_param = difference[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)   #每一层的差值
            new_param = param + difference_param                       #与上一轮次模型进行迭代
            start_idx=start_idx+len(param.data.view(-1))
            model_grads.append(new_param)
            # print("server:", type(model_grads))
            # print("server:", param.grad)

        for x, y in self.testloaderfull:
            optimizer.zero_grad()
            inputs, targets = x.to(self.device), y.to(self.device)
            outputs = server_model(inputs)
            criteria = nn.CrossEntropyLoss()
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step(model_grads)
            distance = LargeMargin(server_model, targets, outputs)

            loader_size = len(distance)
            sum_largeMargin = torch.sum(distance,axis = 0).item()

        return sum_largeMargin/loader_size


    def train(self):
        loss = []

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            #self.selected_users = self.select_users(glob_iter,self.num_users)
            client_LargeMargin = []

            for user in self.benign_users:
                user.train(self.local_epochs) #* user.train_samples
                #avg_LargeMargin = user.testMargin()
                #client_LargeMargin.append(avg_LargeMargin)
                # user.get_LargeMargin()
                # print(user.get_LargeMargin().size())

                
            

            self.aggregate_grads_AVG()

            #self.server_LargeMargin.append(self.getLargeMargin())
            #self.epoch_client_LargeMargin.append(client_LargeMargin)

            #print('server', self.server_LargeMargin)
            #print('server epoch length', len(self.server_LargeMargin))
            #print('client', self.epoch_client_LargeMargin)
            #print('client epoch length', len(self.epoch_client_LargeMargin))
            #print('client length', len(self.epoch_client_LargeMargin[glob_iter]))



            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)


        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results()
        self.save_model()
        #self.save_ServerlargeMargin()
        #self.save_ClientslargeMargin()
    
