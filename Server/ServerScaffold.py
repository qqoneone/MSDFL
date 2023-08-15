import torch
import os
import torch.nn as nn
from User.UserScaffold import UserScaffold
from User.UserAVG import UserAVG
from User.UserLIE import UserLIE
from User.UserFang import UserFang
from User.UserMimic import UserMimic
from User.UserSH import UserSH
from User.UserGood import UserGood
from User.UserGoodScaffold import UserGoodScaffold
from User.UserAVGbackdoor import UserAVGbackdoor
from Server.Server import Server
from Server.sgd import SGD
from utils.model_utils import read_data, read_user_data
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Implementation for FedAvg Server

class Scaffold(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users, times,n_attackers,attacker_type,lambda_JR,num_goodUsers,iid,resumeround):
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
        self.resumeround=resumeround


        #self.ServerDeltaControl={}
        self.ServerControl={}
        for k, v in self.model.named_parameters():
            self.ServerControl[k] = torch.zeros_like(v.data)
            #self.ServerDeltaControl[k]= torch.zeros_like(v.data)
        print("ServerControl initialized!")




        if num_goodUsers>0:
            for i in range(num_goodUsers):
                id, train,test  = read_user_data(i, data, dataset)
                user = UserGoodScaffold(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
                user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if total_users-n_attackers-num_goodUsers>0:
            for i in range(num_goodUsers, total_users-n_attackers):
                id, train,test  = read_user_data(i, data, dataset)
                user = UserScaffold(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer)
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
        print("Finished creating Scaffold server.")

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


    def aggregate_grads_Scaffold(self):
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
        self.optimizer.step(model_grads)
        
        c_temp={}
        for k, v in self.model.named_parameters():
            c_temp[k] = torch.zeros_like(v.data)
        for user in self.benign_users:
            for k, v in self.model.named_parameters():
                c_temp[k] += user.UserDeltaControl[k] / (len(self.benign_users))  # averaging
                #if torch.sum(torch.isnan(self.ServerControl[k].data)) >0 :
                    #print(self.ServerControl[k].data)
                #if user.id==1 and k =="conv1.weight":
                    #print(user.UserDeltaControl[k])
        

        for k, v in self.model.named_parameters():
            self.ServerControl[k].data+= c_temp[k].data 
            #if k =="conv1.weight":
                #print(self.ServerControl[k].data)
                #print(torch.sum(torch.isnan(self.ServerControl[k].data)))
                






    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            #self.selected_users = self.select_users(glob_iter,self.num_users)
            
            for user in self.benign_users:
                user.train(self.local_epochs,self.ServerControl) #* user.train_samples
                
            

            self.aggregate_grads_Scaffold()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results()
        self.save_model()
        model_path = os.path.join("rebuttal")
        a=""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        dict_attacker_type=self.attacker_type
        if self.n_attackers==0:
            dict_attacker_type="Noattack"
        if self.num_goodUsers>0:
            a="_"+"WithModelJacobian"
        torch.save(self.model,  os.path.join(model_path,dict_attacker_type+"_"+self.algorithm + a+"_"+str(self.resumeround)+".pt"))
    