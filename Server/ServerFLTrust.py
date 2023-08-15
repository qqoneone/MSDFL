import torch
import os
import torch.nn as nn
from User.UserAVG import UserAVG
from Server.Server import Server
from User.UserLIE import UserLIE
from User.UserFang import UserFang
from User.UserMimic import UserMimic
from User.UserSH import UserSH
from User.UserGood import UserGood
from User.UserGoodRisk import UserGoodRisk
from Server.sgd import SGD
from utils.model_utils import read_data, read_user_data
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Implementation for FedAvg Server
def target_transform(target):
    return int(target)
class FLTrust(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users, times,n_attackers,attacker_type,lambda_JR,num_goodUsers,iid,resume,mode):
        super().__init__(device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users, times)
        self.attacker_type=attacker_type
        
        # Initialize data for all  users
        data = read_data(dataset,iid)
        total_users = len(data[0])
        self.num_goodUsers=num_goodUsers
        self.total_users=total_users
        self.n_attackers=n_attackers
        self.benign_users=[]
        self.attackers=[]
        self.attacker_type=attacker_type
        self.mode=mode
        self.iid=iid
        if resume==True:
            model_path = os.path.join("TrainedModels", dataset)
            a=""
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if self.n_attackers==0:
                self.attacker_type="Noattack"
            if self.num_goodUsers>0:
                a="_"+"WithModelJacobian"
            model_path=os.path.join(model_path,self.attacker_type+"_"+self.algorithm + a+".pt")
            self.load_model(model_path)
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion=nn.CrossEntropyLoss()

        if num_goodUsers>0:
            for i in range(num_goodUsers):
                id, train,test  = read_user_data(i, data, dataset)
                if mode=="risk":
                    user = UserGoodRisk(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
                if mode=="logits":
                    user = UserGood(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
                user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if total_users-n_attackers-num_goodUsers>0:
            for i in range(total_users-n_attackers-num_goodUsers):
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




        # Root dataset is simulated as a root user in FL system
        root_dataset=self.generate_root_data(dataset)
        if num_goodUsers>0:
            if mode =="logits":
                self.root_user=UserGood(device, id+1, root_dataset, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
            if mode =="risk":
                self.root_user=UserGoodRisk(device, id+1, root_dataset, model, batch_size, learning_rate,local_epochs, optimizer,lambda_JR)
        else:
            self.root_user=UserAVG(device, id+1, root_dataset, model, batch_size, learning_rate,local_epochs, optimizer)
        print("Initializing root user.")
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FLTrust server.")
        if mode=="risk":
            print("Initialized good risk clients.")
        if mode=="logits":
            print("Initialized good logits clients.")
    def generate_root_data(self,dataset):
        if dataset=="Cifar10":
            data_loc='/data/wudi/cifar10_data/'
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CIFAR10(root=data_loc, train=True,download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True)
            for inputs,targets in trainloader:
                train_data = [(x, y) for x, y in zip(inputs,targets)]
                break
            return train_data
        if dataset =="MNIST":
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            data_loc='/data/wudi/MNIST_data/'
            trainset = torchvision.datasets.MNIST(root=data_loc, train=True,download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True)
            for inputs,targets in trainloader:
                train_data = [(x, y) for x, y in zip(inputs,targets)]
                break
            return train_data
        if dataset =="FashionMNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            data_loc='/data/wudi/MNIST_data/'
            trainset = torchvision.datasets.FashionMNIST(root=data_loc, train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True)
            for inputs,targets in trainloader:
                train_data = [(x, y) for x, y in zip(inputs,targets)]
                break
            return train_data
        if dataset == "SVHN":
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data_loc='/data/wudi/SVHN_data/'
            trainset = torchvision.datasets.SVHN(root=data_loc, split='train',download=True, transform=transform,target_transform=target_transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True)
            for inputs,targets in trainloader:
                train_data = [(x, y) for x, y in zip(inputs,targets)]
                break
            return train_data


    def Evaluate(self):
        global best_acc
        self.model.eval()
        device=self.device
        net=self.model
        criterion=self.criterion
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print("Average Global Accurancy: ", 100.*correct/total)
            print("Average Global Trainning Loss: ",test_loss/100)
    def aggregate_grads_FLTrust(self):
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        self.optimizer.zero_grad()
        #if(self.num_users = self.to)
        root_grad=self.root_user.get_grads()
        user_grads=[]
        final_grad=[] 
        for user in self.benign_users:
            param_grad=user.get_grads()
            user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)
        if self.n_attackers>0:
            if self.attacker_type=="Fang" or self.attacker_type=="SH":
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

        final_grad= self.FLTrust(user_grads,root_grad)

        #print(final_grad)
        start_idx=0
        model_grads=[]
        for param in self.model.parameters():
            param_=final_grad[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.to(self.device)
            model_grads.append(param_)
        self.optimizer.step(model_grads)
    def FLTrust(self,all_grads,root_grad):
        stack_root_grad=torch.stack([root_grad]*self.total_users)
        TS=torch.cosine_similarity(stack_root_grad,all_grads)
        relu = nn.ReLU(inplace=True)
        print(TS)
        TS=relu(TS)
        norm_root_grad=torch.norm(root_grad,2)
        final_grad=[]
        for TSi,user_grad in zip(TS,all_grads):
            norm_user_grad=torch.norm(user_grad,2)
            final_user_grad=TSi*norm_root_grad*user_grad
            final_user_grad=final_user_grad/norm_user_grad
            final_grad=final_user_grad if len(final_grad)==0 else final_grad+final_user_grad
        final_grad=final_grad/torch.sum(TS)
        return final_grad


    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #sending current parameters to all users
            self.send_parameters()
            # sending current parameters to root user
            self.root_user.set_parameters(self.model)




            # Evaluate model each interation
            self.evaluate()

            #benign clients training            
            for user in self.benign_users:
                user.train(self.local_epochs) #* user.train_samples
            #root user training
            self.root_user.train(self.local_epochs)
                
            

            self.aggregate_grads_FLTrust()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results()
        self.save_model()
    