import torch
import os
import torch.nn as nn
from User.UserAVG import UserAVG
from User.UserLIE import UserLIE
from User.UserAVGbackdoor import UserAVGbackdoor
from User.UserAVGbackdoorAttack import UserAVGbackdoorAttack
from Server.Server import Server
from Server.sgd import SGD
from utils.model_utils import read_data, read_user_data
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Implementation for FedAvg Server

class FedAvgBackdoor(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users, times,n_attackers,attacker_type,backdoor,target_label,attack_epoch):
        super().__init__(device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset,iid=True)
        total_users = len(data[0])
        self.n_attackers=n_attackers
        self.benign_users=[]
        self.attackers=[]
        #load model
        self.load_model("/home/Wudi-007/ModelJacobian/TrainedModels/Cifar100/backdoor_FedAvg.pt")
        ###
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion=nn.CrossEntropyLoss()
        self.dataset=dataset
        self.backdoor=backdoor
        self.target_label=target_label
        self.backdoor_acc=[]
        self.attacker_type=attacker_type
        if dataset=="Cifar100":
            for i in range(total_users-n_attackers):
                id, train  = read_user_data(i, data, dataset)
                user = UserAVGbackdoor(device, id, train, model, batch_size, learning_rate,local_epochs, optimizer)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            if self.n_attackers>0 and attacker_type=="backdoor" :
                print("Initializing %d %s attackers" %(n_attackers,attacker_type))
                for i in range(n_attackers):
                    id, train  = read_user_data(i, data, dataset)
                    user=UserAVGbackdoorAttack( device, id, train, model, batch_size, learning_rate,local_epochs, optimizer,backdoor,target_label,attack_epoch)
                    self.benign_users.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
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

            mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
            testset = torchvision.datasets.CIFAR100(root='/data/wudi/cifar100_data/', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)    

        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

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
            self.rs_glob_acc.append(100.*correct/total)
            self.rs_glob_loss.append(test_loss/100)
            print("Average Global Accurancy: ", 100.*correct/total)
            print("Average Global Trainning Loss: ",test_loss/100)
        if self.n_attackers>0:
            with torch.no_grad():
                backdoor_total=0
                backdoor_correct=0
                for batch_idx, (inputs, targets) in enumerate(self.testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    for i,j in zip(predicted,targets):
                        if j==self.backdoor:
                            backdoor_total=backdoor_total+1
                        if j==self.backdoor and i==self.target_label:
                            backdoor_correct=backdoor_correct+1
                print("Average Backdoor Accurancy: ", 100.*backdoor_correct/backdoor_total)
                self.backdoor_acc.append(100.*backdoor_correct/backdoor_total)



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
            final_grad=param_grad*user.train_samples / total_train if len(final_grad)==0 else final_grad+param_grad*user.train_samples / total_train

        
        #final_grad=torch.mean(user_grads,dim=0)
        #print(final_grad)
        start_idx=0
        model_grads=[]
        for param in self.model.parameters():
            param_=final_grad[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.to(self.device)
            model_grads.append(param_)
        self.optimizer.step(model_grads)



    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.Evaluate()

            #self.selected_users = self.select_users(glob_iter,self.num_users)
            
            for user in self.benign_users:
                user.train(glob_iter) #* user.train_samples
                
            

            self.aggregate_grads_AVG()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
        self.save_model()
    