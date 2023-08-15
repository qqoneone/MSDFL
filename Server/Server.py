import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics, LargeMargin
from torch.utils.data import DataLoader
from utils.model_utils import read_data, read_user_data
import copy

class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,
                 num_glob_iters, local_epochs, optimizer,num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_glob_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_train_loss=[]
        self.times = times
        self.rs_glob_Jacobian_loss=[]
        self.rs_glob_RiskJacobian_loss=[]
        self.server_LargeMargin = []
        self.epoch_client_LargeMargin = []
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

        return copy.deepcopy(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
    




    def save_model(self):
        model_path = os.path.join("TrainedModels", self.dataset)
        a=""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if self.n_attackers==0:
            self.attacker_type="Noattack"
        if self.num_goodUsers>0:
            a="_"+"WithModelJacobian"
        
        torch.save(self.model,  os.path.join(model_path,self.attacker_type+"_"+self.algorithm + a+".pt"))

    def load_model(self,model_path):
        assert (os.path.exists(model_path))
        print("model loaded!")
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        '''
        alg = self.dataset + "_" + self.algorithm
        alg = alg  +  "_" + str(self.num_users)+"_"+str(self.n_attackers)+"_" +self.attacker_type 
        alg = alg + "_" + str(self.times)
        if self.attacker_type=="backdoor":
            alg=alg+"_" +str(self.backdoor)+"_" +str(self.target_label) 
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_glob_loss', data=self.rs_glob_loss)
                if self.n_attackers>0 and self.attacker_type=="backdoor":
                    hf.create_dataset('backdoor_glob_acc', data=self.backdoor_acc)
                hf.close()
        '''
        alg = self.dataset + "_" + self.algorithm
        alg = alg  +  "_" + str(self.num_users)+"_"+self.attacker_type+"_"+str(self.n_attackers)+"_"+self.mode+"_"+str(self.num_goodUsers) 
        alg = alg + "_" + str(self.times)+"_"+str(self.lambda_JR)
        if self.iid==True:
            alg = alg + "_" + "IID"
        else:
            alg = alg + "_" + "NonIID"
        alg_acc=alg+"_acc"
        alg_loss=alg+"_loss"
        np.save(os.path.join("journalresult", alg_acc),self.rs_glob_acc)
        np.save(os.path.join("journalresult", alg_loss),self.rs_glob_loss)

    def save_jacobian(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg  +  "_" + str(self.num_users)+"_"+self.attacker_type+"_"+str(self.n_attackers)+"_"+self.mode+"_"+str(self.num_goodUsers) 
        alg = alg + "_" + str(self.times)
        if self.iid==True:
            alg = alg + "_" + "IID"
        else:
            alg = alg + "_" + "NonIID"
        alg_loss=alg+"_Jacobianloss"
        np.save(os.path.join("journalCurve", alg_loss),self.rs_glob_Jacobian_loss)

    def save_riskjacobian(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg  +  "_" + str(self.num_users)+"_"+self.attacker_type+"_"+str(self.n_attackers)+"_"+self.mode+"_"+str(self.num_goodUsers) 
        alg = alg + "_" + str(self.times)
        if self.iid==True:
            alg = alg + "_" + "IID"
        else:
            alg = alg + "_" + "NonIID"
        alg_loss=alg+"_RiskJacobianloss"
        np.save(os.path.join("journalCurve", alg_loss),self.rs_glob_RiskJacobian_loss)

    def save_ServerlargeMargin(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "serverLarginMargin"
        if self.iid==True:
            alg = alg + "_" + "IID"
        else:
            alg = alg + "_" + "NonIID"
        np.save(os.path.join("journalCurve", alg), self.server_LargeMargin)

    def save_ClientslargeMargin(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "clientsLarginMargin"
        if self.iid==True:
            alg = alg + "_" + "IID"
        else:
            alg = alg + "_" + "NonIID"
        np.save(os.path.join("journalCurve", alg), self.epoch_client_LargeMargin)

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct

    def testJacobian(self):
        Jloss=0
        for c in self.users:
            jloss=c.testJacobian()
            Jloss=Jloss+jloss
        return Jloss
    def testRiskJacobian(self):
        Jloss=0
        for c in self.users:
            jloss=c.testRiskJacobian()
            Jloss=Jloss+jloss
        return Jloss


    def get_LargeMargin(self):
        data = read_data(self.dataset, "iid")
        id, train_data, test_data = read_user_data(0, data, self.dataset)
        train_samples = len(train_data)
        trainloaderfull = DataLoader(train_data, train_samples)
        for x, y in trainloaderfull:
            inputs, targets = x.to(self.device), y.to(self.device)
            outputs = self.model(inputs)
            distance = LargeMargin(self.model, targets, outputs)
            return distance
        # distance_list = []
        # distance_sum = []
        # for c in self.users:
        #     if(len(distance_sum) == 0):
        #         distance_sum.append()
        #
        #     c_distance = c.get_LargeMargin
        #     distance_list.append(c_distance)
        #     if(len(distance_sum) == 0):
        #         distance_sum.append(c_distance)
        #     else:
        #         distance_sum = [[distance_sum + c_distance for distance_sum, c_distance in zip(sublist_a, sublist_b)] for sublist_a, sublist_b in zip(distance_sum, c_distance)]
        # return distance_list,distance_sum

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses


    def evaluate(self):

        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)

    def evaluateJacobian(self):
        JacobianLoss=self.testJacobian()
        JacobianLoss=JacobianLoss/self.num_users
        self.rs_glob_Jacobian_loss.append(JacobianLoss)
        print("Average Global Jacobian Regularization: ",JacobianLoss)
    def evaluateRiskJacobian(self):
        JacobianLoss=self.testRiskJacobian()
        JacobianLoss=JacobianLoss/self.num_users
        self.rs_glob_RiskJacobian_loss.append(JacobianLoss)
        print("Average Global Risk Jacobian Regularization: ",JacobianLoss)

    def evaluateLargeMargin(self):
        largeMarginList,largeMarginSum = self.get_LargeMargin()
        largeMarginServer =  [[x / 50 + 1 for x in sublist] for sublist in largeMarginSum]



    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
