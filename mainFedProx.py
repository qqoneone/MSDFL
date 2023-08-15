import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import sys

from Models.alexnet import AlexNet
from Models.models import *
from Models.svhn import svhn
from Server.ServerFedProx import FedProx


import torch
torch.manual_seed(0)

def main(dataset, algorithm,batch_size, learning_rate, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, times, gpu,n_attackers,attacker_type,lambda_JR,num_GoodUsers,iid,mu):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda")

    for i in range(times):
        print("---------------Running time:------------",i)
        if dataset=="Cifar10":
            model = AlexNet().to(device)
        if dataset=="SVHN":
            model =CifarNet().to(device)
        if dataset=="Cifar100":
            model = AlexNet(num_classes=20).to(device)
        if dataset=="MNIST" or dataset=="FashionMNIST":
            model=LeNet5().to(device)
        if dataset=="synthetic":
            model = Mclr_Logistic(60,10).to(device)
        # select algorithm
        torch.save(model,  "test.pt")
        if(algorithm == "FedProx"):
            server = FedProx(device, dataset, algorithm, model, batch_size, learning_rate,num_glob_iters, local_epochs, optimizer, numusers, i,n_attackers,attacker_type,lambda_JR,num_GoodUsers,iid,mu)
        server.train()
        server.test()

    # Average data 
    #average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FashionMNIST", choices=["MNIST", "synthetic", "Cifar10","Cifar100","SVHN","FashionMNIST"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=600)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedProx",choices=["pFedMe", "PerAvg", "FedAvg","FedProx"]) 
    parser.add_argument("--numusers", type=int, default=50, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--attacker_type", type=str, default="LIE", choices=["LIE","Fang","SH","Mimic"])
    parser.add_argument("--n_attackers", type=int, default=0, help="The number of attackers")
    parser.add_argument("--lambda_JR", type=float, default=0.0001, help="The propotion of Model Jacobian regularization in Total Loss")
    parser.add_argument("--num_GoodUsers", type=int, default=0, help="The number of Good Users in the system")
    parser.add_argument("--iid", type=bool, default=False, help="IID or NonIID")
    parser.add_argument("--mu", type=float, default=1, help="Hyper parameter for FedProx")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Attacker Type       : {}".format(args.attacker_type))
    print("Number of attackers       : {}".format(args.n_attackers))
    print("lambda_JR       : {}".format(args.lambda_JR))
    print("Number of Good Users       : {}".format(args.num_GoodUsers))
    print("IID       : {}".format(args.iid))
    print("MU       : {}".format(args.mu))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        times = args.times,
        gpu=args.gpu,
        n_attackers=args.n_attackers,
        attacker_type=args.attacker_type,
        lambda_JR=args.lambda_JR,
        num_GoodUsers=args.num_GoodUsers,
        iid=args.iid,
        mu=args.mu
        )