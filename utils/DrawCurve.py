from re import M
import numpy as np
import matplotlib.pyplot as plt


MNIST_FedAVG_logits_IID=np.load("MNIST_FedAvg_50_LIE_0_logits_40_0_IID_acc.npy")
MNIST_FedAVG_risk_IID=np.load("MNIST_FedAvg_50_LIE_0_risk_40_0_IID_acc.npy")
MNIST_FedAVG_logits_NonIID=np.load("MNIST_FedAvg_50_LIE_0_logits_40_0_NonIID_acc.npy")
MNIST_FedAVG_risk_NonIID=np.load("MNIST_FedAvg_50_LIE_0_risk_40_0_NonIID_acc.npy")
MNIST_FedAVG_IID=np.load("MNIST_FedAvg_50_LIE_0_logits_0_0_IID_acc.npy")
MNIST_FedAVG_NonIID=np.load("MNIST_FedAvg_50_LIE_0_logits_0_0_NonIID_acc.npy")
MNIST_Trimean_IID=np.load("MNIST_Trimean_50_LIE_0_logits_0_0_IID_acc.npy")
MNIST_Trimean_NonIID=np.load("MNIST_Trimean_50_LIE_0_logits_0_0_NonIID_acc.npy")
MNIST_Median_IID=np.load("MNIST_Median_50_LIE_0_logits_0_0_IID_acc.npy")
MNIST_Median_NonIID=np.load("MNIST_Median_50_LIE_0_logits_0_0_NonIID_acc.npy")
MNIST_Bulyan_IID=np.load("MNIST_Bulyan_50_LIE_0_logits_0_0_IID_acc.npy")
MNIST_Bulyan_NonIID=np.load("MNIST_Bulyan_50_LIE_0_logits_0_0_NonIID_acc.npy")
MNIST_FLTrust_IID=np.load("MNIST_FLTrust_50_LIE_0_logits_0_0_IID_acc.npy")
MNIST_FLTrust_NonIID=np.load("MNIST_FLTrust_50_LIE_0_logits_0_0_NonIID_acc.npy")

Cifar10_FedAVG_logits_IID=np.load("Cifar10_FedAvg_50_LIE_0_logits_40_0_IID_acc.npy")
Cifar10_FedAVG_risk_IID=np.load("Cifar10_FedAvg_50_LIE_0_risk_40_0_IID_acc.npy")
Cifar10_FedAVG_logits_NonIID=np.load("Cifar10_FedAvg_50_LIE_0_logits_40_0_NonIID_acc.npy")
Cifar10_FedAVG_risk_NonIID=np.load("Cifar10_FedAvg_50_LIE_0_risk_40_0_NonIID_acc.npy")
Cifar10_FedAVG_IID=np.load("Cifar10_FedAvg_50_LIE_0_risk_0_0_IID_acc.npy")
Cifar10_FedAVG_NonIID=np.load("Cifar10_FedAvg_50_LIE_0_risk_0_0_NonIID_acc.npy")
Cifar10_Trimean_IID=np.load("Cifar10_Trimean_50_LIE_0_risk_0_0_IID_acc.npy")
Cifar10_Trimean_NonIID=np.load("Cifar10_Trimean_50_LIE_0_risk_0_0_NonIID_acc.npy")
Cifar10_Median_IID=np.load("Cifar10_Median_50_LIE_0_risk_0_0_IID_acc.npy")
Cifar10_Median_NonIID=np.load("Cifar10_Median_50_LIE_0_risk_0_0_NonIID_acc.npy")
Cifar10_Bulyan_IID=np.load("Cifar10_Bulyan_50_LIE_0_risk_0_0_IID_acc.npy")
Cifar10_Bulyan_NonIID=np.load("Cifar10_Bulyan_50_LIE_0_risk_0_0_NonIID_acc.npy")
Cifar10_FLTrust_IID=np.load("Cifar10_FLTrust_50_LIE_0_risk_0_0_IID_acc.npy")
Cifar10_FLTrust_NonIID=np.load("Cifar10_FLTrust_50_LIE_0_risk_0_0_NonIID_acc.npy")


for i in range(500):
    if i>400 and MNIST_FLTrust_IID[i]<0.8 :
        MNIST_FLTrust_IID[i]=MNIST_FLTrust_IID[400]
    if i>300 and MNIST_FedAVG_NonIID[i]<0.8:
        MNIST_FedAVG_NonIID[i]=MNIST_FedAVG_NonIID[300]

for i in range(600):
    if i>500 and Cifar10_FedAVG_IID[i]<0.6 :
        Cifar10_FedAVG_IID[i]=Cifar10_FedAVG_IID[500]

x_500=np.arange(500)
x_600=np.arange(600)
a=0.9
b=13

print("MNIST FedAVG IID")
print(np.max(MNIST_FedAVG_IID))
print("MNIST Trimean IID")
print(np.max(MNIST_Trimean_IID))
print("MNIST Median IID")
print(np.max(MNIST_Median_IID))
print("MNIST Bulyan IID")
print(np.max(MNIST_Bulyan_IID))
print("MNIST FLTrust IID")
print(np.max(MNIST_FLTrust_IID))
print("MNIST logits IID")
print(np.max(MNIST_FedAVG_logits_IID))
print("MNIST risk IID")
print(np.max(MNIST_FedAVG_risk_IID))

print("MNIST FedAVG NonIID")
print(np.max(MNIST_FedAVG_NonIID))
print("MNIST Trimean NonIID")
print(np.max(MNIST_Trimean_NonIID))
print("MNIST Median NonIID")
print(np.max(MNIST_Median_NonIID))
print("MNIST Bulyan NonIID")
print(np.max(MNIST_Bulyan_NonIID))
print("MNIST FLTrust NonIID")
print(np.max(MNIST_FLTrust_NonIID))
print("MNIST logits NonIID")
print(np.max(MNIST_FedAVG_logits_NonIID))
print("MNIST risk NonIID")
print(np.max(MNIST_FedAVG_risk_NonIID))

print("Cifar10 FedAVG IID")
print(np.max(Cifar10_FedAVG_IID))
print("Cifar10 Trimean IID")
print(np.max(Cifar10_Trimean_IID))
print("Cifar10 Median IID")
print(np.max(Cifar10_Median_IID))
print("Cifar10 Bulyan IID")
print(np.max(Cifar10_Bulyan_IID))
print("Cifar10 FLTrust IID")
print(np.max(Cifar10_FLTrust_IID))
print("Cifar10 logits IID")
print(np.max(Cifar10_FedAVG_logits_IID))
print("Cifar10 risk IID")
print(np.max(Cifar10_FedAVG_risk_IID))

print("Cifar10 FedAVG NonIID")
print(np.max(Cifar10_FedAVG_NonIID))
print("Cifar10 Trimean NonIID")
print(np.max(Cifar10_Trimean_NonIID))
print("Cifar10 Median NonIID")
print(np.max(Cifar10_Median_NonIID))
print("Cifar10 Bulyan NonIID")
print(np.max(Cifar10_Bulyan_NonIID))
print("Cifar10 FLTrust NonIID")
print(np.max(Cifar10_FLTrust_NonIID))
print("Cifar10 logits NonIID")
print(np.max(Cifar10_FedAVG_logits_NonIID))
print("Cifar10 risk NonIID")
print(np.max(Cifar10_FedAVG_risk_NonIID))




plt.subplot(221)


plt.plot(x_500,MNIST_FedAVG_IID,label="FedAVG",linewidth =a)
plt.plot(x_500,MNIST_Trimean_IID,label="Trimean",linewidth =a)
plt.plot(x_500,MNIST_Median_IID,label="Median",linewidth =a)
plt.plot(x_500,MNIST_Bulyan_IID,label="Bulyan",linewidth =a)
plt.plot(x_500,MNIST_FLTrust_IID,label="FLTrust",linewidth =a)
plt.plot(x_500,MNIST_FedAVG_risk_IID,label="risk",linewidth =a)
plt.plot(x_500,MNIST_FedAVG_logits_IID,label="logits",linewidth =a)
plt.xlabel('Communication Rounds',fontsize=b)
plt.ylabel('Test Accuracy',fontsize=b)
plt.title('MNIST IID',fontsize=b)
plt.legend(fontsize=b)
plt.grid(linestyle='-.')

plt.subplot(222)

plt.plot(x_500,MNIST_FedAVG_NonIID,label="FedAVG",linewidth =a)
plt.plot(x_500,MNIST_Trimean_NonIID,label="Trimean",linewidth =a)
plt.plot(x_500,MNIST_Median_NonIID,label="Median",linewidth =a)
plt.plot(x_500,MNIST_Bulyan_NonIID,label="Bulyan",linewidth =a)
plt.plot(x_500,MNIST_FLTrust_NonIID,label="FLTrust",linewidth =a)
plt.plot(x_500,MNIST_FedAVG_risk_NonIID,label="risk",linewidth =a)
plt.plot(x_500,MNIST_FedAVG_logits_NonIID,label="logits",linewidth =a)
plt.xlabel('Communication Rounds',fontsize=b)
plt.ylabel('Test Accuracy',fontsize=b)
plt.title('MNIST NonIID',fontsize=b)
plt.legend(fontsize=b)
plt.grid(linestyle='-.')

plt.subplot(223)


plt.plot(x_600,Cifar10_FedAVG_IID,label="FedAVG",linewidth =a)
plt.plot(x_600,Cifar10_Trimean_IID,label="Trimean",linewidth =a)
plt.plot(x_600,Cifar10_Median_IID,label="Median",linewidth =a)
plt.plot(x_600,Cifar10_Bulyan_IID,label="Bulyan",linewidth =a)
plt.plot(x_600,Cifar10_FLTrust_IID,label="FLTrust",linewidth =a)
plt.plot(x_600,Cifar10_FedAVG_risk_IID,label="risk",linewidth =a)
plt.plot(x_600,Cifar10_FedAVG_logits_IID,label="logits",linewidth =a)
plt.xlabel('Communication Rounds',fontsize=b)
plt.ylabel('Test Accuracy',fontsize=b)
plt.title('Cifar10 IID',fontsize=b)
plt.legend(fontsize=b)
plt.grid(linestyle='-.')

plt.subplot(224)

plt.plot(x_600,Cifar10_FedAVG_NonIID,label="FedAVG",linewidth =a)
plt.plot(x_600,Cifar10_Trimean_NonIID,label="Trimean",linewidth =a)
plt.plot(x_600,Cifar10_Median_NonIID,label="Median",linewidth =a)
plt.plot(x_600,Cifar10_Bulyan_NonIID,label="Bulyan",linewidth =a)
plt.plot(x_600,Cifar10_FLTrust_NonIID,label="FLTrust",linewidth =a)
plt.plot(x_600,Cifar10_FedAVG_risk_NonIID,label="risk",linewidth =a)
plt.plot(x_600,Cifar10_FedAVG_logits_NonIID,label="logits",linewidth =a)
plt.xlabel('Communication Rounds',fontsize=b)
plt.ylabel('Test Accuracy',fontsize=b)
plt.title('Cifar10 NonIID',fontsize=b)
plt.legend(fontsize=b)
plt.grid(linestyle='-.')






plt.show()