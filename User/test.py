import numpy as np
import torch
import matplotlib.pyplot as plt


server = np.load('/home/Wudi-007/ModelJacobian/journalCurve/Cifar10_FedAvgserverLarginMargin_IID.npy', allow_pickle=True)

# print(server)
client = np.load('/home/Wudi-007/ModelJacobian/journalCurve/Cifar10_FedAvgclientsLarginMargin_IID.npy', allow_pickle=True)

# print(client)

x_600=np.arange(600)

client1 = [item[0] for item in client]
client2 = [item[1] for item in client]
client3 = [item[2] for item in client]
client4 = [item[3] for item in client]
client5 = [item[4] for item in client]
client6 = [item[5] for item in client]
client7 = [item[6] for item in client]
client8 = [item[7] for item in client]
client9 = [item[8] for item in client]
client10 = [item[9] for item in client]
# client_avg = []
# for num in client:
#     print(num)
#     cur_sum = np.sum(num)
#     client_avg.append(cur_sum/len(num))
# print(client_avg)
# print(client1)

plt.figure(figsize=(4,4))
lw = 0.9
fs = 13
# plt.subplots(1,2, constrained_layout=True)
# plt.subplot(121)
# # plt.subplots(figsize = (4,4))
# plt.plot(x_600, server, label = 'server', linewidth =lw)
# plt.xlabel('Communication Rounds',fontsize=fs)
# plt.ylabel('Margin',fontsize=fs)
# plt.title('Server',fontsize=fs)
# plt.legend()
# plt.grid(linestyle='-.')

# plt.subplot(122)
# plt.subplots(figsize = (4,4))
plt.plot(x_600,client1,label="client1", linewidth =lw)
plt.plot(x_600,client2,label="client2", linewidth =lw)
plt.plot(x_600,client3,label="client3", linewidth =lw)
plt.plot(x_600,client4,label="client4", linewidth =lw)
plt.plot(x_600,client5,label="client5", linewidth =lw)
plt.plot(x_600,client6,label="client6", linewidth =lw)
plt.plot(x_600,client7,label="client7", linewidth =lw)
plt.plot(x_600,client8,label="client8", linewidth =lw)
plt.plot(x_600,client9,label="client9", linewidth =lw)
plt.plot(x_600,client10,label="client10", linewidth =lw)
# plt.plot(x_600,client_avg,label="client_avg")
plt.xlabel('Communication Rounds',fontsize=fs)
plt.ylabel('Margin',fontsize=fs)
plt.title('Client1-10',fontsize=fs)
plt.legend()
plt.grid(linestyle='-.')




plt.savefig('/home/Wudi-007/ModelJacobian/journalCurve/clients.pdf',bbox_inches = 'tight')

plt.show()