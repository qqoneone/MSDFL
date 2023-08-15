import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserFang(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, optimizer,n_attackers,aggregation_type):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs)
        self.aggregation_type=aggregation_type
        self.n_attackers=n_attackers
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]
    def compute_lambda_fang(self,all_updates, model_re, n_attackers):

        distances = []
        n_benign, d = all_updates.shape
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1)
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances[distances == 0] = 10000
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
        min_score = torch.min(scores)
        term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
        max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

        return (term_1 + max_wre_dist)
    def multi_krum(self,all_updates, n_attackers, multi_k=False):
        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            #torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)

            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)

            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break

        aggregate = torch.mean(candidates, dim=0)

        return aggregate, np.array(candidate_indices)

    def generated_gradients(self, all_updates):
        n_attackers=self.n_attackers
        model_re=torch.mean(all_updates, 0)
        deviation=torch.sign(model_re)
        lamda = self.compute_lambda_fang(all_updates, model_re,self.n_attackers)
        if lamda>1:
            lamda=1
        print("lamda:")
        print(lamda)
        
        mal_updates=[]
        if self.aggregation_type=="Krum" or self.aggregation_type=="Bulyan":
            threshold = 1e-5
            mal_updates = []
            while lamda > threshold:
                mal_update = (- lamda * deviation)

                mal_updates = torch.stack([mal_update] * n_attackers)
                mal_updates = torch.cat((mal_updates, all_updates), 0)
                agg_grads, krum_candidate = self.multi_krum(mal_updates, self.n_attackers, multi_k=False)
                
                if krum_candidate < n_attackers:
                    return torch.stack([mal_update]*n_attackers)
                
                lamda *= 0.5
            if not len(mal_updates):
                print(lamda, threshold)
                mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
        else:
            b = 2
            max_vector = torch.max(all_updates, 0)[0]
            min_vector = torch.min(all_updates, 0)[0]

            max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
            min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

            max_[max_ == 1] = b
            max_[max_ == 0] = 1 / b
            min_[min_ == 1] = b
            min_[min_ == 0] = 1 / b

            max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
            min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

            rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

            max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
            min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

            mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
                [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

            mal_updates = mal_vec

        return mal_updates
