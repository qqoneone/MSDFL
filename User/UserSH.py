import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserSH(Userbase):
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

    def multi_krum(self,all_updates, n_attackers, multi_k=False):
        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
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
    def tr_mean(self,all_updates, n_attackers):
        sorted_updates,indice = torch.sort(all_updates, 0)
        out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
        return out
    def generated_gradients(self, all_updates):
        model_re=torch.mean(all_updates, 0)
        deviation = model_re / torch.norm(model_re)
        n_attackers=self.n_attackers
        if self.aggregation_type=="Trimean" or self.aggregation_type=="Median" or self.aggregation_type=="FedAvg" or self.aggregation_type=="FLTrust":
            lamda = torch.Tensor([10.0]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
            # print(lamda)
            threshold_diff = 1e-5
            prev_loss = -1
            lamda_fail = lamda
            lamda_succ = 0
            iters = 0 
            while torch.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = (model_re - lamda * deviation)
                mal_updates = torch.stack([mal_update] * n_attackers)
                mal_updates = torch.cat((mal_updates, all_updates), 0)

                agg_grads = self.tr_mean(mal_updates, n_attackers)
                
                loss = torch.norm(agg_grads - model_re)
                
                if prev_loss < loss:
                    # print('successful lamda is ', lamda)
                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2
                prev_loss = loss
                
            mal_update = (model_re - lamda_succ * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)


            return mal_updates

        else:
            lamda = torch.Tensor([3.0]).cuda()

            threshold_diff = 1e-5
            lamda_fail = lamda
            lamda_succ = 0

            while torch.abs(lamda_succ - lamda) > threshold_diff:
                mal_update = (model_re - lamda * deviation)
                mal_updates = torch.stack([mal_update] * n_attackers)
                mal_updates = torch.cat((mal_updates, all_updates), 0)

                agg_grads, krum_candidate = self.multi_krum(mal_updates, n_attackers, multi_k=True)
                if np.sum(krum_candidate < n_attackers) == n_attackers:
                    # print('successful lamda is ', lamda)
                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2

            mal_update = (model_re - lamda_succ * deviation)
            mal_updates = torch.stack([mal_update] *n_attackers)

            return mal_updates