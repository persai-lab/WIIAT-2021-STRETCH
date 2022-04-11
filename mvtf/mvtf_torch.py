import torch
from torch import nn
import torch.autograd


class MVTF_Torch(torch.nn.Module):
    def __init__(self, n_users, n_attempts, n_items, n_factors=2, seed=1024):
        super().__init__()
        torch.random.manual_seed(seed)
        self.n_users = n_users
        self.n_attempts = n_attempts
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.time_factors = nn.Embedding(n_attempts, n_factors * n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.stress_item_factor = nn.Embedding(1, n_factors)

        self.time_biases = nn.Embedding(n_attempts, 1)

        self.stress_user_biases = nn.Embedding(n_users, 1)
        self.stress_item_biases = nn.Embedding(1, 1)

        self.rate_user_biases = nn.Embedding(n_users, 1)
        self.rate_item_biases = nn.Embedding(n_items, 1)

        self.done_user_biases = nn.Embedding(n_users, 1)
        self.done_item_biases = nn.Embedding(n_items, 1)

        # self.user_biases = nn.Embedding(n_users, 1)
        # self.item_biases = nn.Embedding(n_items, 1)

    def forward(self, user, attempt, item, view):
        if view == 1:
            u_factor = self.user_factors(user).squeeze(dim=0)
            t_factor = self.time_factors(attempt).squeeze(dim=0)
            t_matrix = t_factor.reshape(self.n_factors, self.n_factors)
            stress = self.stress_user_biases(user).squeeze() + self.time_biases(attempt).squeeze() \
                     + self.stress_item_biases(torch.tensor(0)).squeeze()
            stress += torch.dot(torch.matmul(u_factor, t_matrix),
                              self.stress_item_factor(torch.tensor(0)))
            stress = torch.sigmoid(stress) * 5.
            # stress = torch.clamp(stress, min=0., max=5.)
            return stress.unsqueeze(dim=0)
        elif view == 2:
            u_factor = self.user_factors(user).squeeze(dim=0)
            i_factor = self.item_factors(item).squeeze(dim=0)
            rate = self.rate_user_biases(user).squeeze() + self.rate_item_biases(item).squeeze()
            rate += torch.dot(u_factor, i_factor)
            rate = torch.sigmoid(rate) * 5.
            # rate = torch.clamp(rate, min=1., max=5.)
            return rate.unsqueeze(dim=0)

        elif view == 3:
            u_factor = self.user_factors(user).squeeze(dim=0)
            t_factor = self.time_factors(attempt).squeeze(dim=0)
            t_matrix = t_factor.reshape(self.n_factors, self.n_factors)
            i_factor = self.item_factors(item).squeeze(dim=0)
            pred = self.done_user_biases(user).squeeze() + self.time_biases(attempt).squeeze() + \
                   self.done_item_biases(item).squeeze()
            # print(u_factor.shape)
            # print(t_matrix.shape)
            # print(i_factor.shape)
            pred += torch.dot(torch.matmul(u_factor, t_matrix), i_factor)
            return torch.sigmoid(pred).unsqueeze(dim=0)
