import torch
from torch import Tensor
import torch.nn as nn
import config as cfg

class NSLoss(nn.Module):
    '''negetive sampling loss'''
    def __init__(self, sim=None, act=None, loss=None):
        """
        Args:
            sim: a similarity function
            act: a activator function that map the similarity to a valid domain
            loss: a criterion measuring the predicted similarities and the ground truth labels
        """
        super(NSLoss, self).__init__()
        if sim is not None:
            self.sim = sim
        else:
            self.sim = self.inner_product
        if act is not None:
            self.act = act
        else:
            self.act = self.identity
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.MSELoss()

    @staticmethod
    def inner_product(x, y):
        return x.mul(y).sum(dim=1, keepdim=True)  # -1 最后一维 相加
        # return x.mm(y.t())

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def get_adjacent(idx_s: Tensor, idx_t: Tensor, adj_mat) ->Tensor:
        """
        Given indices, get corresponding weights from a weighted adjacency matrix.
        Args:
            i_s: row indices
            i_t: column indices
            weights: a weighted adjacency matrix, a sparse matrix is preferred as it saves memory
        Return:
            a weight vector of length len(i_s)
        """
        if adj_mat is None:
            return torch.ones(len(idx_s))
        else:
            idx_s = idx_s.tolist()
            idx_t = idx_t.tolist()
            adjacent = adj_mat[idx_s, idx_t]
            return torch.FloatTensor(adjacent).squeeze()

    @staticmethod
    def sample(neg_num: int, batch_size: int, probs=None, node_num: int=16) ->Tensor:
        """
        Get indices of negative samples w.r.t. given probability distribution.
        Args:
            neg: number of negative samples
            probs: a probability vector for a multinomial distribution
            batch_size: batch size
            scale: maximum index, valid only when probs is None, leading to a uniform distribution over [0, scale - 1]
        Return:
            a LongTensor with shape [neg, batch_size]
        """
        assert neg_num > 0
        if probs is None:
            idx = torch.Tensor(batch_size * neg_num).uniform_(0, node_num).long()
        else:
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs)
            idx = torch.multinomial(probs, batch_size * neg_num, True)  # true 有放回取样
        return idx.view(neg_num, batch_size)

    def get_xy(self, *input):
        # s for source and t for target
        """
        Calculate the pairwise similarities between two set of vectors in the common space.
        inputs contains a pack of arguements.
        Args:
            embed_s/embed_t: Tensor, [N, dim], user embedding matrix from the source/target network
            idx_s/idx_t: LongTensor, [batch], user indices from the source/target network
            map_s/map_t: MLP, mappings that map users from source/target network to the common space
            neg_num: number of negative samples, int
            probs: a probability vector for the negative sampling distribution
            adj_mat: a sparse weighted adjacency matrix
        Return:
            a similarity vectors and its corresponding ground truth labels.
        """
        embed_s, embed_t, idx_s, idx_t, map_s, map_t,\
            neg_num, probs, adj_mat = input
        x_s, x_t = embed_s[idx_s], embed_t[idx_t]
        x_s, x_t = map_s(x_s), map_t(x_t)
        # calculate node similarities, stand for positive sample
        # x_pos = x_t
        y_pos = self.get_adjacent(idx_s, idx_t, adj_mat)

        if neg_num > 0:
            batch_size = len(idx_s)
            idx_neg = self.sample(neg_num, batch_size, probs, len(embed_t))
            # idx_neg = self.sample(neg_num, batch_size=batch_size, node_num=len(embed_t))
            x_neg = torch.stack([map_t(embed_t[idx]) for idx in idx_neg])
            y_neg = torch.stack([
                        self.get_adjacent(idx_s, idx, adj_mat) for idx in idx_neg
                    ]).view(-1)
            x_t = torch.cat([x_t.unsqueeze(0), x_neg], dim=0)  # [6, 128, 640]
            y = torch.cat([y_pos.view(-1), y_neg])
        else:
            # x_t = x_pos
            y = y_pos

        return x_s, x_t, y

    def forward(self, *input):
        x_s, x_t, y = self.get_xy(*input)
        # x = torch.cat([x_s, x_t]).view(-1)
        y_hat = self.act(
                torch.stack([self.sim(x_s, x) for x in x_t
                           ]).view(-1)
                )  # 原文Eq.11
        if y_hat.is_cuda:
            y = y.to(cfg.device)
        # y_hat = y_hat.to(cfg.d)
        return self.loss(y_hat, y)  # Eq.12

