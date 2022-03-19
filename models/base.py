import logging

import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
from utils.extensions import cosine
import config as cfg
from utils.general import write_pickle
from utils.log import create_logger
import numpy as np


class Instructor:
    def __init__(self):
        self.log = create_logger(
            __name__, silent=False,
            to_disk=True, log_file=cfg.log)

    def rename_log(self, filename):
        logging.shutdown()
        os.rename(cfg.log, filename)

    @staticmethod
    def optimize(opt, loss):
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def load_data(input, batch_size):
        data = DataWrapper(input)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True)
        return batches

    @staticmethod
    def early_stop(current, results,
                   size=3, epsilon=5e-5):
        results[:-1] = results[1:]
        results[-1] = current
        assert len(results) == 2 * size
        pre = results[:size].mean()
        post = results[size:].mean()
        return abs(pre - post) > epsilon


class DataWrapper(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class UIL(Instructor):
    """
    A base class for training a UIL model
    """
    def __init__(self, links, k):
        """
        Args:
            idx: ground truth user pairs for training and testing
            k: number of candidates
        """
        super(UIL, self).__init__()
        self.links = links
        _, self.link_test = links
        self.k = k  # k candidates

    def get_embeds(self, is_eval):
        raise NotImplementedError

    def eval_hit_p(self, mask=None, default=0.):
        """
        Evaluation precision@k and hit-precision@k in the training set and testing set.
        Args:
            mask: a matrix masking known matched user pairs
            default: default similarity for matched user pairs in the training set
        Return:
            hit-precision@k
        """
        with torch.no_grad():
            embed_s, embed_t = self.get_embeds(is_eval=True)
            similarity_mat = self.sim_pairwise(embed_s, embed_t)
            if mask is not None:
                similarity_mat *= mask
            train_link, test_link = self.links
            coverage, hit_p = self.get_metrics(similarity_mat,
                                               self.k, train_link)
            self.log.info('Train Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))

            row, col = [list(i) for i in zip(*train_link)]
            # delete nodes and connected link in train set
            # mask similarities of matched user pairs in the train set
            similarity_mat[row] = default
            similarity_mat[:, col] = default
            coverage, hit_p = self.get_metrics(similarity_mat,
                                               self.k, test_link)
            self.log.info('Test Coverage {:.4f} | Hit {:.4f}'.format(
                coverage, hit_p
            ))
        return hit_p

    def sim_pairwise(self, xs, ys):
        return cosine(xs, ys)

    def get_metrics(self, sims_mat, k, link):
        """
        Calculate the average precision@k and hit_precision@k from two sides, i.e., source-to-target and target-to-source.
        Args:
          sims_mat: similarity matrix
          k: number of candidates
          link: index pairs of matched users, i.e., the ground truth
        Return:
          coverage: precision@k
          hit: hit_precison@k
        """
        # row: source node  col: target node
        row, col = [list(i) for i in zip(*link)]
        target = sims_mat[row, col].reshape((-1, 1))
        s_sim = sims_mat[row]
        t_sim = sims_mat.t()[col]
        # match users from source to target
        c_s, h_s = self.score(s_sim, target, k)
        # match users from target to source
        c_t, h_t = self.score(t_sim, target, k)
        # averaging the scores from both sides
        return (c_s + c_t) / 2, (h_s + h_t) / 2

    @staticmethod
    def score(sims: Tensor, target: Tensor, k: int) -> tuple:
        """
        Calculate the average precision@k and hit_precision@k from while matching users from the source network to the target network.
        Args:
            sims: similarity matrix
            k: number of candidates
            target:
        Return:
            coverage: precision@k
            hit: hit_precison@k
        """
        # number of users with similarities larger than the matched users
        rank = (sims >= target).sum(1)
        # rank = min(rank, k + 1)
        rank = rank.min(torch.tensor(k + 1).to(cfg.device))
        temp = (k + 1 - rank).float()
        hit_p = (temp / k).mean()
        coverage = (temp > 0).float().mean()
        return coverage, hit_p

    def report_hit(self, sims_orig, mask=None, default=0.):
        with torch.no_grad():
            train, test = self.links
            for k in [10 * i for i in range(1, 6)]:
                sims = sims_orig.clone()
                coverage, hit_p = self.get_metrics(sims, k, train)
                self.log.info('Train Coverage@{} {:.4f} | Hit@{} {:.4f}'.format(
                    k, coverage, k, hit_p
                ))
                row, col = [list(i) for i in zip(*train)]
                sims[row] = default
                sims[:, col] = default
                coverage, hit_p = self.get_metrics(sims, k, test)
                self.log.info('Test Coverage@{} {:.4f} | Hit@{} {:.4f}'.format(
                    k, coverage, k, hit_p
                ))

    def save_embeddings(self, embed, file_path):
        embed = embed.detach().cpu().numpy()
        fout = open(file_path, 'w')
        node_num = len(embed)
        embed_dim = len(embed[0])
        fout.write("{} {}\n".format(node_num, embed_dim))
        for node in range(node_num):
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in embed[node]])))
        fout.close()
        # write_pickle(embed, file_path)

    def getPairs(self, link, adj_s, adj_t):
        '''得到link对在两个网络中的邻接节点'''
        s_idx, t_idx = link
        s_adj, t_adj = adj_s.copy(), adj_t.copy()
        s_n, t_n = set(range(adj_s.shape[0])), \
                   set(range(adj_t.shape[0]))
        # 选出不属于link对的节点
        # s_i, t_i = list(s_n - set(s_idx.numpy())), \
        #            list(t_n - set(t_idx.numpy()))
        s_i, t_i = list(s_n - set(s_idx)), \
                   list(t_n - set(t_idx))
        s_adj[s_i, :] = 0
        s_adj[:, s_i] = 0
        t_adj[t_i, :] = 0
        t_adj[:, t_i] = 0

        s_pair = list(zip(*[i.astype(np.int64) for i in s_adj.nonzero()]))
        t_pair = list(zip(*[i.astype(np.int64) for i in t_adj.nonzero()]))
        # s_pair = [torch.from_numpy(i) for i in s_adj.nonzero()]
        # t_pair = [torch.from_numpy(i) for i in t_adj.nonzero()]
        return s_pair, t_pair