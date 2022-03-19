from itertools import chain

import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
import torch.optim as opt
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange
import config as cfg
import utils.extensions as npe
from models.loss import NSLoss
from models.base import DataWrapper
from models.base import UIL
from utils.general import write_pickle, read_pickle

net_key = ['s', 't']

class GraphUIL(UIL):
    def __init__(self, embeds, adj_s, adj_t, links, gnns, k):
        super(GraphUIL, self).__init__(links, k=k)
        # s,t网络的边属性字典 {'s':{pairs, weight, adj_mat}, 't':{}}
        # s,t网络的边属性字典 {'s':{pairs, weight, adj_mat}, 't':{}}
        self.edgeAttr = self.getEdgeAttr(adj_s, adj_t)
        shape = adj_s.shape[0], adj_t.shape[0]  # (9734,9514)

        if links is not None:
            link_train, link_test = links
            link_mat = npe.pair2sparse(link_train, shape)  # s-->t有link的矩阵位置为1 (9734,9514)
            self.link_attr_train = self.addEdgeAttr(link_mat)  # real links of train set
        # s,t两个网络中 每一列是一条表边(vi,vj) edge['s'].size (2, E)
        self.edge_idx = dict(s=self.adj2edge(adj_s), t=self.adj2edge(adj_t))

        # transfer embeds type to tensor
        if isinstance(embeds[0], torch.Tensor):
            self.embeds = dict(zip(net_key, (emb.to(cfg.device) for emb in embeds)))
        else:
            self.embeds = dict(zip(net_key, (torch.tensor(emb).float().to(cfg.device) for emb in embeds)))
        self.gnns = nn.ModuleList(gnns).to(cfg.device)
        dim = cfg.dim_feature + cfg.msa_out_dim * len(self.gnns[0].msas)
        # dim = cfg.dim_feature + 64 + 32
        self.common = nn.ModuleList([
            MLP(dim, dim),
            MLP(dim, dim)
        ]).to(cfg.device)

        # optimizer
        self.opt_embed = opt.Adam(
            chain(self.gnns.parameters()),
            lr=cfg.lr
        )

        self.opt_label = opt.Adam(
            chain(self.common.parameters(),
                  self.gnns.parameters()),
            lr=cfg.lr
        )

        self.opt_map = opt.Adam(
            chain(self.common.parameters()),
            lr=1e-4
        )

        # loss
        self.loss_intra = NSLoss(
            act=nn.Sigmoid(),
            loss=nn.MSELoss()
        )

        self.loss_label = NSLoss(
            sim=nn.CosineSimilarity(),
            act=nn.ReLU(),
            loss=nn.MSELoss()
        )
        self.mse = nn.MSELoss()

    @staticmethod
    def addEdgeAttr(adj_mat, exponent=3/4, percent=cfg.percent):
        """
       Given a similarity matrix, create weights for negative sampling and indices of similar users.
       Args:
           mat: similarity matrix
           exponent: a coefficient to downplay the similarities to create negative sampling weights, default: 3/4 (as suggested by word2vec)
           percent: percent of users to filter, range in [0, 100]
       Return:
           pairs: user pairs with high similairties
           weights: negative sampling weights
           mat: similarity matrix
       """
        if not isinstance(adj_mat, np.ndarray):
            adj_mat = adj_mat.toarray()
        weights = np.abs(adj_mat > 0).sum(axis=0) ** exponent
        clamp = npe.clamp_mat(adj_mat, percent)
        pairs = [i.tolist() for i in clamp.nonzero()]
        pairs = list(zip(*pairs))

        attr_keys = ['pairs', 'weights', 'adj_mat']
        attr_value = pairs, weights, csr_matrix(adj_mat)
        return dict(zip(attr_keys, attr_value))

    @staticmethod
    def adj2edge(adj_mat):
        # get edge(vi, vj) from adjacent matrix
        # size (2, E)
        return torch.tensor(list(zip(*adj_mat.nonzero()))).long().t().to(cfg.device)

    def getEdgeAttr(self, adj_s, adj_t):
        # get [pair, weight, adj_mat] of network s and t
        edgeAttr = {'s': {}, 't': {}}
        edgeAttr['s'] = self.addEdgeAttr(adj_s)
        edgeAttr['t'] = self.addEdgeAttr(adj_t)
        return edgeAttr

    def get_embeds(self, is_eval=False):
        # get node embedding of two networks
        # @is_test: when test, use MLP to embedding
        embed_s, embed_t = [self.gnns[0](self.embeds['s'], self.edge_idx['s']),
                            self.gnns[1](self.embeds['t'], self.edge_idx['t'])
                            ]
        if is_eval:
            embed_s = self.common[0](embed_s)
            embed_t = self.common[1](embed_t)
        return embed_s, embed_t

    def get_sims(self):
        f_s, f_t = self.get_embeds(is_eval=True)
        sims = self.sim_pairwise(f_s, f_t)
        return sims

    @staticmethod
    def get_pair_batch(pairs, batch_size):
        '''
        @:pairs: sample pairs[(vs, vt), ...], size=cfg.batch_size
        @:return: batch [(vs1, vs2, ...), (vt1, ...)]
        '''
        idx = RandomSampler(pairs, replacement=True,
                            num_samples=batch_size)
        pair_list = [pairs[i] for i in list(idx)]
        data = DataWrapper(pair_list)
        batches = DataLoader(
            data, batch_size=batch_size,
            shuffle=True)
        _, batch = next(enumerate(batches))  # only one batch in batches
        return batch

    def global_loss(self, net_name, pair, weight, adj_mat):
        idx_s, idx_t = pair
        gnn_i = 0 if net_name == 's' else 1
        embed_s = self.gnns[gnn_i](self.embeds[net_name], self.edge_idx[net_name])
        loss_batch = self.loss_intra(
            embed_s, embed_s,
            idx_s, idx_t,
            lambda x: x,
            lambda x: x,
            cfg.neg_num,
            weight,
            adj_mat
        )
        # loss_local = self.mse(embed_s[idx_s], embed_s[idx_t])
        return loss_batch

    def local_loss(self, net_name, pair, adj_mat):
        idx_s, idx_t = pair
        # edge_index = self.find_edge(np.append(idx_s, idx_t),
        #                             adj_mat)
        # edge_index = torch.tensor(edge_index).to(cfg.device)
        gnn_i = 0 if net_name == 's' else 1
        embed_s = self.gnns[gnn_i](
                        self.embeds[net_name], self.edge_idx[net_name]
                    )
        loss_batch = self.mse(embed_s[idx_s], embed_s[idx_t])
        return loss_batch

    def match_loss(self, pair, weight, adj_mat):
        idx_s, idx_t = pair

        self.opt_label.zero_grad()
        embed_s, embed_t = self.get_embeds()
        loss_batch = self.loss_label(
            embed_s, embed_t,
            idx_s, idx_t,
            self.common[0],
            self.common[1],
            cfg.neg_num,
            weight,
            adj_mat
        )
        return loss_batch

    def train(self):
        hit_best = .0

        pairs_s, wgt_s, adj_s = self.edgeAttr['s'].values()  # source net
        pairs_t, wgt_t, adj_t = self.edgeAttr['t'].values()  # target net
        pairs_l, wgt_l, adj_l = self.link_attr_train.values()  # link match

        # steps_per_epoch = len(pairs_l) // cfg.batch_size + 10  # can not larger than len(pairs_l)
        steps_per_epoch = 50
        self.log.info('Starting training...')
        for epoch in trange(1, cfg.epochs + 1):

            loss_epc = .0
            for step in range(steps_per_epoch):
                # ========= get batch data ===========
                # get batch data
                s_pair = self.get_pair_batch(pairs_s, cfg.batch_size)
                t_pair = self.get_pair_batch(pairs_t, cfg.batch_size)
                l_pair = self.get_pair_batch(pairs_l, cfg.batch_size)
                # ========= global loss ==========
                loss_g_s = self.global_loss('s', s_pair, wgt_s, adj_s)
                loss_g_t = self.global_loss('t', t_pair, wgt_t, adj_t)
                loss_global = loss_g_s + loss_g_t
                # ========== local loss ==========
                loss_l_s = self.local_loss('s', s_pair, adj_s)
                loss_l_t = self.local_loss('t', t_pair, adj_t)
                loss_local = loss_l_s + loss_l_t

                # ========== match loss ==========
                loss_match = self.match_loss(l_pair, wgt_l, adj_l)

                # sum all loss
                loss_batch = cfg.alpha * loss_global + loss_match + cfg.beta * loss_local
                loss_epc += loss_batch
                # update parameters of net
                self.opt_label.zero_grad()
                loss_batch.backward()
                self.opt_label.step()

            self.log.info('epoch {:03d} loss_all {:.4f}'.format(
                            epoch, loss_epc / steps_per_epoch))
            # ======= evaluate ==========
            # self.common.eval()
            self.gnns.eval()
            hit_p = self.eval_hit_p()
            if hit_p > hit_best:
                hit_best = hit_p
                f_s_map, f_t_map = self.get_embeds(is_eval=True)  # mapped embeds
                sims = self.sim_pairwise(f_s_map, f_t_map)
                write_pickle(sims, cfg.sims_path)

            print('hit: {:.4f}'.format(hit_best))

        sims = read_pickle(cfg.sims_path)
        self.report_hit(sims)
        self.rename_log('/hit{:.2f}_'.format(hit_best * 100).join(
            cfg.log.split('/')
        ))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out,
                 dim_hid=None, act=None):
        super(MLP, self).__init__()
        if act is None:
            act = nn.Tanh()
        if dim_hid is None:
            dim_hid = dim_in * 2   # 原文x_feature=64, dim_hid=128, 两倍关系
        # 2-layers
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            # nn.Dropout(0.5),
            act,
            nn.Linear(dim_hid, dim_out)
        )

    def forward(self, x):
        return self.model(x)



