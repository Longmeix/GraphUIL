import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import random
from utils.general import read_pickle
import numpy as np
import torch
import config as cfg
import argparse
from models.netEncode import NetEncode
from UIL.GraphUIL import GraphUIL


def seed_torch(seed=2021):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(ratio=0.8):
    return \
        read_pickle('./data/graph/adj_s.pkl'), \
        read_pickle('./data/graph/adj_t.pkl'), \
        read_pickle('./data/graph/embeds.pkl'), \
        read_pickle('./data/label/train_test_{:.1f}.pkl'.format(ratio))

    # 一二: 两个网络的邻接矩阵 s:(9734, 9734)  t:(9514, 9514)
    # 第三个: 有两个嵌入矩阵，分别与网络s和网络t对应 [[9732个*256维],[9514个*256]]
    # 第四个: 网络s和网络t间的真实身份链接，并且已经做了训练集，测试集的分割 [[train:7610个], [test: 1903个]]


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def match(adj_s, adj_t,
          embeds, link_train_test,
          k=30, option=''):
    # @emb: embeddings
    # @link_train_set: labeled link
    options = set(option.split(','))
    dir = 'data/MNE'
    if not os.path.exists(dir):
        os.mkdir(dir)
    # symmetric matrix
    adj_s = ((adj_s + adj_s.T) > 0) * 1.
    adj_t = ((adj_t + adj_t.T) > 0) * 1.

    if cfg.model == 'GraphUIL':
        gnns = [NetEncode(), NetEncode()]  # the original GraphUIL
    uil = GraphUIL(embeds, adj_s, adj_t, link_train_test, gnns, k)
    print(params_count(gnns[0]))
    uil.train()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='GraphUIL', type=str)
    parser.add_argument('--ratio', default=cfg.ratio, type=float)
    parser.add_argument('--k', default=cfg.k, type=int)
    parser.add_argument('--options', default='structure', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    # rand, system configure
    seed_torch()
    os.chdir('./')

    # 1. initial configuration
    args = get_args()
    print(args)
    cfg.init_args(args)
    print(f'alpha:{cfg.alpha}, beta:{cfg.beta}')

    modeldata_path = './data/{}'.format(cfg.model)
    if not os.path.exists(modeldata_path):
        os.mkdir(modeldata_path)
    # 2. load data
    adj_s, adj_t, embeds, link_train_test = load_data(cfg.ratio)
    # 3. match
    match(adj_s, adj_t, embeds, link_train_test, k=cfg.k, option=cfg.options)
