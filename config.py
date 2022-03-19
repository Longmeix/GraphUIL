import os
import torch
from time import strftime, gmtime, localtime


ratio = 0.3
k = 20  # hit@k
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = 'GraphUIL'
options = 'structure'
epochs = 100

# ---- netEncode ----
dim_feature = 256
num_layer = 2  # 探讨过平滑，改变层数来观察GraphUIL指标变化, 最好为1
# lr = 5e-4
lr = 1e-4
weight_decay = 1e-3
batch_size = 2 ** 7

# ---- GraphUIL ----
neg_num = 5
supervised = True
msa_out_dim = 64
alpha = 10
beta = 1

# ----- other config -----
percent = 99

# ---- MLP ----
MLP_hid = 128
# save similarity when hit best
sims_path = 'data/{}/{}_sims.pkl'.format(model, model)

log = strftime("logs/{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(
        model, ''.join([s[0] for s in options.split()]), ratio
    ), gmtime())


def init_args(args):
    global device, ratio, model, options, epochs, log, sims_path
    global alpha, beta

    ratio = args.ratio
    device = args.device
    model = args.model
    options = args.options
    epochs = args.epochs
    k = args.k

    pickle_path = './logs'
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    log = strftime("logs/{}_{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(
        model, ''.join([s[0] for s in options.split()]), k, ratio
    ), localtime())

    sims_path = 'data/sims/{}_sims.pkl'.format(model)

