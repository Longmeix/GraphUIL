import numpy as np
from scipy.sparse import csr_matrix


# ------ numpy extension --------------------
# get percent useful mat
def clamp_mat(mat, percent, up=True):
    row = np.percentile(mat, percent, axis=1, keepdims=True)
    col = np.percentile(mat, percent, axis=0, keepdims=True)
    if up:
        return mat * (mat >= row) * (mat >= col)
    else:
        return mat * (mat <= row) * (mat <= col)

# 将连接对转换为邻接矩阵的形式
def pair2sparse(pairs, shape):
    s, t = list(zip(*pairs))
    mat = csr_matrix((np.ones(len(s)), (s, t)),
                     shape=shape)
    return mat


# ------ torch extension --------------------
def cosine(xs, ys, epsilon=1e-8):
    """
    Efficiently calculate the pairwise cosine similarities between two set of vectors.
    Args:
        xs: feature matrix, [N, dim]
        ys: feature matrix, [M, dim]
        epsilon: a small number to avoid dividing by zero
    Return:
        a [N, M] matrix of pairwise cosine similarities
    """
    mat = xs @ ys.t()
    x_norm = xs.norm(2, dim=1) + epsilon
    y_norm = ys.norm(2, dim=1) + epsilon
    x_diag = (1 / x_norm).diag()
    y_diag = (1 / y_norm).diag()
    return x_diag @ mat @ y_diag


def distance_pairwise(xs, ys, scale=1):
    # xs /= scale
    # ys /= scale
    x_sq = (xs ** 2).sum(dim=1, keepdim=True)
    y_sq = (ys ** 2).sum(dim=1, keepdim=True)
    xy = xs @ ys.t()
    return (x_sq - 2 * xy + y_sq.t()).sqrt()


# -------- data process ----------
