import numpy as np
import torch


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    # 计算两组特征向量之间的距离（默认为欧式距离），并返回他们之间的距离
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    # 返回的p2p[i]表示feat_y的第i个点跟feat_x中的第p2p[i]个点最近
    p2p = dist.argmin(dim=dim)
    return p2p


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    # 通过C12将evecs_x映射到y然后计算点到点的映射
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)


# 将点到点映射转换functional map
def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    # torch.linalg.lstsq(Y,X).solution
    # 找到一个解 x，使得方程组的残差 ||Ax - b|| 最小
    # 返回的解 C21 是使得 Ax ≈ B 的近似解
    # 注：这里的p2p是point-to-point map (shape x -> shape y)，
    # nn_query中的p2p是point-to-point map (shape y -> shape x).
    # evecs_y[p2p, :]就是让evecs_y重新排列，使得重新排列后每个点与evecs_x距离最近的点对齐
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21


# 使用ZoomOut方法细化点到点映射
def refine_pointmap_zoomout(p2p, evecs_x, evecs_y, k_start, step=1):
    """
    ZoomOut to refine a point-to-point map
    Args:
        p2p: point-to-point map: shape x -> shape y. [Vx]
        evecs_x: eigenvectors of shape x. [Vx, K]
        evecs_y: eigenvectors of shape y. [Vy, K]
        k_start (int): number of eigenvectors to start
        step (int, optional): step size. Default 1.
    """
    k_end = evecs_x.shape[1]
    inds = np.arange(k_start, k_end + step, step)

    p2p_refined = p2p
    for i in inds:
        C21_refined = pointmap2fmap(p2p_refined, evecs_x[:, :i], evecs_y[:, :i])
        p2p_refined = fmap2pointmap(C21_refined, evecs_y[:, :i], evecs_x[:, :i])

    return p2p_refined
