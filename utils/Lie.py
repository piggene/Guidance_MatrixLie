import torch
import numpy as np
from scipy.spatial.transform import Rotation


EPS = 1e-4


def is_SO3(R):
    test_0 = torch.allclose(R @ R.transpose(1, 2), torch.eye(3).repeat(len(R), 1, 1).to(R), atol=EPS)
    test_1 = torch.allclose(R.transpose(1, 2) @ R, torch.eye(3).repeat(len(R), 1, 1).to(R), atol=EPS)

    test = test_0 and test_1

    return test


def is_SE3(T):
    test_0 = is_SO3(T[:, :3, :3])
    test_1 = torch.equal(T[:, 3, :3], torch.zeros_like(T[:, 3, :3]))
    test_2 = torch.equal(T[:, 3, 3], torch.ones_like(T[:, 3, 3]))

    test = test_0 and test_1 and test_2

    return test


def inv_SO3(R):
    assert R.shape[1:] == (3, 3), f"inv_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"
    assert is_SO3(R), "inv_SO3: input must be SO(3) matrices"

    inv_R = R.transpose(1, 2)

    return inv_R


def inv_SE3(T):
    assert T.shape[1:] == (4, 4), f"inv_SE3: input must be of shape (N, 4, 4). Current shape: {tuple(T.shape)}"
    assert is_SE3(T), "inv_SE3: input must be SE(3) matrices"

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    inv_T = torch.eye(4).repeat(len(T), 1, 1).to(T)
    inv_T[:, :3, :3] = inv_SO3(R)
    inv_T[:, :3, 3] = - torch.einsum('nij,nj->ni', inv_SO3(R), p)

    return inv_T


def bracket_so3(w):
    # vector -> matrix
    if w.shape[1:] == (3,):
        zeros = w.new_zeros(len(w))

        out = torch.stack([
            torch.stack([zeros, -w[:, 2], w[:, 1]], dim=1),
            torch.stack([w[:, 2], zeros, -w[:, 0]], dim=1),
            torch.stack([-w[:, 1], w[:, 0], zeros], dim=1)
        ], dim=1)

    # matrix -> vector
    elif w.shape[1:] == (3, 3):
        out = torch.stack([w[:, 2, 1], w[:, 0, 2], w[:, 1, 0]], dim=1)

    else:
        raise f"bracket_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w.shape)}"

    return out


def bracket_se3(S):
    # vector -> matrix
    if S.shape[1:] == (6,):
        w_mat = bracket_so3(S[:, :3])

        out = torch.cat((
            torch.cat((w_mat, S[:, 3:].unsqueeze(2)), dim=2),
            S.new_zeros(len(S), 1, 4)
        ), dim=1)

    # matrix -> vector
    elif S.shape[1:] == (4, 4):
        w_vec = bracket_so3(S[:, :3, :3])

        out = torch.cat((w_vec, S[:, :3, 3]), dim=1)

    else:
        raise f"bracket_se: input must be of shape (N, 6) or (N, 4, 4). Current shape: {tuple(S.shape)}"

    return out


def log_SO3(R):
    # return logSO3(R)
    n = R.shape[0]
    assert R.shape == (n, 3, 3), f"log_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"
    assert is_SO3(R), "log_SO3: input must be SO(3) matrices"

    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    w_mat = torch.zeros_like(R)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS))

    is_regular = (tr_R + 1 > EPS)
    is_singular = (tr_R + 1 <= EPS)

    theta = theta.unsqueeze(1).unsqueeze(2)

    w_mat_regular = (1 / (2 * torch.sin(theta[is_regular]) + EPS)) * (R[is_regular] - R[is_regular].transpose(1, 2)) * theta[is_regular]

    w_mat_singular = (R[is_singular] - torch.eye(3).to(R)) / 2

    w_vec_singular = torch.sqrt(torch.diagonal(w_mat_singular, dim1=1, dim2=2) + 1)
    w_vec_singular[torch.isnan(w_vec_singular)] = 0

    w_1 = w_vec_singular[:, 0]
    w_2 = w_vec_singular[:, 1] * (torch.sign(w_mat_singular[:, 0, 1]) + (w_1 == 0))
    w_3 = w_vec_singular[:, 2] * torch.sign(4 * torch.sign(w_mat_singular[:, 0, 2]) + 2 * (w_1 == 0) * torch.sign(w_mat_singular[:, 1, 2]) + 1 * (w_1 == 0) * (w_2 == 0))

    w_vec_singular = torch.stack([w_1, w_2, w_3], dim=1)

    w_mat[is_regular] = w_mat_regular
    w_mat[is_singular] = bracket_so3(w_vec_singular) * torch.pi

    return w_mat


def log_SE3(T):
    assert T.shape[1:] == (4, 4), f"log_SE3: input must be of shape (N, 4, 4). Current shape: {tuple(T.shape)}"
    assert is_SE3(T), "log_SE3: input must be SE(3) matrices"

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS)).unsqueeze(1).unsqueeze(2)

    w_mat = log_SO3(R)
    w_mat_hat = w_mat / (theta + EPS)

    inv_G = torch.eye(3).repeat(len(T), 1, 1).to(T) - (theta / 2) * w_mat_hat + (1 - (theta / (2 * torch.tan(theta / 2) + EPS))) * w_mat_hat @ w_mat_hat

    S = torch.zeros_like(T)
    S[:, :3, :3] = w_mat
    S[:, :3, 3] = torch.einsum('nij,nj->ni', inv_G, p)

    return S


def exp_so3(w_vec):
    if w_vec.shape[1:] == (3, 3):
        w_vec = bracket_so3(w_vec)
    elif w_vec.shape[1:] != (3,):
        raise f"exp_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w_vec.shape)}"

    R = torch.eye(3).repeat(len(w_vec), 1, 1).to(w_vec)

    theta = w_vec.norm(dim=1)

    is_regular = theta > EPS

    w_vec_regular = w_vec[is_regular]
    theta_regular = theta[is_regular]

    theta_regular = theta_regular.unsqueeze(1)

    w_mat_hat_regular = bracket_so3(w_vec_regular / theta_regular)

    theta_regular = theta_regular.unsqueeze(2)

    R[is_regular] = torch.eye(3).repeat(len(w_vec_regular), 1, 1).to(w_vec_regular) + torch.sin(theta_regular) * w_mat_hat_regular + (1 - torch.cos(theta_regular)) * w_mat_hat_regular @ w_mat_hat_regular

    return R


def exp_se3(S):
    if S.shape[1:] == (4, 4):
        S = bracket_se3(S)
    elif S.shape[1:] != (6,):
        raise f"exp_se3: input must be of shape (N, 6) or (N, 4, 4). Current shape: {tuple(S.shape)}"

    w_vec = S[:, :3]
    p = S[:, 3:]

    T = torch.eye(4).repeat(len(S), 1, 1).to(S)

    theta = w_vec.norm(dim=1)

    is_regular = theta > EPS
    is_singular = theta <= EPS

    w_vec_regular = w_vec[is_regular]
    theta_regular = theta[is_regular]

    theta_regular = theta_regular.unsqueeze(1)

    w_mat_hat_regular = bracket_so3(w_vec_regular / theta_regular)

    theta_regular = theta_regular.unsqueeze(2)

    G = theta_regular * torch.eye(3).repeat(len(S), 1, 1).to(S) + (1 - torch.cos(theta_regular)) * w_mat_hat_regular + (theta_regular - torch.cos(theta_regular)) * w_mat_hat_regular @ w_mat_hat_regular

    T[is_regular, :3, :3] = exp_so3(w_vec_regular)
    T[is_regular, :3, 3] = torch.einsum('nij,nj->ni', G, p)

    T[is_singular, :3, :3] = torch.eye(3).repeat(is_singular.sum(), 1, 1)
    T[is_singular, :3, 3] = p

    return T


def large_adjoint(T):
    assert T.shape[1:] == (4, 4), f"large_adjoint: input must be of shape (N, 4, 4). Current shape: {tuple(T.shape)}"
    assert is_SE3(T), "large_adjoint: input must be SE(3) matrices"

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    large_adj = T.new_zeros(len(T), 6, 6)
    large_adj[:, :3, :3] = R
    large_adj[:, 3:, :3] = bracket_so3(p) @ R
    large_adj[:, 3:, 3:] = R

    return large_adj


def small_adjoint(S):
    if S.shape[1:] == (4, 4):
        w_mat = S[:, :3, :3]
        v_mat = bracket_so3(S[:, :3, 3])
    elif S.shape[1:] == (6,):
        w_mat = bracket_so3(S[:, :3])
        v_mat = bracket_so3(S[:, 3:])
    else:
        raise f"small_adj: input must be of shape (N, 6) or (N, 4, 4). Current shape: {tuple(S.shape)}"

    small_adj = S.new_zeros(len(S), 6, 6)
    small_adj[:, :3, :3] = w_mat
    small_adj[:, 3:, :3] = v_mat
    small_adj[:, 3:, 3:] = w_mat

    return small_adj


def Lie_bracket(u, v):
    if u.shape[1:] == (3,):
        u = bracket_so3(u)
    elif u.shape[1:] == (6,):
        u = bracket_se3(u)

    if v.shape[1:] == (3,):
        v = bracket_so3(v)
    elif v.shape[1:] == (6,):
        v = bracket_se3(v)

    return u @ v - v @ u


def is_quat(quat):
    test = torch.allclose(quat.norm(dim=1), quat.new_ones(len(quat)))

    return test


def super_fibonacci_spiral(num_Rs):
    phi = 1.414213562304880242096980    # sqrt(2)
    psi = 1.533751168755204288118041

    s = np.arange(num_Rs) + 1 / 2

    t = s / num_Rs
    d = 2 * np.pi * s

    r = np.sqrt(t)
    R = np.sqrt(1 - t)

    alpha = d / phi
    beta = d / psi

    quats = np.stack([r * np.sin(alpha), r * np.cos(alpha), R * np.sin(beta), R * np.cos(beta)], axis=1)

    Rs = Rotation.from_quat(quats).as_matrix()

    return Rs


def SE3_geodesic_dist(T_1, T_2):
    assert len(T_1) == len(T_2), f"SE3_geodesic_dist: inputs must have the same batch_size. Current shapes: T_1 - {tuple(T_1.shape)}, T_2 - {tuple(T_2.shape)}"
    assert is_SE3(T_1) and is_SE3(T_2), "SE3_geodesic_dist: inputs must be SE(3) matrices"

    R_1 = T_1[:, :3, :3]
    R_2 = T_2[:, :3, :3]
    p_1 = T_1[:, :3, 3]
    p_2 = T_2[:, :3, 3]

    delta_R = bracket_so3(log_SO3(torch.einsum('bij,bjk->bik', inv_SO3(R_1), R_2)))
    delta_p = p_1 - p_2

    dist = (delta_R ** 2 + delta_p ** 2).sum(1).sqrt()

    return dist


def get_fibonacci_sphere(num_points):
    points = []

    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points += [np.array([x, y, z])]

    points = np.stack(points)

    return points


#gene: sped up version for deployment!!

VALIDATE = False
EPS = 1e-4


def inv_SE3_fast(T):
    # (...,4,4) -> (...,4,4)
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = T.new_empty(T.shape)
    out[..., :3, :3] = Rt
    out[..., :3, 3]  = -(Rt @ p.unsqueeze(-1)).squeeze(-1)
    out[..., 3, :3]  = 0
    out[..., 3, 3]   = 1
    if VALIDATE:  # optional slow checks
        assert torch.allclose(out[..., :3, :3] @ R, torch.eye(3, device=T.device, dtype=T.dtype).expand_as(R), atol=1e-5)
    return out

def bracket_so3_fast(x):
    # vector -> matrix  or  matrix -> vector
    if x.shape[-2:] == (3, 3):
        # matrix -> vector (vee)
        return torch.stack([x[..., 2, 1], x[..., 0, 2], x[..., 1, 0]], dim=-1)
    else:
        # vector -> matrix (hat)
        wx, wy, wz = x.unbind(-1)
        zeros = torch.zeros_like(wx)
        m = torch.stack([
            torch.stack([zeros, -wz,   wy], dim=-1),
            torch.stack([wz,   zeros, -wx], dim=-1),
            torch.stack([-wy,  wx,    zeros], dim=-1),
        ], dim=-2)
        return m

def log_SO3_fast(R):
    # assumes reasonably valid rotations
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(((tr - 1) * 0.5).clamp(-1+EPS, 1-EPS))  # (...,)

    w = torch.zeros_like(R)
    theta_ = theta[..., None, None]  # broadcast
    reg = (tr + 1 > EPS)
    sing = ~reg

    if reg.any():
        Rr = R[reg]
        th = theta_[reg]
        w[reg] = (Rr - Rr.transpose(-1, -2)) * (th / (2*torch.sin(th) + EPS))

    if sing.any():
        Rs = R[sing]
        # for pi rotations; fallback
        w_s = (Rs - torch.eye(3, device=R.device, dtype=R.dtype)) * 0.5
        w_vec = torch.sqrt(w_s.diagonal(dim1=-2, dim2=-1) + 1)
        w_vec[torch.isnan(w_vec)] = 0
        w1 = w_vec[..., 0]
        w2 = w_vec[..., 1] * (torch.sign(w_s[..., 0, 1]) + (w1 == 0))
        w3 = w_vec[..., 2] * torch.sign(
            4*torch.sign(w_s[..., 0, 2]) + 2*(w1 == 0)*torch.sign(w_s[..., 1, 2]) + 1*(w1 == 0)*(w2 == 0)
        )
        w[sing] = bracket_so3_fast(torch.stack([w1, w2, w3], dim=-1)) * torch.pi
    return w  # skew-symmetric matrices

def exp_so3_fast(w_mat):
    # accepts (...,3,3) skew OR (...,3) vector; returns (...,3,3)
    if w_mat.shape[-2:] != (3,3):
        w_mat = bracket_so3_fast(w_mat)  # convert vector -> skew
    # vee to norms
    w_vec = bracket_so3_fast(w_mat)      # (...,3)
    theta = torch.linalg.norm(w_vec, dim=-1)  # (...)
    R = torch.eye(3, device=w_mat.device, dtype=w_mat.dtype).expand(*w_mat.shape[:-2], 3, 3).clone()

    reg = theta > EPS
    if reg.any():
        wr = w_mat[reg]
        th = theta[reg][..., None, None]
        I = torch.eye(3, device=wr.device, dtype=wr.dtype).expand_as(wr)
        R[reg] = I + torch.sin(th) * wr + (1 - torch.cos(th)) * (wr @ wr)
    return R

def log_SE3_fast(T):
    # returns matrix-form se(3) logs: (...,4,4) with 3x3 skew + 3x1 v
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    w = log_SO3_fast(R)  # (...,3,3)
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(((tr - 1) * 0.5).clamp(-1+EPS, 1-EPS))[..., None, None]  # (...,1,1)
    w_hat = w / (theta + EPS)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(*T.shape[:-2], 3, 3)
    invG = I - (theta/2) * w_hat + (1 - (theta / (2*torch.tan(theta/2) + EPS))) * (w_hat @ w_hat)

    S = T.new_zeros(T.shape)
    S[..., :3, :3] = w
    S[..., :3, 3]  = (invG @ p.unsqueeze(-1)).squeeze(-1)
    return S

def se3_log_to_vec_fast(T):
    # (...,4,4) -> (...,6)
    S = log_SE3_fast(T)
    w = bracket_so3_fast(S[..., :3, :3])
    v = S[..., :3, 3]
    return torch.cat([w, v], dim=-1)  # (...,6)
