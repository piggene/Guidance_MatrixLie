import math
import os
import torch

_pack = None
_device = "cpu"

# -------- Tunables (override via env) --------
# Neighborhood support (suppresses FPs)
_neigh = int(os.getenv("RM4D_NEIGH", "1"))                # radius r -> (2r+1)^4 neighborhood
_min_neigh_on = int(os.getenv("RM4D_MIN_NEIGH_ON", "6"))  # minimum occupied voxels in neighborhood
_min_neigh_frac = float(os.getenv("RM4D_MIN_NEIGH_FRAC", "0.06"))  # fraction of K, 0=off
_use_soft_score = int(os.getenv("RM4D_USE_SOFT_SCORE", "1"))       # 0/1 distance-weighted sum
_soft_thresh = float(os.getenv("RM4D_SOFT_THRESH", "6.5"))         # soft score threshold
_require_center = int(os.getenv("RM4D_REQUIRE_CENTER", "0"))       # 0/1 require center bin occupied

# Range gating tolerance (prevents edge aliasing)
_edge_eps = float(os.getenv("RM4D_EDGE_EPS", "0.49"))   # <= 0.5

# 4D multilinear (“tent”) interpolation (reduces FNs at bin edges)
_use_tent = int(os.getenv("RM4D_TENT", "1"))            # 0/1 enable tent score
_tent_thresh = float(os.getenv("RM4D_TENT_THRESH", "0.28"))  # score in [0,1]
# small relaxation band to avoid flicker right at the threshold (hysteresis-like, stateless)
_tent_relax = float(os.getenv("RM4D_TENT_RELAX", "0.04"))     # effective threshold = thresh - relax

# How to fuse tent vs. neighborhood verdicts: 'max' (OR), 'and' (AND), or 'tent_then_neigh'
# - 'max': good recall; 'and': strict; 'tent_then_neigh': tent must pass, then require support -> few FPs
_fuse = os.getenv("RM4D_FUSE", "max").lower()           # 'max' | 'and' | 'tent_then_neigh'

# Near-vertical stabilization epsilon
_phi_eps = float(os.getenv("RM4D_PHI_EPS", "1e-4"))


def rm4d_load(path: str, device="cpu"):
    """Load pack and move grid to device (kept as uint8 for memory)."""
    global _pack, _device
    _pack = torch.load(path, map_location=device)
    _device = device
    _pack["grid"] = _pack["grid"].to(device=device, dtype=torch.uint8)


@torch.no_grad()
def _pose_to_keys(T: torch.Tensor, eps: float = None):
    """
    T: (...,4,4) -> bx,by,z,alpha (flat tensors on _device)
    Stable φ near vertical: use pose yaw when a_xy ~ 0 to avoid canonization jitter.
    """
    if eps is None:
        eps = _phi_eps
    TD = T.to(_device).reshape(-1, 4, 4)
    p = TD[:, :3, 3]
    R = TD[:, :3, :3]
    a = R[:, :, 2]                              # tool z-axis

    alpha = torch.acos(torch.clamp(a[:, 2], -1.0, 1.0))
    r2 = a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1]
    phi_a = torch.atan2(a[:, 1], a[:, 0])
    phi_p = torch.atan2(p[:, 1], p[:, 0])
    phi = torch.where(r2 < eps * eps, phi_p, phi_a)

    c, s = torch.cos(-phi), torch.sin(-phi)
    pxp = c * p[:, 0] + s * p[:, 1]
    pyp = -s * p[:, 0] + c * p[:, 1]
    bx = -pxp
    by = -pyp
    z = p[:, 2]
    return bx, by, z, alpha


def _to_float_index(v, lo, hi, n):
    """
    Continuous bin coordinate t in [0, n-1] for multilinear interpolation.
    """
    # guard against degenerate range
    scale = (n - 1) / max(hi - lo, 1e-8)
    return (v - lo) * scale


def _multilinear_4d(Gu8, bx_t, by_t, z_t, a_t):
    """
    4D multilinear interpolation over the 16 corner bins (2 per axis).
    Inputs are continuous indices t in [0, n-1] for each axis.
    Returns score in [0,1].
    """
    G = Gu8.float()  # (nbx, nby, nz, na) on device
    nbx, nby, nz, na = G.shape

    def _two(i_t, n):
        i0 = torch.floor(i_t).to(torch.long)
        i1 = torch.clamp(i0 + 1, 0, n - 1)
        w1 = (i_t - i0.float()).clamp(0.0, 1.0)
        w0 = 1.0 - w1
        i0 = i0.clamp(0, n - 1)
        return i0, i1, w0, w1

    bx0, bx1, wx0, wx1 = _two(bx_t, nbx)
    by0, by1, wy0, wy1 = _two(by_t, nby)
    z0,  z1,  wz0,  wz1 = _two(z_t,  nz)
    a0,  a1,  wa0,  wa1 = _two(a_t,  na)

    # 16-corner sum: ((wx0/wx1) x (wy0/wy1) x (wz0/wz1) x (wa0/wa1))
    def W(x, y, z, a):
        return x * y * z * a

    s = (
        G[bx0, by0, z0, a0] * W(wx0, wy0, wz0, wa0) +
        G[bx1, by0, z0, a0] * W(wx1, wy0, wz0, wa0) +
        G[bx0, by1, z0, a0] * W(wx0, wy1, wz0, wa0) +
        G[bx1, by1, z0, a0] * W(wx1, wy1, wz0, wa0) +
        G[bx0, by0, z1, a0] * W(wx0, wy0, wz1, wa0) +
        G[bx1, by0, z1, a0] * W(wx1, wy0, wz1, wa0) +
        G[bx0, by1, z1, a0] * W(wx0, wy1, wz1, wa0) +
        G[bx1, by1, z1, a0] * W(wx1, wy1, wz1, wa0) +
        G[bx0, by0, z0, a1] * W(wx0, wy0, wz0, wa1) +
        G[bx1, by0, z0, a1] * W(wx1, wy0, wz0, wa1) +
        G[bx0, by1, z0, a1] * W(wx0, wy1, wz0, wa1) +
        G[bx1, by1, z0, a1] * W(wx1, wy1, wz0, wa1) +
        G[bx0, by0, z1, a1] * W(wx0, wy0, wz1, wa1) +
        G[bx1, by0, z1, a1] * W(wx1, wy0, wz1, wa1) +
        G[bx0, by1, z1, a1] * W(wx0, wy1, wz1, wa1) +
        G[bx1, by1, z1, a1] * W(wx1, wy1, wz1, wa1)
    )
    # since G is 0/1, s ∈ [0,1]
    return s


@torch.no_grad()
def energy_cost(x: torch.Tensor) -> torch.Tensor:
    """
    x: (...,4,4) SE(3) (world, meters) -> (...,1)  (0 feasible, 100 infeasible)

    Strategy:
      • Range-gate with half-bin margins.
      • Tent score: 4D multilinear interpolation over 16 corners (reduces bin-edge FNs).
      • Neighborhood support: (2r+1)^4 density (and optional distance-weighted soft score).
      • Fuse both with RM4D_FUSE policy: 'max' | 'and' | 'tent_then_neigh'.
    """
    assert _pack is not None, "Call rm4d_load('rm4d_franka.pt', device=...) first."
    lead = x.shape[:-2]
    bx, by, z, alpha = _pose_to_keys(x)  # (N,)
    G = _pack["grid"]                     # (nbx, nby, nz, na)

    nbx, nby, nz, na = G.shape
    bx_lo, bx_hi = _pack["bx_range"]
    by_lo, by_hi = _pack["by_range"]
    z_lo,  z_hi  = _pack["z_range"]
    a_lo,  a_hi  = _pack["alpha_range"]

    # half-bin sizes for tolerant range gating
    def half_bin(lo, hi, n):
        return 0.5 * (hi - lo) / max(n - 1, 1)

    eps_bx = half_bin(bx_lo, bx_hi, nbx) * _edge_eps
    eps_by = half_bin(by_lo, by_hi, nby) * _edge_eps
    eps_z  = half_bin(z_lo,  z_hi,  nz ) * _edge_eps
    eps_a  = half_bin(a_lo,  a_hi,  na ) * _edge_eps

    in_range = (
        (bx >= bx_lo - eps_bx) & (bx <= bx_hi + eps_bx) &
        (by >= by_lo - eps_by) & (by <= by_hi + eps_by) &
        (z  >=  z_lo - eps_z ) & (z  <=  z_hi + eps_z ) &
        (alpha >= a_lo - eps_a) & (alpha <= a_hi + eps_a)
    )

    ok = torch.zeros_like(in_range, dtype=torch.bool, device=in_range.device)
    if not in_range.any():
        return (~ok).to(x.dtype).reshape(lead + (1,)) * 100.0

    # indexes / continuous coords for points that passed range gate
    bx_ir, by_ir, z_ir, a_ir = bx[in_range], by[in_range], z[in_range], alpha[in_range]

    # ---- Tent score (multilinear over 16 bins) ----
    tent_ok = torch.zeros_like(bx_ir, dtype=torch.bool)
    if _use_tent:
        bx_t = _to_float_index(bx_ir, bx_lo, bx_hi, nbx)
        by_t = _to_float_index(by_ir, by_lo, by_hi, nby)
        z_t  = _to_float_index(z_ir,  z_lo,  z_hi,  nz)
        a_t  = _to_float_index(a_ir,  a_lo,  a_hi,  na)
        s = _multilinear_4d(G.to(_device), bx_t, by_t, z_t, a_t)  # [0,1]
        tent_ok = (s >= max(0.0, _tent_thresh - _tent_relax))
    else:
        tent_ok = torch.ones_like(bx_ir, dtype=torch.bool)

    # ---- Neighborhood support (density) ----
    neigh_ok = torch.ones_like(bx_ir, dtype=torch.bool)
    if _neigh > 0:
        # map to nearest bin (center) for neighbor indexing
        def _to_index(v, lo, hi, n):
            t = (v - lo) / max(hi - lo, 1e-8) * (n - 1)
            return torch.round(t).to(torch.long).clamp_(0, n - 1)
        bx_i = _to_index(bx_ir, bx_lo, bx_hi, nbx)
        by_i = _to_index(by_ir, by_lo, by_hi, nby)
        z_i  = _to_index(z_ir,  z_lo,  z_hi,  nz)
        a_i  = _to_index(a_ir,  a_lo,  a_hi,  na)

        rng = torch.arange(-_neigh, _neigh + 1, device=_device)
        dx, dy, dz, da = torch.meshgrid(rng, rng, rng, rng, indexing="ij")
        dx = dx.reshape(-1); dy = dy.reshape(-1); dz = dz.reshape(-1); da = da.reshape(-1)
        K = dx.numel()

        bx_n = (bx_i[:, None] + dx[None, :]).clamp(0, nbx - 1)
        by_n = (by_i[:, None] + dy[None, :]).clamp(0, nby - 1)
        z_n  = (z_i[:,  None] + dz[None, :]).clamp(0, nz  - 1)
        a_n  = (a_i[:,  None] + da[None, :]).clamp(0, na  - 1)

        neigh = (G[bx_n, by_n, z_n, a_n] > 0)  # (N, K)
        occ_count = neigh.sum(dim=1)

        need = _min_neigh_on
        if _min_neigh_frac > 0.0:
            need = max(need, int(math.ceil(_min_neigh_frac * K)))
        neigh_ok = (occ_count >= max(1, need))

        if _require_center:
            center_mask = (dx == 0) & (dy == 0) & (dz == 0) & (da == 0)
            center_ok = neigh[:, center_mask].squeeze(1)
            neigh_ok = neigh_ok & center_ok

        if _use_soft_score:
            # separable distance weights (closer bins contribute more)
            wx = 1.0 / (1.0 + dx.abs().float())
            wy = 1.0 / (1.0 + dy.abs().float())
            wz = 1.0 / (1.0 + dz.abs().float())
            wa = 1.0 / (1.0 + da.abs().float())
            W = wx * wy * wz * wa                     # (K,)
            soft = (neigh.float() * W[None, :]).sum(dim=1)
            neigh_ok = neigh_ok & (soft >= _soft_thresh)

    # ---- Fuse decisions ----
    if _fuse == "and":
        occ = tent_ok & neigh_ok
    elif _fuse == "tent_then_neigh":
        occ = tent_ok & neigh_ok
    else:  # 'max' (OR)
        occ = tent_ok | neigh_ok

    ok[in_range] = occ
    return (~ok).to(x.dtype).reshape(lead + (1,)) * 1000.0
