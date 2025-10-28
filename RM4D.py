#!/usr/bin/env python3
# rm4d_true.py  — Build a TRUE RM4D (4D) map from random FK samples.
# Usage:
#   python rm4d_true.py --panda_xml franka_emika_panda/panda.xml --out rm4d_franka.pt
#
# The 4D grid axes are:
#   (b_x, b_y, z, alpha)
# where:
#   - (b_x,b_y): canonical base position (meters)
#   - z       : TCP height (meters)
#   - alpha   : tilt = angle between tool z-axis and world z (radians)
#
# This follows RM4D’s canonicalization: rotate around world-z by -phi to align
# the approach’s XY projection; then (b_x,b_y)=-(p'_x,p'_y), z=p_z (unchanged by z-rot),
# alpha = arccos(a_z). See RM4D Sec. III-A. 

import os, time, argparse, tempfile
import numpy as np
import torch
import mujoco

TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)

# --------------------- MuJoCo helpers ---------------------

def _build_scene_only_panda(panda_xml_path: str):
    base_dir = os.path.dirname(os.path.abspath(panda_xml_path))
    panda_xml_basename = os.path.basename(panda_xml_path)

    scene_xml = f"""
<mujoco>
  <compiler angle="radian" meshdir="."/>
  <option gravity="0 0 -9.81" integrator="RK4"/>
  <include file="{panda_xml_basename}"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>
  </worldbody>
</mujoco>
""".strip()

    fd, tmp_path = tempfile.mkstemp(prefix="scene_", suffix=".xml", dir=base_dir)
    with os.fdopen(fd, "w") as f:
        f.write(scene_xml)

    m = mujoco.MjModel.from_xml_path(tmp_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def _jid(nm):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    def _bid(nm):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,  nm)

    ee_body = _bid("panda_hand") if _bid("panda_hand") >= 0 else _bid("panda_link7")
    # find 7 hinge joints
    name_sets = [[f"panda_joint{i}" for i in range(1,8)],
                 [f"joint{i}" for i in range(1,8)],
                 [f"fr3_joint{i}" for i in range(1,8)]]
    arm_joint_ids = None
    for cand in name_sets:
        ids = [j for j in (_jid(n) for n in cand) if j >= 0]
        if len(ids) == 7:
            arm_joint_ids = ids; break
    if arm_joint_ids is None:
        hinge_ids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        assert len(hinge_ids) >= 7, "Could not find 7 hinge joints"
        arm_joint_ids = hinge_ids[:7]

    return m, d, ee_body, arm_joint_ids

def _sample_q(m, arm_joint_ids, rng):
    qs = []
    for j in arm_joint_ids:
        lo, hi = (-2.8973, 2.8973)
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
        qs.append(rng.uniform(lo, hi))
    return np.array(qs, dtype=float)

def _fk_tcp(m, d, ee_body, q, arm_joint_ids):
    for j, v in zip(arm_joint_ids, q):
        d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, d)
    R_hand = d.xmat[ee_body].reshape(3,3).copy()
    p_hand = d.xpos[ee_body].copy()
    p_tcp  = p_hand + R_hand @ TCP_OFFSET_HAND_LOCAL
    return p_tcp, R_hand

# ------------------------ RM4D mapping ------------------------

def pose_to_rm4d_keys(p_tcp: np.ndarray, R_hand: np.ndarray):
    """
    Inputs:
      p_tcp: (3,) TCP position in world
      R_hand: (3,3) rotation, tool z-axis = R_hand[:,2]
    Returns:
      b_x, b_y, z, alpha   (floats)
    """
    a = R_hand[:, 2]                             # approach (tool z-axis)
    alpha = float(np.arccos(np.clip(a[2], -1.0, 1.0)))  # angle to world z

    # rotate around world-z by -phi so a projects to xz half-plane
    phi = float(np.arctan2(a[1], a[0]))          # yaw of approach’s XY
    c, s = np.cos(-phi), np.sin(-phi)
    pxp = c * p_tcp[0] + s * p_tcp[1]
    pyp = -s * p_tcp[0] + c * p_tcp[1]
    # canonical base position is the negative of rotated TCP XY
    b_x = -pxp
    b_y = -pyp
    z   = float(p_tcp[2])                        # unchanged by z-rotation
    return b_x, b_y, z, alpha

# ------------------------ Builder ------------------------

def build_rm4d(panda_xml: str,
               # grid ranges (edit to your setup; defaults are sane for Franka on table)
               bx_range=(-0.92, 0.92), by_range=(-0.92, 0.92),   # canonical base X/Y [m]
               z_range=(0.00, 1.20),                         # TCP height [m]
               alpha_range=(0.0, np.pi),                     # tilt [rad]
               # resolution
               bx_vox=0.01, by_vox=0.01, z_vox=0.01, alpha_bins=100,
               # sampling
               n_samples=800_000, seed=1):
    m, d, ee_body, arm_joint_ids = _build_scene_only_panda(panda_xml)
    rng = np.random.default_rng(seed)

    # grid dims
    def N(lo, hi, vox): return max(1, int(np.round((hi - lo) / vox)))
    nbx = N(*bx_range, bx_vox)
    nby = N(*by_range, by_vox)
    nz  = N(*z_range,  z_vox)
    na  = max(1, int(alpha_bins))

    G = np.zeros((nbx, nby, nz, na), dtype=np.uint8)

    def bin1(v, lo, hi, n):
        t = (v - lo) / max(hi - lo, 1e-8) * (n - 1)
        return int(np.clip(np.round(t), 0, n - 1))

    t0 = time.time(); hits = 0
    for i in range(n_samples):
        q = _sample_q(m, arm_joint_ids, rng)
        p, R = _fk_tcp(m, d, ee_body, q, arm_joint_ids)
        bx, by, z, a = pose_to_rm4d_keys(p, R)

        if not (bx_range[0] <= bx <= bx_range[1] and
                by_range[0] <= by <= by_range[1] and
                z_range[0]  <= z  <= z_range[1]  and
                alpha_range[0] <= a <= alpha_range[1]):
            # out of the chosen map box
            pass
        else:
            ibx = bin1(bx, *bx_range, nbx)
            iby = bin1(by, *by_range, nby)
            iz  = bin1(z,  *z_range,  nz)
            ia  = bin1(a,  *alpha_range, na)
            if G[ibx,iby,iz,ia] == 0:
                G[ibx,iby,iz,ia] = 1; hits += 1

        if (i+1) % 20000 == 0:
            cov = 100.0 * hits / G.size
            rate = (i+1) / max(time.time()-t0, 1e-6)
            print(f"[RM4D] {i+1}/{n_samples}  nonzero={hits} ({cov:.4f}%)  ~{rate:.0f}/s")

    pack = {
        "grid": torch.from_numpy(G),
        "bx_range": bx_range, "by_range": by_range, "z_range": z_range, "alpha_range": alpha_range,
        "bx_vox": bx_vox, "by_vox": by_vox, "z_vox": z_vox, "alpha_bins": na,
        "tcp_offset_hand_local": torch.tensor(TCP_OFFSET_HAND_LOCAL, dtype=torch.float32),
    }
    return pack

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panda_xml", type=str, default="franka_emika_panda/panda.xml")
    ap.add_argument("--bx", type=float, nargs=2, default=[-1.0, 1.0])
    ap.add_argument("--by", type=float, nargs=2, default=[-1.0, 1.0])
    ap.add_argument("--z",  type=float, nargs=2, default=[0.00, 1.20])
    ap.add_argument("--alpha", type=float, nargs=2, default=[0.0, np.pi])
    ap.add_argument("--bx_vox", type=float, default=0.01)
    ap.add_argument("--by_vox", type=float, default=0.01)
    ap.add_argument("--z_vox",  type=float, default=0.01)
    ap.add_argument("--alpha_bins", type=int, default=100)
    ap.add_argument("--n_samples", type=int, default=60_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="rm4d_franka.pt")
    args = ap.parse_args()

    pack = build_rm4d(
        panda_xml=args.panda_xml,
        bx_range=tuple(args.bx), by_range=tuple(args.by),
        z_range=tuple(args.z), alpha_range=tuple(args.alpha),
        bx_vox=args.bx_vox, by_vox=args.by_vox, z_vox=args.z_vox, alpha_bins=args.alpha_bins,
        n_samples=args.n_samples, seed=args.seed
    )
    torch.save(pack, args.out)
    G = pack["grid"]; print(f"[RM4D] saved -> {args.out}  shape={tuple(G.shape)}  bytes≈{G.numel()}")

if __name__ == "__main__":
    main()
