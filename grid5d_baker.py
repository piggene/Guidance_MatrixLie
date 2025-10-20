#!/usr/bin/env python3
# grid5d_baker_robot_only.py
#
# Build a strict 5D reachability grid over (x,y,z,yaw,pitch) for the Franka TCP.
# A cell is 1 only if IK solves *for all tested roll angles* at that pose.
# No object or collision checks — feasibility == IK solvable to tolerance.

import os, time, argparse
import numpy as np
import torch
import mujoco

# ======================== Shared settings (match your demo) ==================

# TCP offset in hand local frame (center between fingertips)
TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)

# IK knobs (keep identical to your mujoco_grasp_demo.py)
MAX_DQ_PER_STEP = 0.06
BIAS_GAIN        = 0.0
LAM_BASE         = 1e-3

def set_seeds(seed=1):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

# ============================== Scene =======================================

def build_scene_with_panda_only(panda_xml_path: str):
    """
    Create a temporary scene XML in the same directory as panda_xml_path so that
    <include file="..."> resolves. Adds only a floor; no object.
    """
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

    import tempfile
    fd, tmp_path = tempfile.mkstemp(prefix="scene_", suffix=".xml", dir=base_dir)
    with os.fdopen(fd, "w") as f:
        f.write(scene_xml)

    m = mujoco.MjModel.from_xml_path(tmp_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, tmp_path

# ============================== IK utils ====================================

def dls(J, err, lam=3e-3):
    JT = J.T
    return JT @ np.linalg.solve(J @ JT + (lam**2) * np.eye(J.shape[0]), err)

def clamp_qpos_in_range(m, qpos, joint_ids):
    for j in joint_ids:
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            adr = m.jnt_qposadr[j]
            qpos[adr] = np.clip(qpos[adr], lo, hi)

def get_arm_qpos(m, d, arm_joint_ids):
    return np.array([d.qpos[m.jnt_qposadr[j]] for j in arm_joint_ids], dtype=float)

def get_joint_mids(m, d, arm_joint_ids):
    mids = []
    for j in arm_joint_ids:
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            mids.append(0.5*(lo+hi))
        else:
            mids.append(d.qpos[m.jnt_qposadr[j]])
    return np.array(mids, dtype=float)

def ik_step_to_pose(m, d, ee_body_id, arm_joint_ids, p_target, q_target_wxyz,
                    step_gain=0.5, lam=LAM_BASE, bias_gain=BIAS_GAIN):
    # current HAND pose
    R_hand = d.xmat[ee_body_id].reshape(3, 3).copy()
    p_hand = d.xpos[ee_body_id].copy()

    # TCP (hand + offset)
    p_cur = p_hand + R_hand @ TCP_OFFSET_HAND_LOCAL
    R_cur = R_hand
    q_cur = np.empty(4); mujoco.mju_mat2Quat(q_cur, R_cur.flatten())

    # pos error
    pos_err = p_target - p_cur

    # ori error: target * conj(cur) -> angle-axis
    q_conj = q_cur.copy(); q_conj[1:] *= -1
    q_err = np.empty(4); mujoco.mju_mulQuat(q_err, q_target_wxyz, q_conj)
    w, x, y, z = q_err
    v = np.array([x, y, z], dtype=float)
    nv = np.linalg.norm(v)
    rot_err = np.zeros(3) if nv < 1e-12 else (2.0 * np.arctan2(nv, max(w, 1e-12)) / nv) * v

    # weight orientation a bit lower
    W = np.diag([1, 1, 1, 0.25, 0.25, 0.25])
    err6 = np.concatenate([pos_err, rot_err])
    errw = W @ err6

    # Jacobian at TCP point
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, p_cur, ee_body_id)
    J = np.vstack([jacp, jacr]); Jw = W @ J

    # restrict to 7 arm joints
    dof_addrs = np.array([m.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=int)
    J_arm = Jw[:, dof_addrs]

    # adaptive damping
    svals = np.linalg.svd(J_arm, compute_uv=False)
    lam_eff = lam * (10.0 if svals.min() < 1e-3 else 1.0)

    dq_task = dls(J_arm, errw, lam=lam_eff) * step_gain
    q_now = get_arm_qpos(m, d, arm_joint_ids)
    q_mid = get_joint_mids(m, d, arm_joint_ids)
    dq_bias = bias_gain * (q_mid - q_now)
    dq_arm = np.clip(dq_task + dq_bias, -MAX_DQ_PER_STEP, MAX_DQ_PER_STEP)

    # apply to qpos
    for j, dq in zip(arm_joint_ids, dq_arm):
        adr = m.jnt_qposadr[j]; d.qpos[adr] += dq

    clamp_qpos_in_range(m, d.qpos, arm_joint_ids)
    mujoco.mj_forward(m, d)

    return float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

def R_from_ypr(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.zeros((3,3), dtype=float)
    R[0,0] =  cy*cp*cr - sy*sr;  R[0,1] = -cy*cp*sr - sy*cr; R[0,2] =  cy*sp
    R[1,0] =  sy*cp*cr + cy*sr;  R[1,1] = -sy*cp*sr + cy*cr; R[1,2] =  sy*sp
    R[2,0] =        -sp*cr;      R[2,1] =         sp*sr;     R[2,2] =  cp
    return R

def check_feasible_pose(m, d, ee_body_id, arm_joint_ids, T_target: np.ndarray,
                        iters: int = 220, restarts: int = 2,
                        step_gain: float = 0.55, lam: float = LAM_BASE,
                        tol_pos: float = 7e-4, tol_rot: float = 6e-3) -> bool:
    """
    True iff IK reaches target within tolerances (no collision logic).
    """
    R = T_target[:3, :3]; p = T_target[:3, 3]
    q_target = np.empty(4, dtype=np.float64); mujoco.mju_mat2Quat(q_target, R.reshape(-1))

    # turn OFF contacts (not used) and actuation while solving
    backup_flags = int(m.opt.disableflags)
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    try:
        qpos_backup = d.qpos.copy()
        solved = False
        for r in range(restarts):
            # jittered home
            home_q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 2.2, 0.8])
            jitter = np.random.uniform(-0.15, 0.15, size=7)
            for j,q in zip(arm_joint_ids, home_q + jitter):
                d.qpos[m.jnt_qposadr[j]] = q
            mujoco.mj_forward(m, d)

            pe = re = 0.0
            for k in range(iters):
                sg = step_gain * (0.6 if k > 120 else 0.8 if k > 60 else 1.0)
                pe, re = ik_step_to_pose(m, d, ee_body_id, arm_joint_ids, p, q_target,
                                         step_gain=sg, lam=lam, bias_gain=BIAS_GAIN)
                if pe < tol_pos and re < tol_rot:
                    solved = True; break
            if solved: break

        # revert state
        d.qpos[:] = qpos_backup; mujoco.mj_forward(m, d)
        return solved
    finally:
        m.opt.disableflags = backup_flags
        mujoco.mj_forward(m, d)

# ============================== Baking ======================================

@torch.no_grad()
def bake_grid5d_robot_only(panda_xml: str,
                           # spatial grid
                           origin_xyz = (-0.2, -0.6, 0.0),
                           size_xyz   = ( 1.2,  1.2, 1.2),
                           vox_xyz    = ( 0.02, 0.02, 0.02),
                           # orientation bins
                           yaw_bins: int = 24, pitch_bins: int = 13,
                           # strict roll robustness
                           roll_samples: int = 8,
                           tol_pos: float = 7e-4,
                           tol_rot: float = 6e-3,
                           restarts: int = 2,
                           seed: int = 1,
                           save_path: str = "grid5d_robot.pt"):
    """
    Builds a uint8 grid G[Dx,Dy,Dz,Ny,Np] where cell==1 iff IK solves for ALL
    tested roll angles at that (x,y,z,yaw,pitch). Robot-only scene.
    """
    set_seeds(seed)

    # scene
    m, d, _ = build_scene_with_panda_only(panda_xml)

    # ids
    def _body(nm): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    def _jid(nm):  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    ee_body = _body("panda_hand") if _body("panda_hand") >= 0 else _body("panda_link7")

    # 7 arm joints (heuristics)
    name_sets = [
        [f"panda_joint{i}" for i in range(1, 8)],
        [f"joint{i}" for i in range(1, 8)],
        [f"fr3_joint{i}" for i in range(1, 8)],
    ]
    arm_joint_ids = None
    for cand in name_sets:
        ids = [j for j in (_jid(nm) for nm in cand) if j >= 0]
        if len(ids) == 7:
            arm_joint_ids = ids; break
    if arm_joint_ids is None:
        hinge_ids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        assert len(hinge_ids) >= 7, "Could not find 7 hinge joints"
        arm_joint_ids = hinge_ids[:7]

    # grid geometry
    origin = np.asarray(origin_xyz, float)
    size   = np.asarray(size_xyz, float)
    vox    = np.asarray(vox_xyz, float)
    dims   = np.maximum(1, np.round(size / vox).astype(int))
    Dx, Dy, Dz = dims.tolist()

    # angle supports
    yaw_min, yaw_max = -np.pi, np.pi
    pit_min, pit_max = -0.5*np.pi, 0.5*np.pi
    yaw_vals = np.linspace(yaw_min, yaw_max, yaw_bins, endpoint=True)
    pit_vals = np.linspace(pit_min, pit_max, pitch_bins, endpoint=True)
    rolls    = np.linspace(-np.pi, np.pi, roll_samples, endpoint=False)

    G = np.zeros((Dx, Dy, Dz, yaw_bins, pitch_bins), dtype=np.uint8)

    total_cells = Dx * Dy * Dz * yaw_bins * pitch_bins
    done = 0; t0 = time.time(); last_print = t0

    for ix in range(Dx):
        for iy in range(Dy):
            for iz in range(Dz):
                p = origin + vox * np.array([ix, iy, iz], float)
                # keep above floor slightly (optional)
                if p[2] < 0.02: p[2] = 0.02
                for iyaw, yaw in enumerate(yaw_vals):
                    for ipit, pit in enumerate(pit_vals):
                        all_ok = True
                        for roll in rolls:
                            R = R_from_ypr(yaw, pit, roll)
                            T = np.eye(4); T[:3,:3] = R; T[:3,3] = p
                            ok = check_feasible_pose(
                                m, d, ee_body, arm_joint_ids, T,
                                tol_pos=tol_pos, tol_rot=tol_rot,
                                restarts=restarts
                            )
                            if not ok:
                                all_ok = False; break
                        G[ix,iy,iz,iyaw,ipit] = 1 if all_ok else 0
                        done += 1
                        if time.time() - last_print > 2.0:
                            pct = 100.0 * done / total_cells
                            elapsed = time.time() - t0
                            eta = elapsed / max(done,1) * (total_cells - done)
                            print(f"[grid5d IK-only] {pct:6.2f}%  ETA ~ {eta/60:.1f} min")
                            last_print = time.time()

    torch.save({
        "grid": torch.from_numpy(G),                # uint8 (Dx,Dy,Dz,Ny,Np)
        "origin": torch.tensor(origin, dtype=torch.float32),
        "voxel": torch.tensor(vox, dtype=torch.float32),
        "yaw_bins": yaw_bins, "pitch_bins": pitch_bins,
        "yaw_min": yaw_min, "yaw_max": yaw_max,
        "pitch_min": pit_min, "pitch_max": pit_max,
        # metadata for reproducibility
        "tcp_offset_hand_local": torch.tensor(TCP_OFFSET_HAND_LOCAL, dtype=torch.float32),
        "tol_pos": float(tol_pos), "tol_rot": float(tol_rot),
        "roll_samples": int(roll_samples),
    }, save_path)
    sz = G.size
    print(f"[grid5d IK-only] saved to {save_path}  shape={G.shape}  bytes≈{sz}B  time={(time.time()-t0)/60:.1f} min")

# ================================ CLI =======================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panda_xml", type=str, default="franka_emika_panda/panda.xml")

    # grid params
    ap.add_argument("--origin", type=float, nargs=3, default=[-0.2, -0.6, 0.0])
    ap.add_argument("--size",   type=float, nargs=3, default=[ 1.2,  1.2, 1.2])
    ap.add_argument("--voxel",  type=float, nargs=3, default=[ 0.02, 0.02, 0.02])

    ap.add_argument("--yaw_bins", type=int, default=24)
    ap.add_argument("--pitch_bins", type=int, default=13)
    ap.add_argument("--roll_samples", type=int, default=8)

    # IK & solver params
    ap.add_argument("--tol_pos", type=float, default=7e-4)
    ap.add_argument("--tol_rot", type=float, default=6e-3)
    ap.add_argument("--restarts", type=int, default=2)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="grid5d_robot.pt")
    args = ap.parse_args()

    bake_grid5d_robot_only(
        panda_xml=args.panda_xml,
        origin_xyz=tuple(args.origin),
        size_xyz=tuple(args.size),
        vox_xyz=tuple(args.voxel),
        yaw_bins=args.yaw_bins, pitch_bins=args.pitch_bins,
        roll_samples=args.roll_samples,
        tol_pos=args.tol_pos, tol_rot=args.tol_rot,
        restarts=args.restarts, seed=args.seed,
        save_path=args.out
    )

if __name__ == "__main__":
    main()
