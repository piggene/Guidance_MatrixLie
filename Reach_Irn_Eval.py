#!/usr/bin/env python3
"""
reach_irn_live_probe.py — Live probe loop to compare IK feasibility vs model prediction.

What it does
------------
1) Randomly samples grasp poses (position + 6D orientation) within bounds.
2) Checks **IK feasibility** using your exact IKEval pipeline (ground truth: 0=reac
   hable, 1=unreachable).
3) Evaluates your **trained model** on the same pose and classifies with a thresho
   ld.
4) Repeats **forever** until Ctrl-C. Prints running confusion matrix and metrics.

Notes
-----
- Labels keep your convention: y=1 means **UNREACHABLE** (positive class), y=0 me
  ans reachable.
- By default, this script assumes the checkpoint was trained with the original IR
  N that outputs **sigmoid probabilities**. If your model outputs logits, pass `--
  model_logits`.

Examples
--------
# Default: one batch of 32 poses per tick, fast metrics refresh
python reach_irn_live_probe.py \
  --ckpt checkpoints/reach_irn.pt \
  --panda_xml franka_emika_panda/panda.xml \
  --batch_per_tick 32 --threshold 0.50

# If your checkpoint is logits (no sigmoid in forward):
python reach_irn_live_probe.py --ckpt checkpoints/reach_irn.pt --model_logits

# Probe near the floor (harder IK) and print each mismatch
python reach_irn_live_probe.py \
  --ckpt checkpoints/reach_irn.pt \
  --val_z 0.03 0.06 --print_mismatches

Quit with Ctrl-C at any time.
"""

import os, math, time, argparse, numpy as np
import torch, torch.nn as nn
import mujoco

# ---------------- Constants (match your sim) ----------------
TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)
MAX_DQ_PER_STEP = 0.06
LAM_BASE = 1e-3

# ---------------- Scene & robot utils ----------------
def build_scene(panda_xml_path: str):
    base_dir = os.path.dirname(os.path.abspath(panda_xml_path))
    xml = f"""
<mujoco>
  <compiler angle="radian" meshdir="."/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <include file="{os.path.basename(panda_xml_path)}"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>
  </worldbody>
</mujoco>
""".strip()
    tmp = os.path.join(base_dir, "_reach_scene.xml")
    os.makedirs(base_dir, exist_ok=True)
    with open(tmp, "w") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_path(tmp)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def find_arm_joints_and_ee(m):
    def _jid(nm):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        return j if j >= 0 else None
    for names in [[f"panda_joint{i}" for i in range(1,8)],
                  [f"joint{i}" for i in range(1,8)],
                  [f"fr3_joint{i}" for i in range(1,8)]]:
        ids = [j for j in (_jid(n) for n in names) if j is not None]
        if len(ids) == 7:
            arm = ids
            break
    else:
        hinges = [j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        assert len(hinges)>=7, "Could not find 7 hinge joints"
        arm = hinges[:7]
    for nm in ["panda_hand","panda_link7","hand","link7"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            ee = bid
            break
    else:
        ee = m.nbody-1
    return arm, ee


def get_arm_qpos(m, d, arm_joint_ids):
    return np.array([d.qpos[m.jnt_qposadr[j]] for j in arm_joint_ids], dtype=float)


def get_joint_mids(m, d, arm_joint_ids):
    mids=[]
    for j in arm_joint_ids:
        if m.jnt_limited[j]:
            lo,hi = m.jnt_range[j]; mids.append(0.5*(lo+hi))
        else:
            mids.append(d.qpos[m.jnt_qposadr[j]])
    return np.array(mids, dtype=float)


def dls(J, err, lam=3e-3):
    JT = J.T
    return JT @ np.linalg.solve(J@JT + (lam**2)*np.eye(J.shape[0]), err)


def ik_step_to_pose(m, d, ee_body_id, arm_joint_ids, p_target, q_target_wxyz,
                    step_gain=0.5, lam=LAM_BASE, bias_gain=0.0):
    R_hand = d.xmat[ee_body_id].reshape(3,3).copy()
    p_hand = d.xpos[ee_body_id].copy()
    p_cur = p_hand + R_hand @ TCP_OFFSET_HAND_LOCAL
    R_cur = R_hand
    q_cur = np.empty(4); mujoco.mju_mat2Quat(q_cur, R_cur.flatten())
    pos_err = p_target - p_cur

    q_conj = q_cur.copy(); q_conj[1:] *= -1
    q_err = np.empty(4); mujoco.mju_mulQuat(q_err, q_target_wxyz, q_conj)
    w,x,y,z = q_err; v = np.array([x,y,z], float); nv = np.linalg.norm(v)
    if nv < 1e-12:
        rot_err = np.zeros(3)
    else:
        angle = 2.0*np.arctan2(nv, max(w,1e-12)); rot_err = (angle/nv)*v

    W = np.diag([1,1,1,0.25,0.25,0.25])
    err6 = np.concatenate([pos_err, rot_err]); errw = W @ err6

    jacp = np.zeros((3,m.nv)); jacr = np.zeros((3,m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, p_cur, ee_body_id)
    J = np.vstack([jacp, jacr]); Jw = W @ J
    dof_addrs = np.array([m.jnt_dofadr[j] for j in arm_joint_ids], int)
    J_arm = Jw[:, dof_addrs]
    svals = np.linalg.svd(J_arm, compute_uv=False)
    lam_eff = lam * (10.0 if svals.min() < 1e-3 else 1.0)

    dq_task = dls(J_arm, errw, lam=lam_eff) * step_gain
    q_now = get_arm_qpos(m, d, arm_joint_ids)
    q_mid = get_joint_mids(m, d, arm_joint_ids)
    dq_arm = dq_task + bias_gain * (q_mid - q_now)
    dq_arm = np.clip(dq_arm, -MAX_DQ_PER_STEP, MAX_DQ_PER_STEP)
    for j, dq in zip(arm_joint_ids, dq_arm):
        adr = m.jnt_qposadr[j]; d.qpos[adr] += dq
    for j in arm_joint_ids:
        if m.jnt_limited[j]:
            lo,hi = m.jnt_range[j]; adr = m.jnt_qposadr[j]
            d.qpos[adr] = np.clip(d.qpos[adr], lo, hi)
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

# ---------------- Sampling & features ----------------

def _project_so3(R):
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def rot_to_6d(R):
    return np.concatenate([R[:,0], R[:,1]], 0)


def T_to_feat(T):
    return np.asarray(np.concatenate([T[:3,3], rot_to_6d(T[:3,:3])],0), dtype=np.float32)


def sample_pose(bounds, rng, z_floor=0.03):
    x = rng.uniform(*bounds['x'])
    y = rng.uniform(*bounds['y'])
    z = max(rng.uniform(*bounds['z']), z_floor)
    axis = rng.normal(size=3); axis /= (np.linalg.norm(axis)+1e-9)
    ang = rng.uniform(-math.pi, math.pi)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3)+math.sin(ang)*K+(1-math.cos(ang))*(K@K)
    R = _project_so3(R)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=[x,y,z]
    return T

# ---------------- IK labeler ----------------
class IKEval:
    def __init__(self, m,d,ee,arm,pos_tol=7e-4,rot_tol=6e-3, max_iters=800):
        self.m,self.d,self.ee,self.arm = m,d,ee,arm
        self.pos_tol,self.rot_tol = pos_tol,rot_tol
        self.max_iters=max_iters
    def plan(self, T, step_gain=0.55, lam=LAM_BASE, z_floor=0.03):
        R = T[:3,:3]; p = T[:3,3].copy(); p[2]=max(p[2], z_floor)
        q = np.empty(4); mujoco.mju_mat2Quat(q, R.flatten())
        di = mujoco.MjData(self.m); di.qpos[:]=self.d.qpos; di.qvel[:]=self.d.qvel
        mujoco.mj_forward(self.m, di)
        for _ in range(self.max_iters):
            pe,re = ik_step_to_pose(self.m, di, self.ee, self.arm, p, q, step_gain=step_gain, lam=lam)
            if pe < self.pos_tol and re < self.rot_tol:
                qg = get_arm_qpos(self.m, di, self.arm); return True, qg
        return False, get_arm_qpos(self.m, self.d, self.arm)
    def short_traj_safe(self, q_goal, steps=24):
        q0 = get_arm_qpos(self.m, self.d, self.arm)
        for s in np.linspace(0,1,steps):
            q = (1-s)*q0 + s*q_goal
            di = mujoco.MjData(self.m); di.qpos[:] = self.d.qpos
            for j,qj in zip(self.arm, q):
                adr = self.m.jnt_qposadr[j]
                if self.m.jnt_limited[j]:
                    lo,hi = self.m.jnt_range[j]
                    if qj < lo-1e-4 or qj > hi+1e-4: return False
                    di.qpos[adr] = np.clip(qj, lo, hi)
                else:
                    di.qpos[adr] = qj
            mujoco.mj_forward(self.m, di)
        return True
    def label(self, T):
        ok,qg = self.plan(T)
        return 0 if (ok and self.short_traj_safe(qg)) else 1

# ---------------- Model (prob or logits) ----------------
class IRNProb(nn.Module):
    """Matches your original training script: returns **probability** via sigmoid in forward."""
    def __init__(self, h=128, depth=3, drop=0.1):
        super().__init__()
        layers=[nn.Linear(9,h), nn.ReLU(True)]
        for _ in range(depth-1):
            layers += [nn.Linear(h,h), nn.ReLU(True), nn.Dropout(drop)]
        self.f = nn.Sequential(*layers); self.out = nn.Linear(h,1)
    def forward(self,x):
        return torch.sigmoid(self.out(self.f(x))).squeeze(-1)

class IRNLogits(nn.Module):
    """Logits model (use --model_logits if your checkpoint was trained without sigmoid)."""
    def __init__(self, h=128, depth=3, drop=0.1):
        super().__init__()
        layers=[nn.Linear(9,h), nn.ReLU(True)]
        for _ in range(depth-1):
            layers += [nn.Linear(h,h), nn.ReLU(True), nn.Dropout(drop)]
        self.f = nn.Sequential(*layers); self.out = nn.Linear(h,1)
    def forward(self,x):
        return self.out(self.f(x)).squeeze(-1)

# ---------------- Main loop ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='checkpoints/reach_irn.pt')
    ap.add_argument('--panda_xml', type=str, default='franka_emika_panda/panda.xml')
    ap.add_argument('--device', type=str, default='cuda:1')
    ap.add_argument('--seed', type=int, default=123)

    # Bounds for random sampling (val_* overrides if set)
    ap.add_argument('--x', type=float, nargs=2, default=[0.2,0.9])
    ap.add_argument('--y', type=float, nargs=2, default=[0.2,0.9])
    ap.add_argument('--z', type=float, nargs=2, default=[0.05,0.9])
    ap.add_argument('--val_x', type=float, nargs=2, default=None)
    ap.add_argument('--val_y', type=float, nargs=2, default=None)
    ap.add_argument('--val_z', type=float, nargs=2, default=None)

    # IK settings
    ap.add_argument('--max_ik_iters', type=int, default=800)
    ap.add_argument('--pos_tol', type=float, default=7e-4)
    ap.add_argument('--rot_tol', type=float, default=6e-3)

    # Model settings
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--threshold', type=float, default=0.50)
    ap.add_argument('--model_logits', action='store_true', help='Use if ckpt outputs logits (no sigmoid in forward).')

    # Loop behavior
    ap.add_argument('--batch_per_tick', type=int, default=32, help='How many random poses per iteration.')
    ap.add_argument('--sleep', type=float, default=0.0, help='Seconds to sleep between ticks (0 = as fast as possible).')
    ap.add_argument('--print_mismatches', action='store_true', help='Print details for FP/FN samples.')

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bounds = {
        'x': tuple(args.val_x) if args.val_x else tuple(args.x),
        'y': tuple(args.val_y) if args.val_y else tuple(args.y),
        'z': tuple(args.val_z) if args.val_z else tuple(args.z),
    }

    # Build scene & labeler
    m,d = build_scene(args.panda_xml)
    arm,ee = find_arm_joints_and_ee(m)
    home = np.array([0.0,-0.6,0.0,-2.2,0.0,2.2,0.8], float)
    for j,q in zip(arm, home):
        d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_forward(m,d)

    ike = IKEval(m,d,ee,arm, pos_tol=args.pos_tol, rot_tol=args.rot_tol, max_iters=args.max_ik_iters)

    # Device & model
    dev = torch.device(args.device if (args.device=="cpu" or ("cuda" in args.device and torch.cuda.is_available())) else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.model_logits:
        model = IRNLogits(h=args.hidden, depth=args.depth, drop=args.drop).to(dev)
    else:
        model = IRNProb(h=args.hidden, depth=args.depth, drop=args.drop).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    # Running stats
    tp=tn=fp=fn=0
    n_total=0
    t0=time.time()

    print("[probe] starting... press Ctrl-C to stop. Threshold=%.2f  bounds=%s" % (args.threshold, bounds))

    try:
        while True:
            # sample a batch of poses
            feats=[]; Ts=[]; gts=[]
            for _ in range(args.batch_per_tick):
                T = sample_pose(bounds, rng, z_floor=bounds['z'][0])
                y = ike.label(T)  # 0=reachable, 1=unreachable
                x = T_to_feat(T)
                Ts.append(T); gts.append(y); feats.append(x)

            xb = torch.from_numpy(np.stack(feats).astype(np.float32)).to(dev)
            with torch.no_grad():
                out = model(xb)
                if args.model_logits:
                    prob = torch.sigmoid(out).cpu().numpy()
                else:
                    prob = out.detach().cpu().numpy()
            pred = (prob >= args.threshold).astype(np.int32)

            # update confusion counts
            gts_arr = np.asarray(gts, dtype=np.int32)
            tp += int(((pred==1)&(gts_arr==1)).sum())
            tn += int(((pred==0)&(gts_arr==0)).sum())
            fp += int(((pred==1)&(gts_arr==0)).sum())
            fn += int(((pred==0)&(gts_arr==1)).sum())
            n_total += len(gts)

            # optionally print mismatches
            if args.print_mismatches:
                mism_idx = np.where(pred != gts_arr)[0]
                for i in mism_idx[:10]:  # cap prints per tick
                    T = Ts[i]; p = prob[i]; gt = gts_arr[i]; pr = pred[i]
                    pos = T[:3,3]
                    print(f"[mismatch] gt={gt} pred={pr} prob={p:.3f}  pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

            # metrics
            acc = (tp+tn)/max(n_total,1)
            prec = tp/max(tp+fp,1) if (tp+fp)>0 else 0.0
            rec = tp/max(tp+fn,1) if (tp+fn)>0 else 0.0
            tnr = tn/max(tn+fp,1) if (tn+fp)>0 else 0.0
            f1 = (2*prec*rec)/max(prec+rec,1e-9) if (prec+rec)>0 else 0.0
            bal_acc = 0.5*(rec+tnr)
            dt = time.time()-t0
            rate = n_total/max(dt,1e-9)
            print(f"[tick] N={n_total} | acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} spec={tnr:.4f} bal_acc={bal_acc:.4f} | tp={tp} tn={tn} fp={fp} fn={fn} | {rate:.1f} samp/s")

            if args.sleep>0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n[probe] stopped by user.")
        print(f"Final: N={n_total} | acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} spec={tnr:.4f} bal_acc={bal_acc:.4f} | tp={tp} tn={tn} fp={fp} fn={fn}")

if __name__ == '__main__':
    main()
