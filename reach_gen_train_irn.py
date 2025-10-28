#!/usr/bin/env python3
# reach_gen_train_irn.py
import os, math, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import mujoco

# ---------- constants (match your sim) ----------
TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)
MAX_DQ_PER_STEP = 0.06
LAM_BASE = 1e-3

# ---------- minimal scene (robot + floor) ----------
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
    with open(tmp, "w") as f: f.write(xml)
    m = mujoco.MjModel.from_xml_path(tmp)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d

# ---------- utility: find 7 arm joints; EE body ----------
def find_arm_joints_and_ee(m):
    def _jid(nm):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        return j if j >= 0 else None
    # common naming
    for names in [[f"panda_joint{i}" for i in range(1,8)],
                  [f"joint{i}" for i in range(1,8)],
                  [f"fr3_joint{i}" for i in range(1,8)]]:
        ids = [j for j in (_jid(n) for n in names) if j is not None]
        if len(ids) == 7: arm = ids; break
    else:
        hinges = [j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        assert len(hinges)>=7, "Could not find 7 hinge joints"
        arm = hinges[:7]
    # ee body
    for nm in ["panda_hand","panda_link7","hand","link7"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0: ee = bid; break
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
    if nv < 1e-12: rot_err = np.zeros(3)
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

    # clamp ranges
    for j in arm_joint_ids:
        if m.jnt_limited[j]:
            lo,hi = m.jnt_range[j]; adr = m.jnt_qposadr[j]
            d.qpos[adr] = np.clip(d.qpos[adr], lo, hi)
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

# ---------- sampling + features ----------
def rot_to_6d(R): return np.concatenate([R[:,0], R[:,1]], 0)
def T_to_feat(T):
    return np.asarray(np.concatenate([T[:3,3], rot_to_6d(T[:3,:3])],0), dtype=np.float32)

def sample_poses(N, bounds, yaw_only=True, z_floor=0.03, rng=None):
    rng = rng or np.random.default_rng()
    X = rng.uniform(*bounds['x'], N)
    Y = rng.uniform(*bounds['y'], N)
    Z = rng.uniform(*bounds['z'], N)
    Ts=[]
    for x,y,z in zip(X,Y,Z):
        z = max(z, z_floor)
        if yaw_only:
            yaw = rng.uniform(-math.pi, math.pi); cy,sy=math.cos(yaw),math.sin(yaw)
            R = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],float)
        else:
            axis = rng.normal(size=3); axis/= (np.linalg.norm(axis)+1e-9)
            ang = rng.uniform(-math.pi, math.pi)
            K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            R = np.eye(3)+math.sin(ang)*K+(1-math.cos(ang))*(K@K)
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=[x,y,z]; Ts.append(T)
    return Ts

# ---------- labeler ----------
class IKEval:
    def __init__(self, m,d,ee,arm,pos_tol=7e-4,rot_tol=6e-3):
        self.m,self.d,self.ee,self.arm = m,d,ee,arm
        self.pos_tol,self.rot_tol = pos_tol,rot_tol
    def plan(self, T, max_iters=800, step_gain=0.55, lam=LAM_BASE, z_floor=0.03):
        R = T[:3,:3]; p = T[:3,3].copy(); p[2]=max(p[2], z_floor)
        q = np.empty(4); mujoco.mju_mat2Quat(q, R.flatten())
        di = mujoco.MjData(self.m); di.qpos[:]=self.d.qpos; di.qvel[:]=self.d.qvel
        mujoco.mj_forward(self.m, di)
        for _ in range(max_iters):
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
                else: di.qpos[adr] = qj
            mujoco.mj_forward(self.m, di)
        return True
    def label(self, T):
        ok,qg = self.plan(T)
        return 0 if (ok and self.short_traj_safe(qg)) else 1

# ---------- tiny MLP ----------
class IRN(nn.Module):
    def __init__(self, h=128, depth=3, drop=0.1):
        super().__init__()
        layers=[nn.Linear(9,h), nn.ReLU(True)]
        for _ in range(depth-1):
            layers += [nn.Linear(h,h), nn.ReLU(True), nn.Dropout(drop)]
        self.f = nn.Sequential(*layers); self.out = nn.Linear(h,1)
    def forward(self,x): return torch.sigmoid(self.out(self.f(x))).squeeze(-1)

# ---------- main (generate + train) ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panda_xml", type=str, default="franka_emika_panda/panda.xml")
    ap.add_argument("--ckpt_out", type=str, default="checkpoints/reach_irn.pt")
    ap.add_argument("--train_N", type=int, default=100000)
    ap.add_argument("--val_N", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--yaw_only", action="store_true", default=True)
    ap.add_argument("--x", type=float, nargs=2, default=[0.2,0.9])
    ap.add_argument("--y", type=float, nargs=2, default=[0.2,0.9])
    ap.add_argument("--z", type=float, nargs=2, default=[0.05,0.9])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)

    m,d = build_scene(args.panda_xml)
    arm,ee = find_arm_joints_and_ee(m)
    # set a comfy home pose similar to yours
    home = np.array([0.0,-0.6,0.0,-2.2,0.0,2.2,0.8], float)
    for j,q in zip(arm, home): d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_forward(m,d)

    ike = IKEval(m,d,ee,arm)
    bounds = {'x':tuple(args.x), 'y':tuple(args.y), 'z':tuple(args.z)}
    rng = np.random.default_rng(42)

    def build_split(N):
        Ts = sample_poses(N, bounds, yaw_only=args.yaw_only, rng=rng)
        X = np.stack([T_to_feat(T) for T in Ts]).astype(np.float32)
        y = np.zeros((N,), np.int64)
        for i,T in enumerate(Ts):
            y[i] = ike.label(T)
            if (i+1)%1000==0: print(f"[gen] {i+1}/{N} reachable={(1-y[:i+1]).mean():.3f}")
        return torch.from_numpy(X), torch.from_numpy(y)

    print("[IRN] generating train…"); Xtr,ytr = build_split(args.train_N)
    print("[IRN] generating val…");   Xva,yva = build_split(args.val_N)

    tr = DataLoader(TensorDataset(Xtr,ytr), batch_size=args.batch, shuffle=True, num_workers=2)
    va = DataLoader(TensorDataset(Xva,yva), batch_size=4096, shuffle=False)

    dev = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    model = IRN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best, best_state = 1e9, None
    for ep in range(1, args.epochs+1):
        model.train(); tot=n=0
        for xb,yb in tr:
            xb=xb.to(dev); yb=yb.float().to(dev)
            p = model(xb); loss = F.binary_cross_entropy(p, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0); n += xb.size(0)
        tr_loss = tot/n

        model.eval(); vt=vn=accn=0
        with torch.no_grad():
            for xb,yb in va:
                xb=xb.to(dev); yb=yb.float().to(dev)
                p = model(xb); vt += F.binary_cross_entropy(p, yb).item()*xb.size(0)
                pred = (p>=0.5).long().cpu()
                accn += (pred==yb.long().cpu()).sum().item(); vn += xb.size(0)
        va_loss = vt/vn; va_acc = accn/vn
        print(f"[ep {ep}] train {tr_loss:.4f} | val {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss < best: best, best_state = va_loss, {k:v.detach().cpu() for k,v in model.state_dict().items()}

    torch.save(best_state or model.state_dict(), args.ckpt_out)
    print(f"[IRN] saved → {args.ckpt_out}")

if __name__ == "__main__":
    main()
