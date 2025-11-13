#!/usr/bin/env python3
# reach_gen_train_irn.py  (cylindrical sampling)
import os, math, argparse, json, csv, random, datetime, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import mujoco

# ---------- constants (match your sim) ----------
TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)
MAX_DQ_PER_STEP = 0.06
LAM_BASE = 1e-3

# ---------- utils: reproducibility ----------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

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
    with open(tmp, "w") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_path(tmp)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d

# ---------- utility: find 7 arm joints; EE body ----------
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

# ---------- sampling + features ----------
def _project_so3(R):
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R

def rot_to_6d(R): return np.concatenate([R[:,0], R[:,1]], 0)

def T_to_feat(T):
    return np.asarray(np.concatenate([T[:3,3], rot_to_6d(T[:3,:3])],0), dtype=np.float32)

def sample_poses_cyl(N, cyl_bounds, z_floor=0.03, rng=None):
    """
    Sample target poses using cylindrical coordinates:
      r in [r_min, r_max], theta in [th_min, th_max], z in [z_min, z_max]
    Convert to Cartesian: x=r*cos(theta), y=r*sin(theta)
    """
    rng = rng or np.random.default_rng()
    r = rng.uniform(*cyl_bounds['r'], N)
    theta = rng.uniform(*cyl_bounds['theta'], N)
    z = rng.uniform(*cyl_bounds['z'], N)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    Ts=[]
    for xi, yi, zi in zip(x, y, z):
        zi = max(zi, z_floor)
        axis = rng.normal(size=3); axis/= (np.linalg.norm(axis)+1e-9)
        ang = rng.uniform(-math.pi, math.pi)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.eye(3)+math.sin(ang)*K+(1-math.cos(ang))*(K@K)
        R = _project_so3(R)
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=[xi, yi, zi]; Ts.append(T)
    return Ts

# ---------- labeler ----------
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

# ---------- tiny MLP ----------
class IRN(nn.Module):
    def __init__(self, h=128, depth=3, drop=0.1):
        super().__init__()
        layers=[nn.Linear(9,h), nn.ReLU(True)]
        for _ in range(depth-1):
            layers += [nn.Linear(h,h), nn.ReLU(True), nn.Dropout(drop)]
        self.f = nn.Sequential(*layers); self.out = nn.Linear(h,1)
    def forward(self,x): return torch.sigmoid(self.out(self.f(x))).squeeze(-1)

# ---------- IO helpers ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_npz(path, X, y, meta: dict):
    ensure_dir(os.path.dirname(path) or ".")
    np.savez_compressed(path, X=X, y=y, meta=json.dumps(meta))
    print(f"[data] wrote {path}  (N={len(y)}, pos_rate={(1-y).mean():.3f})")

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    X = torch.from_numpy(z["X"])
    y = torch.from_numpy(z["y"])
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    return X, y, meta

def write_args_json(path, args_dict):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)
    print(f"[run] args → {path}")

def open_csv_logger(path, header):
    ensure_dir(os.path.dirname(path) or ".")
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(header)
        f.flush()
    return f, w

# ---------- main (generate + train) ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panda_xml", type=str, default="franka_emika_panda/panda.xml")
    ap.add_argument("--ckpt_out", type=str, default="checkpoints/reach_irn.pt")
    ap.add_argument("--data_dir", type=str, default="data/reach_irn")
    ap.add_argument("--log_dir", type=str, default="logs/reach_irn")

    # data control
    ap.add_argument("--gen_data", action="store_true", default=False,
                    help="If set, (re)generate train/val NPZ; otherwise load existing.")
    ap.add_argument("--train_npz", type=str, default="",
                    help="Explicit path to train .npz (overrides data_dir).")
    ap.add_argument("--val_npz", type=str, default="",
                    help="Explicit path to val .npz (overrides data_dir).")
    ap.add_argument("--train_N", type=int, default=1600000)
    ap.add_argument("--val_N", type=int, default=80000)
    ap.add_argument("--gen_log_every", type=int, default=1000,
                    help="Print ETA every K samples during data gen.")

    # training
    ap.add_argument("--epochs", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    # early stopping
    ap.add_argument("--early_stop_acc", type=float, default=-1.0,
                    help="Stop if val_acc >= this threshold; negative disables.")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="Stop if no val_acc improvement for P epochs; 0 disables.")

    # workspace & IK (CYLINDRICAL)
    ap.add_argument("--r", type=float, nargs=2, default=[0.2, 0.9],
                    help="Radial distance range [m] from base frame origin.")
    ap.add_argument("--theta", type=float, nargs=2, default=[-math.pi, math.pi],
                    help="Azimuth range [rad]. Use -pi to pi for full wrap.")
    ap.add_argument("--z", type=float, nargs=2, default=[0.05, 0.9],
                    help="Vertical range [m].")
    ap.add_argument("--max_ik_iters", type=int, default=800)

    args = ap.parse_args()
    set_seeds(args.seed)

    ensure_dir(args.data_dir)
    ensure_dir(os.path.dirname(args.ckpt_out) or ".")
    ensure_dir(args.log_dir)

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_args_json(os.path.join(args.log_dir, f"run_{run_stamp}.json"), vars(args))

    # build scene
    m,d = build_scene(args.panda_xml)
    arm,ee = find_arm_joints_and_ee(m)
    home = np.array([0.0,-0.6,0.0,-2.2,0.0,2.2,0.8], float)
    for j,q in zip(arm, home): d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_forward(m,d)

    # dataset paths
    train_npz = args.train_npz or os.path.join(args.data_dir, f"train_{args.train_N}_seed{args.seed}.npz")
    val_npz   = args.val_npz   or os.path.join(args.data_dir, f"val_{args.val_N}_seed{args.seed}.npz")

    # generate or load datasets (with ETA logging)
    cyl_bounds = {'r': tuple(args.r), 'theta': tuple(args.theta), 'z': tuple(args.z)}
    rng = np.random.default_rng(args.seed)

    if args.gen_data or (not os.path.exists(train_npz)) or (not os.path.exists(val_npz)):
        print("[IRN] (re)generating datasets…")
        ike = IKEval(m,d,ee,arm, max_iters=args.max_ik_iters)

        def build_split(N, tag):
            Ts = sample_poses_cyl(N, cyl_bounds, rng=rng)
            X = np.stack([T_to_feat(T) for T in Ts]).astype(np.float32)
            y = np.zeros((N,), np.int64)
            t0 = time.time()
            for i,T in enumerate(Ts):
                y[i] = ike.label(T)
                if (i+1) % args.gen_log_every == 0 or (i+1) == N:
                    elapsed = time.time() - t0
                    rate = (i+1) / max(elapsed, 1e-6)
                    remain = (N - (i+1)) / max(rate, 1e-6)
                    pos_rate = (1 - y[:i+1]).mean()
                    print(f"[gen:{tag}] {i+1}/{N} | {rate:6.1f} samp/s | ETA { _fmt_eta(remain) } | pos={pos_rate:.3f}")
            return X, y

        Xtr, ytr = build_split(args.train_N, "train")
        Xva, yva = build_split(args.val_N,   "val")

        meta = dict(cyl_bounds=cyl_bounds, seed=args.seed, max_ik_iters=args.max_ik_iters,
                    panda_xml=args.panda_xml, time=run_stamp, coord_system="cylindrical")
        save_npz(train_npz, Xtr, ytr, meta)
        save_npz(val_npz,   Xva, yva, meta)

        # latest convenience copies
        latest_train = os.path.join(args.data_dir, "train_latest.npz")
        latest_val   = os.path.join(args.data_dir, "val_latest.npz")
        try:
            if os.path.islink(latest_train): os.unlink(latest_train)
            os.symlink(os.path.abspath(train_npz), latest_train)
        except Exception:
            np.savez_compressed(latest_train, X=Xtr, y=ytr, meta=json.dumps(meta))
        try:
            if os.path.islink(latest_val): os.unlink(latest_val)
            os.symlink(os.path.abspath(val_npz), latest_val)
        except Exception:
            np.savez_compressed(latest_val, X=Xva, y=yva, meta=json.dumps(meta))
    else:
        print(f"[IRN] using existing datasets:\n  train: {train_npz}\n  val  : {val_npz}")

    # Load tensors
    Xtr, ytr, _ = load_npz(train_npz)
    Xva, yva, _ = load_npz(val_npz)

    # Dataloaders
    tr = DataLoader(TensorDataset(Xtr,ytr), batch_size=args.batch, shuffle=True, num_workers=2)
    va = DataLoader(TensorDataset(Xva,yva), batch_size=4096, shuffle=False)

    # Device & model
    dev = torch.device(args.device if (args.device=="cpu" or ("cuda" in args.device and torch.cuda.is_available())) else "cpu")
    model = IRN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # CSV logger
    csv_path = os.path.join(args.log_dir, "train_log.csv")
    csv_f, csv_w = open_csv_logger(csv_path, header=[
        "time","run","epoch","train_loss","val_loss","val_acc",
        "train_pos_rate","val_pos_rate","train_N","val_N","device"
    ])
    train_pos_rate = float((1 - ytr.numpy()).mean())
    val_pos_rate   = float((1 - yva.numpy()).mean())

    best_val_acc = -1.0
    patience_left = args.early_stop_patience if args.early_stop_patience > 0 else None

    best_loss, best_state = 1e9, None
    for ep in range(1, args.epochs+1):
        # -------- train --------
        model.train(); tot=n=0
        for xb,yb in tr:
            xb=xb.to(dev); yb=yb.float().to(dev)
            p = model(xb); loss = F.binary_cross_entropy(p, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0); n += xb.size(0)
        tr_loss = tot/n

        # -------- validate --------
        model.eval(); vt=vn=accn=0
        with torch.no_grad():
            for xb,yb in va:
                xb=xb.to(dev); yb=yb.float().to(dev)
                p = model(xb)
                vt += F.binary_cross_entropy(p, yb).item()*xb.size(0)
                pred = (p>=0.5).long().cpu()
                accn += (pred==yb.long().cpu()).sum().item()
                vn += xb.size(0)
        va_loss = vt/vn
        va_acc  = accn/vn

        # -------- logs --------
        print(f"[ep {ep}] train {tr_loss:.4f} | val {va_loss:.4f} acc {va_acc:.4f}")
        csv_w.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            run_stamp, ep, f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}",
            f"{train_pos_rate:.6f}", f"{val_pos_rate:.6f}",
            int(len(ytr)), int(len(yva)), str(dev)
        ])
        csv_f.flush()

        # Track best by val loss (for checkpoint) and best acc (for early stop)
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}

        # Early stop by accuracy threshold
        if args.early_stop_acc >= 0.0 and va_acc >= args.early_stop_acc:
            print(f"[early-stop] Reached val_acc {va_acc:.4f} ≥ {args.early_stop_acc:.4f} at epoch {ep}.")
            break

        # Early stop by patience on val_acc improvements (if enabled)
        if patience_left is not None:
            if va_acc > best_val_acc + 1e-6:
                best_val_acc = va_acc
                patience_left = args.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early-stop] No val_acc improvement for {args.early_stop_patience} epochs (best {best_val_acc:.4f}).")
                    break

    torch.save(best_state or model.state_dict(), args.ckpt_out)
    csv_f.close()
    print(f"[IRN] saved model → {args.ckpt_out}")
    print(f"[IRN] logs → {csv_path}")
    print(f"[IRN] train_npz: {train_npz}")
    print(f"[IRN]   val_npz: {val_npz}")

if __name__ == "__main__":
    main()
