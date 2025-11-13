#!/usr/bin/env python3
# mujoco_grasp_demo.py
import argparse, os, glob, time, tempfile, random, copy
import numpy as np
import torch
import open3d as o3d
import mujoco
import math
from mujoco import viewer
from omegaconf import OmegaConf
from reach_energy_cost import init_energy_cost_irn

# your modules
from loaders import get_dataloader
from models import get_model
from metrics import get_metrics
from utils.visualization import PlotlySubplotsVisualizer
from envs.lib.LieGroup import *   # for SO3_to_quaternion etc.




# -------------------------- Configs / Constants --------------------------

# TCP offset in hand local frame (center between fingertips; tweak ±0.005 if needed)
TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)

# IK safety knobs
MAX_DQ_PER_STEP = 0.06     # rad/step clamp on joint update
BIAS_GAIN        = 0.0     # pull toward joint mid-ranges
LAM_BASE         = 1e-3    # DLS damping (auto-ramps near singularities)

# -------------------------- Utils --------------------------


def _minjerk(s):  # s in [0,1]
    return s**3 * (10 - 15*s + 6*s*s)

def set_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

def clone_mesh(mesh):
    """Open3D-safe clone across versions (0.10–0.18)."""
    if hasattr(mesh, "clone"):
        return mesh.clone()
    # fallback for older versions without .clone()
    return copy.deepcopy(mesh)

def get_single_pcd(obj_type: str,
                   root: str = "dataset/meshes",
                   num_pts: int = 1024,
                   scale: float = 8.0):
    """
    Load one .obj from dataset/meshes/<obj_type>/, return:
      (mesh_path, centered_pc_tensor (3,N), mesh_raw, pc_center_raw (3,))
    pc_center_raw is the centroid in *raw mesh units* so we can align frames in MuJoCo.
    """
    cat_dir = os.path.join(root, obj_type)
    if not os.path.isdir(cat_dir):
        raise ValueError(f"Unknown category '{obj_type}'. Expected folder: {cat_dir}")
    obj_files = sorted(glob.glob(os.path.join(cat_dir, "*.obj")))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in: {cat_dir}")
    mesh_path = obj_files[0]

    mesh_raw = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_raw.is_empty():
        raise ValueError(f"Mesh is empty or failed to load: {mesh_path}")

    # point cloud used by the model (scaled down by 1/scale, then centered)
    mesh_for_pc = clone_mesh(mesh_raw)
    if scale != 1.0:
        mesh_for_pc.scale(1/scale, center=(0, 0, 0))

    # Sample surface points and compute centroid in the scaled space
    pts_scaled = np.asarray(mesh_for_pc.sample_points_uniformly(num_pts).points).T  # (3,N)
    pc_center_scaled = pts_scaled.mean(axis=1, keepdims=True)                       # (3,1)
    pts_centered = pts_scaled - pc_center_scaled                                    # (3,N)

    # Convert centroid back to RAW mesh units
    pc_center_raw = (pc_center_scaled * scale).reshape(3)

    pc_tensor = torch.tensor(pts_centered, dtype=torch.float32)  # (3,N)
    return mesh_path, pc_tensor, mesh_raw, pc_center_raw


def get_grasp_pose(model, obj, device, guide_type, energy_cost):
    model.eval()
    with torch.no_grad():
        obj = obj.to(device)
        grasp_pos = model.single_guide_sample(obj, guide_type, energy_cost = energy_cost)
        return grasp_pos  # (4,4) or (1,4,4) torch

# ---------- Jacobian IK (DLS) for 7-DoF arm joints ----------

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

def _wrap_to_pi(q):
    """Wrap angle(s) to [-pi, pi]. Works on scalars and arrays."""
    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2.0 * np.pi) - np.pi


# ----- IK that solves at the TCP (hand + offset) with damping, bias & dq clamp -----
def ik_step_to_pose(m, d, ee_body_id, arm_joint_ids, p_target, q_target_wxyz,
                    step_gain=0.5, lam=LAM_BASE, bias_gain=BIAS_GAIN):
    """
    One DLS step toward (p_target, q_target_wxyz) using only the 7 arm joints.
    Returns (pos_err_norm, rot_err_norm).
    """
    # HAND pose
    R_hand = d.xmat[ee_body_id].reshape(3, 3).copy()
    p_hand = d.xpos[ee_body_id].copy()

    # CURRENT TCP pose (point attached to the hand at offset)
    p_cur = p_hand + R_hand @ TCP_OFFSET_HAND_LOCAL
    R_cur = R_hand  # orientation same as hand
    q_cur = np.empty(4); mujoco.mju_mat2Quat(q_cur, R_cur.flatten())

    # position error
    pos_err = p_target - p_cur

    # orientation error via quaternion difference (target * conj(cur)) -> rotvec
    q_conj = q_cur.copy(); q_conj[1:] *= -1
    q_err = np.empty(4); mujoco.mju_mulQuat(q_err, q_target_wxyz, q_conj)

    w, x, y, z = q_err
    v = np.array([x, y, z], dtype=float)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        rot_err = np.zeros(3)
    else:
        angle = 2.0 * np.arctan2(nv, max(w, 1e-12))
        rot_err = (angle / nv) * v

    # Weight orientation less so position converges first
    W = np.diag([1, 1, 1, 0.25, 0.25, 0.25])
    err6 = np.concatenate([pos_err, rot_err])
    errw = W @ err6

    # Jacobian at the TCP point (world coords p_cur on ee_body_id)
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, p_cur, ee_body_id)
    J = np.vstack([jacp, jacr])   # (6, nv)

    Jw = W @ J

    # restrict columns to arm DOFs (7 hinges)
    dof_addrs = np.array([m.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=int)
    J_arm = Jw[:, dof_addrs]      # (6,7)

    # Adaptive damping near singularities
    svals = np.linalg.svd(J_arm, compute_uv=False)
    lam_eff = lam * (10.0 if svals.min() < 1e-3 else 1.0)

    # Task step
    dq_task = dls(J_arm, errw, lam=lam_eff) * step_gain  # (7,)

    # Mid-range bias keeps us away from hard limits
    q_now = get_arm_qpos(m, d, arm_joint_ids)
    q_mid = get_joint_mids(m, d, arm_joint_ids)
    dq_bias = bias_gain * (q_mid - q_now)

    dq_arm = dq_task + dq_bias
    dq_arm = np.clip(dq_arm, -MAX_DQ_PER_STEP, MAX_DQ_PER_STEP)

    # apply to qpos of those 7 hinge joints
    for j, dq in zip(arm_joint_ids, dq_arm):
        adr = m.jnt_qposadr[j]
        d.qpos[adr] += dq

    clamp_qpos_in_range(m, d.qpos, arm_joint_ids)
    mujoco.mj_forward(m, d)

    # return norms for convergence checks
    return float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

# ---------- Training-accurate IK evaluator (exactly as in reach_gen_train_irn.py) ----------
class IKEval:
    def __init__(self, m, d, ee, arm, pos_tol=7e-4, rot_tol=6e-3, max_iters=800):
        self.m, self.d, self.ee, self.arm = m, d, ee, arm
        self.pos_tol, self.rot_tol = pos_tol, rot_tol
        self.max_iters = max_iters

    def plan(self, T, step_gain=0.55, lam=LAM_BASE, z_floor=0.03):
        R = T[:3, :3]
        p = T[:3, 3].copy()
        p[2] = max(p[2], z_floor)
        q = np.empty(4); mujoco.mju_mat2Quat(q, R.flatten())

        # scratch copy of data (same as training)
        di = mujoco.MjData(self.m)
        di.qpos[:] = self.d.qpos
        di.qvel[:] = self.d.qvel
        mujoco.mj_forward(self.m, di)

        for _ in range(self.max_iters):
            pe, re = ik_step_to_pose(self.m, di, self.ee, self.arm, p, q,
                                     step_gain=step_gain, lam=lam)
            if pe < self.pos_tol and re < self.rot_tol:
                qg = get_arm_qpos(self.m, di, self.arm)
                return True, qg
        # failed plan → return current arm qpos as fallback (same as training)
        return False, get_arm_qpos(self.m, self.d, self.arm)

    # def short_traj_safe(self, q_goal, steps=24):
    #     q0 = get_arm_qpos(self.m, self.d, self.arm)
    #     for s in np.linspace(0, 1, steps):
    #         q = (1 - s) * q0 + s * q_goal
    #         di = mujoco.MjData(self.m)
    #         di.qpos[:] = self.d.qpos
    #         for j, qj in zip(self.arm, q):
    #             adr = self.m.jnt_qposadr[j]
    #             if self.m.jnt_limited[j]:
    #                 lo, hi = self.m.jnt_range[j]
    #                 if qj < lo - 1e-4 or qj > hi + 1e-4:
    #                     return False
    #                 di.qpos[adr] = np.clip(qj, lo, hi)
    #             else:
    #                 di.qpos[adr] = qj
    #         mujoco.mj_forward(self.m, di)
    #     return True





# ---------- Scene build (include + markers) ----------

def build_mujoco_scene_with_include(
    panda_xml_path,
    obj_path_abs,
    mesh_scale_factor,
    base_pos_xyz,
    geom_offset_xyz=(0.0, 0.0, 0.0),
    axis_len=0.10,
    axis_rad=0.004,
):
    """
    Create a temporary scene XML in the SAME directory as panda_xml_path
    so <include file="..."> and relative asset paths resolve. Adds two mocap
    bodies: 'tcp_target' (desired) and 'tcp_reached' (current EE marker).

    geom_offset_xyz: a pose offset for the mesh within the "object" body so that
    the body origin can coincide with the model's object frame (pc centroid).
    """
    base_dir = os.path.dirname(os.path.abspath(panda_xml_path))
    panda_xml_basename = os.path.basename(panda_xml_path)

    gx, gy, gz = geom_offset_xyz

    scene_xml = f"""
<mujoco>
  <compiler angle="radian" meshdir="."/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <include file="{panda_xml_basename}"/>

  <asset>
    <mesh name="obj_mesh" file="{obj_path_abs}"
          scale="{mesh_scale_factor} {mesh_scale_factor} {mesh_scale_factor}"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>

    <!-- Body origin IS the object's learned frame (pc centroid). -->
    <body name="object" pos="{base_pos_xyz[0]} {base_pos_xyz[1]} {base_pos_xyz[2]}" mocap="true">
      <geom type="mesh" mesh="obj_mesh" pos="{gx} {gy} {gz}"
            rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    </body>

    <!-- desired TCP frame -->
    <body name="tcp_target" mocap="true">
      <geom type="capsule" fromto="0 0 0  {axis_len} 0 0" size="{axis_rad}" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0  0 {axis_len} 0" size="{axis_rad}" rgba="0 1 0 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0  0 0 {axis_len}" size="{axis_rad}" rgba="0 0 1 1" contype="0" conaffinity="0"/>
      <site name="tcp_target_origin" type="sphere" size="0.006" rgba="1 1 1 1"/>
    </body>

    <!-- reached TCP frame -->
    <body name="tcp_reached" mocap="true">
      <geom type="capsule" fromto="0 0 0  {axis_len} 0 0" size="{axis_rad}" rgba="1 0.5 0.5 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0  0 {axis_len} 0" size="{axis_rad}" rgba="0.5 1 0.5 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0  0 0 {axis_len}" size="{axis_rad}" rgba="0.5 0.5 1 1" contype="0" conaffinity="0"/>
      <site name="tcp_reached_origin" type="sphere" size="0.006" rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    fd, tmp_path = tempfile.mkstemp(prefix="scene_", suffix=".xml", dir=base_dir)
    with os.fdopen(fd, "w") as f:
        f.write(scene_xml)

    m = mujoco.MjModel.from_xml_path(tmp_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, tmp_path


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_result_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--guide_type', type=str, default='none')
    parser.add_argument('--device', default='0')  # 'cpu' or CUDA index string
    parser.add_argument('--obj_type', default='Mug')
    parser.add_argument('--panda_xml', type=str, default='franka_emika_panda/panda.xml')
    parser.add_argument('--pc_scale', type=float, default=8.0, help='Must match your training/data')
    parser.add_argument('--hz', type=int, default=240)
    parser.add_argument('--obj_center', type=float, nargs=3, default=[0.4, 0.4, 0.5],
                        help='Where to place the OBJECT FRAME (pc centroid) in world (x y z).')
    parser.add_argument('--move_duration', type=float, default=2.0,
                        help='Seconds for smooth joint-space motion after IK solves.')
    
    args = parser.parse_args()
    # Load cfg & model
    cfg_file = [f for f in os.listdir(args.train_result_path) if f.endswith('.yml') or f.endswith('.yaml')][0]
    cfg = OmegaConf.load(os.path.join(args.train_result_path, cfg_file))
    cfg.model.ode_solver.name = 'SE3_RK_mk_guide'
    cfg.model.checkpoint = os.path.join(args.train_result_path, args.checkpoint)
    cfg.device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'
    set_seeds(cfg.get('seed', 1))

    model = get_model(cfg.model).to(cfg.device)

    # ---- Mesh & point cloud (correct scaling) ----
    obj_path, obj_tensor, mesh_raw, pc_center_raw = get_single_pcd(args.obj_type, scale=args.pc_scale)
    if mesh_raw.is_empty():
        raise ValueError(f"Failed to load raw mesh for physics/visual: {obj_path}")

    # RAW mesh bbox (for world scaling debug)
    aabb_min = mesh_raw.get_min_bound()
    aabb_max = mesh_raw.get_max_bound()
    size_raw = (aabb_max - aabb_min)

    # Your original convention (cancels with s_total below)
    mesh_scale_factor = float(0.2 / max(size_raw))

    world_size_est = size_raw * mesh_scale_factor
    geom_offset_world = -pc_center_raw * mesh_scale_factor  # shift mesh so body origin = pc centroid

    print(f"[DEBUG] raw size: {size_raw}")
    print(f"[DEBUG] pc_center_raw: {pc_center_raw}")
    print(f"[DEBUG] mesh_scale_factor: {mesh_scale_factor}")
    print(f"[DEBUG] world size ≈ {world_size_est}")
    print(f"[DEBUG] geom offset (world units): {geom_offset_world}")

    base_pos = np.asarray(args.obj_center, dtype=float)  # BODY = object frame (pc centroid)
    obj_path_abs = os.path.abspath(obj_path)

    # ---- Build MuJoCo scene (with target/reached frames) ----
    m, d, tmp_scene_xml = build_mujoco_scene_with_include(
        panda_xml_path=args.panda_xml,
        obj_path_abs=obj_path_abs,
        mesh_scale_factor=mesh_scale_factor,
        base_pos_xyz=base_pos,
        geom_offset_xyz=tuple(geom_offset_world),
        axis_len=0.10,
        axis_rad=0.004,
    )
    
    

    # ---- find Panda joints robustly ----
    def _jid(name):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return j if j >= 0 else None

    # Try common naming schemes
    name_sets = [
        [f"panda_joint{i}" for i in range(1, 8)],
        [f"joint{i}" for i in range(1, 8)],
        [f"fr3_joint{i}" for i in range(1, 8)],
    ]
    arm_joint_ids = None
    for cand in name_sets:
        ids = [j for j in (_jid(nm) for nm in cand) if j is not None]
        if len(ids) == 7:
            arm_joint_ids = ids
            print(f"[INFO] using arm joints: {cand}")
            break
    if arm_joint_ids is None:
        # fall back to “first 7 hinge joints” heuristic
        hinge_ids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if len(hinge_ids) >= 7:
            arm_joint_ids = hinge_ids[:7]
            print("[WARN] joint names not recognized; using first 7 hinge joints.")
        else:
            all_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
            raise RuntimeError(f"Could not find 7 arm joints. Available joints: {all_names}")

    # home pose (same as Bullet)
    home_q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 2.2, 0.8], dtype=float)
    for j, q in zip(arm_joint_ids, home_q):
        d.qpos[m.jnt_qposadr[j]] = q

    # open gripper if present
    for gnm, val in [("panda_finger_joint1", 0.02), ("panda_finger_joint2", 0.02),
                     ("finger_joint1", 0.02), ("finger_joint2", 0.02)]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, gnm)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val


    # --- actuators → PD setpoints helpers ---
    arm_actuator_ids = []
    for i in range(1, 8):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
        if aid >= 0:
            arm_actuator_ids.append(aid)
        else:
            print(f"[WARN] actuator{i} not found")
            
    # ======= ADD: actuator mapping & sanity checks =======
    def _actuator_diag():
        # 1: Count & names
        if len(arm_actuator_ids) != 7:
            print(f"[DIAG] Expected 7 arm actuators, found {len(arm_actuator_ids)} → TRACKING WILL FAIL.")
        # 2: ctrlrange and gains
        for i, aid in enumerate(arm_actuator_ids, start=1):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            cr_lo, cr_hi = m.actuator_ctrlrange[aid] if m.actuator_ctrlrange is not None else (np.nan, np.nan)
            kp = m.actuator_gainprm[aid,0] if m.actuator_gainprm is not None else np.nan
            kd = -m.actuator_biasprm[aid,2] if m.actuator_biasprm is not None else np.nan  # your XML uses "0 -Kp -Kd"
            print(f"[DIAG] {name}: ctrlrange=({cr_lo:.4f},{cr_hi:.4f})  Kp≈{kp:.1f}  Kd≈{kd:.1f}")

    _actuator_diag()
    

    # ======= REPLACE: set_arm_position_targets =======
    _ctrl_clip_happened = False  # global toggle during a trajectory

    def set_arm_position_targets(q_des):
        global _ctrl_clip_happened
        assert len(q_des) == len(arm_actuator_ids)
        for aid, q in zip(arm_actuator_ids, q_des):
            lo, hi = m.actuator_ctrlrange[aid] if m.actuator_ctrlrange is not None else (-np.inf, np.inf)
            q_cmd = float(np.clip(q, lo, hi))
            if not np.isclose(q_cmd, q, atol=1e-9):
                _ctrl_clip_happened = True
            d.ctrl[aid] = q_cmd

    # Hold the home pose under gravity
    set_arm_position_targets(home_q)

    # ---- EE body & object body ----
    def _find_body(names, default_last=True):
        for nm in names:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid >= 0:
                return bid
        return (m.nbody - 1) if default_last else -1

    ee_body = _find_body(["panda_hand", "panda_link7", "hand", "link7"])
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    ike_train = IKEval(m, d, ee_body, arm_joint_ids,
                   pos_tol=7e-4, rot_tol=6e-3, max_iters=800)
    # ---- mocap ids for markers ----
    # mocap id for the object (now mocap-controlled)
    
    def _mocap_id(body_name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0: return -1
        return m.body_mocapid[bid]

    tcp_target_mocapid  = _mocap_id("tcp_target")
    tcp_reached_mocapid = _mocap_id("tcp_reached")
    assert tcp_target_mocapid  != -1, "tcp_target mocap body not found"
    assert tcp_reached_mocapid != -1, "tcp_reached mocap body not found"
    obj_mocap = _mocap_id("object")
    assert obj_mocap != -1, "object mocap body not found"

    # set initial mocap pose to match the starting base_pos / identity rotation
    d.mocap_pos[obj_mocap]  = np.asarray(base_pos, dtype=float)
    d.mocap_quat[obj_mocap] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    mujoco.mj_forward(m, d)

    def _object_T_world_np():
        """Return 4x4 SE(3) of the object mocap body in WORLD frame (NumPy)."""
        p = d.mocap_pos[obj_mocap].copy()    # (3,)
        q = d.mocap_quat[obj_mocap].copy()   # (w, x, y, z)
        Rflat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(Rflat, q)        # fills row-major 3x3 into Rflat
        R = Rflat.reshape(3, 3)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3,  3] = p
        return T

    def _refresh_energy_cost_from_object_pose():
        """Rebuild energy_cost so compose() uses the latest object world pose."""
        nonlocal energy_cost
        Tnp = _object_T_world_np()
        Ttorch = torch.tensor(Tnp, dtype=torch.float32, device=cfg.device)
        energy_cost = init_energy_cost_irn(
            ckpt_path="checkpoints/reach_irn.pt",
            cfg_device=cfg.device,
            s_total=s_total,
            base_pos=Ttorch,   # pass FULL 4x4 world pose of the object
        )

    def set_marker(mocapid, p_world, q_wxyz):
        d.mocap_pos[mocapid]  = np.asarray(p_world, dtype=float)
        d.mocap_quat[mocapid] = np.asarray(q_wxyz, dtype=float)
        
    # Define the random range (tweak as you like)
    # --- Cylindrical-space sampling for the object pose ---
    # r in meters, theta/yaw in radians, z in meters
    CYL_RANGE = {
        "r":     (0.32, 0.70),            # radial distance from base origin
        "theta": (np.deg2rad(-90), np.deg2rad(90)),   # polar angle in base XY
        "z":     (0.12, 0.45),            # height
        "yaw":   (np.deg2rad(-180), np.deg2rad(180)), # object spin about +Z
    }
    BASE_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)  # change if your base frame differs

    def _rand_in(a, b):  # uniform
        return float(a + (b - a) * random.random())

    def _Rz(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c,-s,0.0],
                        [s, c,0.0],
                        [0.0,0.0,1.0]], dtype=float)

    def _yaw_to_quat(yaw):
        # quaternion (w, x, y, z) for rotation about +Z by yaw
        half = 0.5 * yaw
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)

    # current cylindrical/yaw state (kept in sync with the mocap)
    _cyl_state = {"r": None, "theta": None, "z": None, "yaw": 0.0}

    def _sample_cyl_pose():
        r     = _rand_in(*CYL_RANGE["r"])
        theta = _rand_in(*CYL_RANGE["theta"])
        z     = _rand_in(*CYL_RANGE["z"])
        yaw   = _rand_in(*CYL_RANGE["yaw"])
        return r, theta, z, yaw

    def _cyl_to_cart(r, theta, z):
        x = BASE_ORIGIN[0] + r * math.cos(theta)
        y = BASE_ORIGIN[1] + r * math.sin(theta)
        return np.array([x, y, z], dtype=float)

    def move_object_cyl(r, theta, z, yaw):
        """Move object in cylindrical coords and update energy mapper."""
        p_world = _cyl_to_cart(r, theta, z)
        d.mocap_pos[obj_mocap]  = p_world
        d.mocap_quat[obj_mocap] = _yaw_to_quat(yaw)
        mujoco.mj_forward(m, d)

        # keep ‘energy_cost’ translation aligned (its compose() handles scale + translation);
        # rotation will be injected when composing the grasp (see below).
        nonlocal base_pos
        base_pos = p_world.copy()
        _cyl_state.update({"r": r, "theta": theta, "z": z, "yaw": yaw})
        print(f"[OBJECT] r={r:.3f} θ={math.degrees(theta):.1f}° z={z:.3f} yaw={math.degrees(yaw):.1f}°  →  {p_world}")

    def randomize_object_pose_cyl():
        r, th, z, yaw = _sample_cyl_pose()
        move_object_cyl(r, th, z, yaw)

    # --------- IK planning on a scratch copy (no visual warp) ----------
    def plan_ik_joint_goal(T_world_TCP, max_iters=10000, step_gain=0.55, lam=LAM_BASE,
                           pos_tol=7e-4, rot_tol=6e-3, z_floor=0.03):
        """
        Solve IK on a *scratch* MjData copy (d_ik) so the live robot doesn't warp.
        Returns (solved: bool, q_goal_arm: np.ndarray).
        """
        # Normalize incoming torch tensor
        if T_world_TCP.ndim == 3:
            T = T_world_TCP[0]
        else:
            T = T_world_TCP
        T = T.detach().cpu().numpy()
        R = T[:3, :3]
        p = T[:3, 3].copy()

        # Safety clamp on target TCP height (same rule you had)
        if p[2] < z_floor:
            p[2] = z_floor

        q_wxyz = np.empty(4)
        mujoco.mju_mat2Quat(q_wxyz, R.flatten())

        # Scratch data cloned from the *current* sim state
        d_ik = mujoco.MjData(m)
        d_ik.qpos[:] = d.qpos
        d_ik.qvel[:] = d.qvel
        mujoco.mj_forward(m, d_ik)

        solved = False
        pe = re = 0.0
        for k in range(max_iters):
            sg = step_gain * (0.6 if k > 150 else 0.8 if k > 75 else 1.0)
            pe, re = ik_step_to_pose(m, d_ik, ee_body, arm_joint_ids, p, q_wxyz,
                                     step_gain=sg, lam=lam, bias_gain=BIAS_GAIN)
            if pe < pos_tol and re < rot_tol:
                solved = True
                break

        # Extract the solved arm joints from d_ik (or fall back to current if not solved)
        q_goal_arm = get_arm_qpos(m, d_ik if solved else d, arm_joint_ids)
        return solved, q_goal_arm

    # --------- Trajectory scheduler (no stepping here) ----------
    active_traj = {"on": False, "t0": 0.0, "dur": 0.0, "q0": None, "q1": None}
    
    # ======= ADD: goal tracking =======
    _goal_ctx = {
        "tcp_p": None,          # desired p (3,)
        "tcp_q": None,          # desired q (wxyz)
        "pos_tol": 5e-3,        # 5 mm default
        "rot_tol_deg": 2.5,     # 2.5 deg default
    }


    def start_joint_traj(q_goal_arm, duration_sec):
        global _ctrl_clip_happened
        _ctrl_clip_happened = False
        active_traj["on"]  = True
        active_traj["t0"]  = d.time  # use sim time for determinism
        active_traj["dur"] = float(max(duration_sec, 1e-6))
        active_traj["q0"]  = get_arm_qpos(m, d, arm_joint_ids)
        q1_raw = np.asarray(q_goal_arm, dtype=float)
        active_traj["q1"] = _wrap_to_pi(q1_raw)
        
    # --------- Desired EEF pose visualization state ----------
    desired_target = {"has": False, "p": None, "q": None}  # p:(3,), q:wxyz(4,)

    def set_tcp_target_marker(p_world, q_wxyz):
        # reuses your mocap-based marker setter
        d.mocap_pos[tcp_target_mocapid]  = np.asarray(p_world, dtype=float)
        d.mocap_quat[tcp_target_mocapid] = np.asarray(q_wxyz, dtype=float)
    

    # --------- Public: solve IK then start smooth move (non-blocking) ----------
    def move_panda_tcp_to_SE3(T_world_TCP, duration=4.0, **ik_kwargs):
        """
        1) Solve IK offline to get joint goal.
        2) If solved, schedule a min-jerk joint-space trajectory (advanced in the viewer loop).
        """
        solved, q_goal_arm = plan_ik_joint_goal(
            T_world_TCP,
            max_iters=ik_kwargs.get("max_iters", 10000),
            step_gain=ik_kwargs.get("step_gain", 0.55),
            lam=ik_kwargs.get("lam", LAM_BASE),
            pos_tol=ik_kwargs.get("pos_tol", 7e-4),
            rot_tol=ik_kwargs.get("rot_tol", 6e-3),
            z_floor=ik_kwargs.get("z_floor", 0.03),
        )

        if solved:
            print("[IK ✅] Solved. Starting smooth joint motion...")
            start_joint_traj(q_goal_arm, duration)
        else:
            print("[IK ❌] Not feasible. Skipping motion.")

        return solved

    # ----- Make a energy_cost function.... which penalizes non-feasible grasps -----
    s_total = mesh_scale_factor * float(args.pc_scale)  # == 0.2 / max(size_raw) * pc_scale

    energy_cost = None
    _refresh_energy_cost_from_object_pose()
    
    # ---- SPACE: compute TCP from predicted object-centered grasp & move ----
    def move_to_grasp_once(guide_type):
        # (A) model grasp in object frame (torch 4x4 on device)
        T_obj_TCP = get_grasp_pose(model, obj_tensor, cfg.device, guide_type, energy_cost)
        if T_obj_TCP.ndim == 3 and T_obj_TCP.shape[0] == 1:
            T_obj_TCP = T_obj_TCP[0]
        assert T_obj_TCP.shape == (4, 4)
        print(f"Energy at predicted grasp: {energy_cost(T_obj_TCP).item():.4f}")
        
        # (B) compose to WORLD using the *same* logic as energy_cost
        T_world_TCP = energy_cost.compose(T_obj_TCP)  # (4,4) torch on device

        # (C) clamp Z above floor (keep in torch)
        if T_world_TCP[2, 3] < 0.03:
            T_world_TCP = T_world_TCP.clone()
            T_world_TCP[2, 3] = 0.03
        Tnp = T_world_TCP.detach().cpu().numpy()
        R = Tnp[:3, :3]
        p = Tnp[:3, 3].copy()
        q_wxyz = np.empty(4); mujoco.mju_mat2Quat(q_wxyz, R.flatten())
        x, y, z = Tnp[:3, 3]
        if not (0.04 <= x**2 + y**2 <= 0.81 and 0.05 <= z <= 0.9):
            print(f"[WARN] out-of-distribution pose for IRN/IK: ({x:.3f},{y:.3f},{z:.3f})")

        set_tcp_target_marker(p, q_wxyz)
        desired_target["has"] = True
        desired_target["p"] = p
        desired_target["q"] = q_wxyz
        # ======= ADD inside move_to_grasp_once(...) after desired_target[...] is set =======
        _goal_ctx["tcp_p"] = p.copy()
        _goal_ctx["tcp_q"] = q_wxyz.copy()
            
        ok, q_goal = ike_train.plan(Tnp, step_gain=0.55, lam=LAM_BASE, z_floor=0.03)
        reachable = bool(ok)

        print(f"[IK label (offline)] reachable={ok}")

        if not reachable:
            print("[SKIP] Training-consistent IK says UNREACHABLE.")
            label_start("Unreachable", duration=2.0)
            return

        # If reachable by training definition, execute that *same* goal
        label_clear() 
        print("[IK ✅] Training-consistent IK solved. Starting smooth joint motion...")
        start_joint_traj(q_goal, args.move_duration)
     
    def move_to_start_pose(m, d, arm_joint_ids, home_q):
        print("[INFO] Moving to home pose...")
        start_joint_traj(home_q, 2.0)
        desired_target["has"] = False
        # ======= ADD inside move_to_start_pose(...) =======
        _goal_ctx["tcp_p"] = None
        _goal_ctx["tcp_q"] = None        
            
    # ======= ADD: diagnostics helpers =======

    def _quat_err_deg(q_target_wxyz, q_cur_wxyz):
        # q_err = q_t * conj(q_c), angle = 2*atan2(|v|, w)
        w,x,y,z = q_target_wxyz
        qc = q_cur_wxyz.copy(); qc[1:] *= -1
        q = np.empty(4); mujoco.mju_mulQuat(q, np.array([w,x,y,z]), qc)
        w,x,y,z = q; nv = np.linalg.norm([x,y,z])
        if nv < 1e-12: return 0.0
        ang = 2.0*np.arctan2(nv, max(w,1e-12))
        return np.degrees(abs(ang))

    def _tcp_from_state(m, d, ee_body_id, tcp_offset_local=TCP_OFFSET_HAND_LOCAL):
        R_hand = d.xmat[ee_body_id].reshape(3,3)
        p_hand = d.xpos[ee_body_id]
        p_tcp  = p_hand + R_hand @ tcp_offset_local
        q_tcp  = np.empty(4); mujoco.mju_mat2Quat(q_tcp, R_hand.flatten())
        return p_tcp.copy(), q_tcp.copy()


    # ---- Viewer + keyboard ----
    running = True
    paused = False
   
    def on_key(keycode):
        nonlocal running, paused
        try:
            ch = chr(keycode)
        except ValueError:
            ch = ''
        if ch == ' ':
            label_clear()
            move_to_grasp_once(args.guide_type)
        elif ch in ('p', 'P'):
            paused = not paused
        elif ch in ('/', '?'):
            randomize_object_pose_cyl()
            _refresh_energy_cost_from_object_pose()
        elif ch in ('r', 'R'):
            move_to_start_pose(m, d, arm_joint_ids, home_q)
        elif ch in ('q', 'Q', '\x1b'):  # q or ESC
            running = False



    with viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.lookat[:] = (0.0, 0.0, 0.2)
        v.cam.distance   = 2.57293606605
        v.cam.azimuth    = -135.0
        v.cam.elevation  = -44.39410821999
        
        _label = {
            "idx": -1,           # reserved user_scn slot (once)
            "text": "",          # current message
            "t_end": 0.0,        # sim time when it should disappear
        }

        # pre-alloc constants to avoid per-frame allocations
        _LABEL_ZERO = np.zeros(3, dtype=float)
        _LABEL_RMAT = np.eye(3, dtype=float).ravel()
        _LABEL_RGBA_ON  = np.array([1, 1, 1, 1], dtype=float)
        _LABEL_RGBA_OFF = np.array([1, 1, 1, 0], dtype=float)

        def label_start(msg: str, duration: float = 2.0):
            _label["text"] = str(msg)
            _label["t_end"] = float(d.time) + float(duration)

        def label_clear():
            _label["text"] = ""
            _label["t_end"] = 0.0

        def label_draw(v, pos_world):
            """Draw/Hide the single label in-place without growing ngeom."""
            # reserve exactly one slot once
            scn = v.user_scn
            if _label["idx"] < 0:
                if scn.ngeom >= scn.maxgeom:
                    return  # nothing we can do
                _label["idx"] = scn.ngeom
                scn.ngeom += 1

            # if someone reset ngeom (e.g. your code sets to 0), restore so our slot exists
            if scn.ngeom <= _label["idx"]:
                scn.ngeom = _label["idx"] + 1

            active = _label["text"] and (float(d.time) <= _label["t_end"])

            gref = scn.geoms[_label["idx"]]
            mujoco.mjv_initGeom(
                gref,
                mujoco.mjtGeom.mjGEOM_LABEL,
                _LABEL_ZERO,
                np.asarray(pos_world, dtype=float),
                _LABEL_RMAT,
                _LABEL_RGBA_ON if active else _LABEL_RGBA_OFF,
            )
            gref.label = _label["text"] if active else ""

        dt = 1.0 / 500
        while v.is_running() and running:
            # ---- advance scheduled joint-space trajectory (min-jerk) ----
            if active_traj["on"]:
                # progress in sim time
                s_raw = (d.time - active_traj["t0"]) / active_traj["dur"]
                s_raw = float(np.clip(s_raw, 0.0, 1.0))
                s = _minjerk(s_raw)

                q0 = active_traj["q0"]
                q1 = active_traj["q1"]
                q_t = (1.0 - s) * q0 + s * q1
                set_arm_position_targets(q_t)

                if s_raw >= 1.0:
                    # lock final targets and stop trajectory
                    set_arm_position_targets(q1)
                    active_traj["on"] = False
                    # Joint errors
                    q_now = get_arm_qpos(m, d, arm_joint_ids)
                    print(f"actual qpos at end : {q_now}")
                    print(f"desired qpos at end : {q1}")
                    q_err = np.asarray(q1) - np.asarray(q_now)
                    q_err_max = float(np.max(np.abs(q_err)))
                    q_err_rms = float(np.sqrt(np.mean(q_err**2)))

                    # TCP errors
                    p_cur, q_cur = _tcp_from_state(m, d, ee_body)
                    p_tgt = _goal_ctx["tcp_p"]
                    q_tgt = _goal_ctx["tcp_q"]
                    pos_err = float(np.linalg.norm(p_cur - p_tgt)) if p_tgt is not None else float('nan')
                    rot_err_deg = _quat_err_deg(q_tgt, q_cur) if q_tgt is not None else float('nan')

                    # Joint-limit contact check (hard clamp at joint range)
                    hit_joint_limit = False
                    for j in arm_joint_ids:
                        adr = m.jnt_qposadr[j]
                        if m.jnt_limited[j]:
                            lo, hi = m.jnt_range[j]
                            if abs(d.qpos[adr] - lo) < 1e-6 or abs(d.qpos[adr] - hi) < 1e-6:
                                hit_joint_limit = True
                                break

                    # Verdict
                    status = "REACHED"
                    print(
                        f"[TRACKING {status}]  "
                        f"joint_err_max={q_err_max:.4f} rad  joint_err_rms={q_err_rms:.4f} rad  "
                        f"tcp_pos_err={pos_err*1e3:.1f} mm  tcp_rot_err={rot_err_deg:.2f} deg"
                    )

            if not paused:
                mujoco.mj_step(m, d)

            # update REACHED marker at the TCP (hand + offset)
            R_hand = d.xmat[ee_body].reshape(3,3)
            p_hand = d.xpos[ee_body]
            p_tcp  = p_hand + R_hand @ TCP_OFFSET_HAND_LOCAL
            q_tcp  = np.empty(4); mujoco.mju_mat2Quat(q_tcp, R_hand.flatten())
            set_marker(tcp_reached_mocapid, p_tcp, q_tcp)
            
            # keep showing the desired (target) EEF pose until next SPACE updates it
            if desired_target["has"]:
                set_tcp_target_marker(desired_target["p"], desired_target["q"])
                label_draw(v, desired_target["p"])
            else:
                v.user_scn.ngeom = 0            
                
            v.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()
