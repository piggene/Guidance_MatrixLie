#!/usr/bin/env python3
# mujoco_grasp_demo.py
import argparse, os, glob, time, tempfile, random, copy
import numpy as np
import torch
import open3d as o3d
import mujoco
from mujoco import viewer
from omegaconf import OmegaConf

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
BIAS_GAIN        = 0.0    # pull toward joint mid-ranges
LAM_BASE         = 1e-3    # DLS damping (auto-ramps near singularities)

# -------------------------- Utils --------------------------

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

def energy_cost(x):
        """
        Penalize feasibility of grasp pose

        Args:
            x: (..., 4, 4) SE(3) transforms. Supports (D,4,4) or (N,D,4,4).

        Returns:
            penalty: (..., 1) tensor. (D,1) if input is (D,4,4), (N,D,1) if (N,D,4,4).
        """
        
    #TODO check given SE(3) is valid... (is it feasible...)
    


def get_grasp_pose(model, obj, device, guide_type):
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
  <option gravity="0 0 -9.81" integrator="RK4"/>

  <include file="{panda_xml_basename}"/>

  <asset>
    <mesh name="obj_mesh" file="{obj_path_abs}"
          scale="{mesh_scale_factor} {mesh_scale_factor} {mesh_scale_factor}"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>

    <!-- Body origin IS the object's learned frame (pc centroid). -->
    <body name="object" pos="{base_pos_xyz[0]} {base_pos_xyz[1]} {base_pos_xyz[2]}">
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
    parser.add_argument('--obj_center', type=float, nargs=3, default=[0.5, 0.0, 0.4],
                        help='Where to place the OBJECT FRAME (pc centroid) in world (x y z).')
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

    mujoco.mj_forward(m, d)

    # --- actuators → PD setpoints helpers ---
    arm_actuator_ids = []
    for i in range(1, 8):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
        if aid >= 0:
            arm_actuator_ids.append(aid)
        else:
            print(f"[WARN] actuator{i} not found")

    def set_arm_position_targets(q_des):
        assert len(q_des) == len(arm_actuator_ids)
        for aid, q in zip(arm_actuator_ids, q_des):
            d.ctrl[aid] = float(q)

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

    # ---- mocap ids for markers ----
    def _mocap_id(body_name):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0: return -1
        return m.body_mocapid[bid]

    tcp_target_mocapid  = _mocap_id("tcp_target")
    tcp_reached_mocapid = _mocap_id("tcp_reached")
    assert tcp_target_mocapid  != -1, "tcp_target mocap body not found"
    assert tcp_reached_mocapid != -1, "tcp_reached mocap body not found"

    def set_marker(mocapid, p_world, q_wxyz):
        d.mocap_pos[mocapid]  = np.asarray(p_world, dtype=float)
        d.mocap_quat[mocapid] = np.asarray(q_wxyz, dtype=float)

    # ----- IK driver: do solve with contacts+actuation disabled; PD holds the result -----
    def move_panda_tcp_to_SE3(T_world_TCP, iters=200, step_gain=0.5, lam=LAM_BASE):
        """T_world_TCP: torch(4,4) or (1,4,4). Controls the TCP point (offset from hand)."""
        if T_world_TCP.ndim == 3:
            T = T_world_TCP[0]
        else:
            T = T_world_TCP
        T = T.detach().cpu().numpy()
        R = T[:3, :3]
        p = T[:3, 3]

        # clamp target Z above floor a bit
        if p[2] < 0.03:
            p = p.copy(); p[2] = 0.03

        q_wxyz = np.empty(4)
        mujoco.mju_mat2Quat(q_wxyz, R.flatten())

        # visualize DESIRED
        set_marker(tcp_target_mocapid, p, q_wxyz)
        mujoco.mj_forward(m, d)

        # --- Disable actuation & contacts so solver never fights dynamics/contacts
        backup_flags = int(m.opt.disableflags)
        m.opt.disableflags |= (
            mujoco.mjtDisableBit.mjDSBL_ACTUATION |
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )
        try:
            pe = re = 0.0
            for k in range(iters):
                sg = step_gain * (0.6 if k > 120 else 0.8 if k > 60 else 1.0)
                pe, re = ik_step_to_pose(m, d, ee_body, arm_joint_ids, p, q_wxyz,
                                         step_gain=sg, lam=lam, bias_gain=BIAS_GAIN)
                if pe < 7e-4 and re < 6e-3:  # sub-mm, <0.35 deg
                    break
        finally:
            m.opt.disableflags = backup_flags  # re-enable actuation & contacts

        # --- Lock in the solved posture as the PD setpoint so gravity can't pull it away
        set_arm_position_targets(get_arm_qpos(m, d, arm_joint_ids))
        print(f"[IK] done: pos_err={pe:.6f} m, rot_err={re:.6f} rad")

    # ---- SPACE: compute TCP from predicted object-centered grasp & move ----
    def move_to_grasp_once(guide_type):
        # (A) object world pose (BODY ORIGIN == pc centroid)
        R_obj = d.xmat[obj_body].reshape(3, 3)
        p_obj = d.xpos[obj_body].copy()

        # (B) model grasp in object-centered frame (torch 4x4, same frame!)
        T_obj_TCP = get_grasp_pose(model, obj_tensor, cfg.device, guide_type)
        if T_obj_TCP.ndim == 3 and T_obj_TCP.shape[0] == 1:
            T_obj_TCP = T_obj_TCP[0]
        assert T_obj_TCP.shape == (4, 4)
        R_pred = T_obj_TCP[:3, :3].cpu().numpy()
        t_pred = T_obj_TCP[:3, 3].cpu().numpy()

        # (C) translation scaling (cancels pc_scale -> consistent with your training)
        s_total = mesh_scale_factor * float(args.pc_scale)  # == 0.2 / max(size_raw)
        t_world_rel = s_total * t_pred

        # (D) compose world TCP
        R_world_TCP = R_obj @ R_pred
        p_world_TCP = p_obj + R_obj @ t_world_rel
        p_world_TCP[2] = max(p_world_TCP[2], 0.03)  # keep above floor

        # (E) build torch 4x4 for the IK function
        T_world_TCP_np = np.eye(4, dtype=float)
        T_world_TCP_np[:3, :3] = R_world_TCP
        T_world_TCP_np[:3, 3]  = p_world_TCP
        T_world_TCP_torch = torch.from_numpy(T_world_TCP_np).float()

        move_panda_tcp_to_SE3(T_world_TCP_torch, iters=220, step_gain=0.55, lam=LAM_BASE)
        print("[SPACE] moved TCP to predicted grasp (desired frame drawn).")

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
            move_to_grasp_once(args.guide_type)
        elif ch in ('p', 'P'):
            paused = not paused
        elif ch in ('q', 'Q', '\x1b'):  # q or ESC
            running = False

    with viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.lookat[:] = (0.0, 0.0, 0.05)
        v.cam.distance = 1.2
        v.cam.azimuth = 60
        v.cam.elevation = -25

        dt = 1.0 / float(args.hz)
        while v.is_running() and running:
            if not paused:
                mujoco.mj_step(m, d)

            # update REACHED marker at the TCP (hand + offset)
            R_hand = d.xmat[ee_body].reshape(3,3)
            p_hand = d.xpos[ee_body]
            p_tcp  = p_hand + R_hand @ TCP_OFFSET_HAND_LOCAL
            q_tcp  = np.empty(4); mujoco.mju_mat2Quat(q_tcp, R_hand.flatten())
            set_marker(tcp_reached_mocapid, p_tcp, q_tcp)

            v.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()
