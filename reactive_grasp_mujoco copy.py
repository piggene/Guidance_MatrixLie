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


# your modules
from loaders import get_dataloader
from models import get_model
from metrics import get_metrics
from utils.visualization import PlotlySubplotsVisualizer
from envs.lib.LieGroup import *   # for SO3_to_quaternion etc.

# at top
from energy_cost_rm4d import rm4d_load, energy_cost as _energy_cost_world


# -------------------------- Configs / Constants --------------------------

# TCP offset in hand local frame (center between fingertips; tweak ±0.005 if needed)
TCP_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.105], dtype=float)

# IK safety knobs
MAX_DQ_PER_STEP = 0.06     # rad/step clamp on joint update
BIAS_GAIN        = 0.0     # pull toward joint mid-ranges
LAM_BASE         = 1e-3    # DLS damping (auto-ramps near singularities)

# -------------------------- Utils --------------------------

# --- RM4D energy wrapper that tracks the *current* object pose (MuJoCo) ---
class EnergyCostRM4D:
    """
    Callable: energy_cost(T_obj) -> (...,1) (0 if feasible, 100 otherwise)
    - Reads current object pose from MuJoCo (R_obj, p_obj)
    - Converts T_obj (object frame; translation in model units) -> world (meters)
    - Queries RM4D energy
    Also provides: compose(T_obj) -> T_world (to execute the same pose in IK)
    """
    def __init__(self, device, s_total, m, d, obj_body, rm4d_energy_fn):
        self.device = device
        self.m = m
        self.d = d
        self.obj_body = obj_body
        self.rm4d_energy = rm4d_energy_fn
        self.S_TOTAL = torch.tensor(float(s_total), dtype=torch.float32, device=device)

        # buffers to avoid reallocs each call
        self.R_obj = torch.empty((3,3), dtype=torch.float32, device=device)
        self.p_obj = torch.empty((3,),   dtype=torch.float32, device=device)

    def _refresh_obj_pose(self):
        # Pull MuJoCo state (NumPy) -> GPU tensors
        Rnp = self.d.xmat[self.obj_body].reshape(3,3).copy()
        pnp = self.d.xpos[self.obj_body].copy()
        self.R_obj.copy_(torch.from_numpy(Rnp).to(self.device))
        self.p_obj.copy_(torch.from_numpy(pnp).to(self.device))

    @torch.no_grad()
    def compose(self, T_obj: torch.Tensor) -> torch.Tensor:
        """
        T_obj: (...,4,4) OBJECT frame (model units). Returns world-frame (...,4,4) meters.
        """
        T_obj = T_obj.to(self.device)
        self._refresh_obj_pose()

        lead = T_obj.shape[:-2]
        R_rel = T_obj[..., :3, :3]               # (...,3,3)
        t_rel = T_obj[..., :3, 3]                # (...,3)
        t_m   = self.S_TOTAL * t_rel             # (...,3) meters

        # Rw = R_obj * R_rel;  pw = p_obj + R_obj * t_m
        Rw = torch.einsum("ij,...jk->...ik", self.R_obj, R_rel)
        pw = self.p_obj + torch.einsum("ij,...j->...i", self.R_obj, t_m)

        T_w = torch.zeros((*lead, 4, 4), dtype=T_obj.dtype, device=T_obj.device)
        T_w[..., :3, :3] = Rw
        T_w[..., :3, 3]  = pw
        T_w[..., 3, 3]   = 1.0
        return T_w

    @torch.no_grad()
    def __call__(self, T_obj: torch.Tensor) -> torch.Tensor:
        """
        Returns (...,1): 0 if RM4D says reachable, else 100.
        """
        T_w = self.compose(T_obj)
        return self.rm4d_energy(T_w)

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

# -------------------------- RM4D viz helpers --------------------------
def _linspace_centers(lo, hi, n):
    if n <= 1: return np.array([(lo + hi) * 0.5], dtype=float)
    return np.linspace(lo, hi, n)

def _rm4d_envelope_rmax_per_z(pack, z_samples, a_min, a_max):
    """
    For each z in z_samples (world meters), compute the max reachable XY radius r.
    We collapse alpha with any(), then take max radius over occupied XY bins.
    """
    G = pack["grid"]
    if isinstance(G, torch.Tensor):
        G = G.to('cpu', dtype=torch.uint8).numpy()
    nbx, nby, nz, na = G.shape

    bx_lo, bx_hi = pack["bx_range"]
    by_lo, by_hi = pack["by_range"]
    z_lo,  z_hi  = pack["z_range"]
    a_lo,  a_hi  = pack["alpha_range"]

    bx_vals = _linspace_centers(bx_lo, bx_hi, nbx)
    by_vals = _linspace_centers(by_lo, by_hi, nby)
    a_vals  = _linspace_centers(a_lo,  a_hi,  na)
    r_grid  = np.sqrt(bx_vals[:, None]**2 + by_vals[None, :]**2)

    a_mask = (a_vals >= a_min) & (a_vals <= a_max)
    if not a_mask.any():
        a_mask[:] = True

    def _z_to_idx(z):
        if nz <= 1: return 0
        t = (z - z_lo) / max(z_hi - z_lo, 1e-12)
        return int(np.clip(round(t * (nz - 1)), 0, nz - 1))

    r_max_list, z_used = [], []
    z_axis_vals = _linspace_centers(z_lo, z_hi, nz)
    for z in z_samples:
        zc = float(np.clip(z, z_lo, z_hi))
        iz = _z_to_idx(zc)
        occ_xy = (G[:, :, iz, a_mask] > 0).any(axis=2)  # (nbx, nby)
        r_max = float(r_grid[occ_xy].max()) if occ_xy.any() else 0.0
        r_max_list.append(r_max)
        z_used.append(z_axis_vals[iz])
    return np.array(r_max_list, float), np.array(z_used, float)

def _build_boundary_ring_markers(pack,
                                 z_min, z_max,
                                 slices=8, phis=128, radius=0.006,
                                 a_min=0.0, a_max=math.pi):
    """
    Build lightweight ring markers: list of ("sphere", pos, rgba, size)
    drawing a circle at each z slice with radius = max reachable r at that z.
    """
    z0, z1 = pack["z_range"]
    z_min = float(np.clip(z_min, z0, z1))
    z_max = float(np.clip(z_max, z0, z1))
    if z_max < z_min: z_min, z_max = z_max, z_min

    z_samples = np.linspace(z_min, z_max, max(1, int(slices)))
    r_max, z_used = _rm4d_envelope_rmax_per_z(pack, z_samples, a_min, a_max)

    markers = []
    phis = max(8, int(phis))
    phi_vals = np.linspace(0.0, 2.0 * math.pi, phis, endpoint=False)

    for zi, (z, r) in enumerate(zip(z_used, r_max)):
        if r <= 0.0:  # nothing reachable at this z
            continue
        t = 0.25 + 0.6 * (zi / max(1, len(z_used) - 1))  # color gradient by slice
        rgba = np.array([1.0, t, 0.0, 0.9], float)
        for phi in phi_vals:
            p = np.array([r * math.cos(phi), r * math.sin(phi), z], float)
            markers.append(("sphere", p, rgba, np.array([radius, 0.0, 0.0], float)))
    return markers

def _draw_markers_to_user_scn(v, markers):
    scn = v.user_scn
    scn.ngeom = 0
    I = min(len(markers), scn.maxgeom)

    ident9 = np.eye(3, dtype=np.float64).reshape(9)  # flat 3x3

    for k in range(I):
        _, pos, rgba, size = markers[k]
        size64 = np.asarray(size, dtype=np.float64).reshape(3,)
        pos64  = np.asarray(pos,  dtype=np.float64).reshape(3,)
        rgba32 = np.asarray(rgba, dtype=np.float32).reshape(4,)

        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size64,
            pos64,
            ident9,
            rgba32
        )
        scn.ngeom += 1
    return I


# ----------------------------------------------

# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_result_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--guide_type', type=str, default='none')
    parser.add_argument('--device', default='1')  # 'cpu' or CUDA index string
    parser.add_argument('--obj_type', default='Mug')
    parser.add_argument('--panda_xml', type=str, default='franka_emika_panda/panda.xml')
    parser.add_argument('--pc_scale', type=float, default=8.0, help='Must match your training/data')
    parser.add_argument('--hz', type=int, default=240)
    parser.add_argument('--obj_center', type=float, nargs=3, default=[0.5, 0.7, 0.5],
                        help='Where to place the OBJECT FRAME (pc centroid) in world (x y z).')
    parser.add_argument('--move_duration', type=float, default=2.0,
                        help='Seconds for smooth joint-space motion after IK solves.')
    # for RM4D viz
    parser.add_argument('--rm4d_path', type=str, default='rm4d_franka.pt')
    parser.add_argument('--rm4d_boundary_slices', type=int, default=8,
                        help='How many z-level rings to draw')
    parser.add_argument('--rm4d_boundary_phis', type=int, default=128,
                        help='Points per ring on each slice')
    parser.add_argument('--rm4d_boundary_radius', type=float, default=0.006,
                        help='Sphere size for ring points')
    parser.add_argument('--rm4d_alpha_min', type=float, default=0.0)
    parser.add_argument('--rm4d_alpha_max', type=float, default=math.pi)
    parser.add_argument('--rm4d_z_min', type=float, default=0.0)
    parser.add_argument('--rm4d_z_max', type=float, default=1.20)
    
    args = parser.parse_args()
    # Load cfg & model
    cfg_file = [f for f in os.listdir(args.train_result_path) if f.endswith('.yml') or f.endswith('.yaml')][0]
    cfg = OmegaConf.load(os.path.join(args.train_result_path, cfg_file))
    cfg.model.ode_solver.name = 'SE3_RK_mk_guide'
    cfg.model.checkpoint = os.path.join(args.train_result_path, args.checkpoint)
    cfg.device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'
    set_seeds(cfg.get('seed', 1))

    model = get_model(cfg.model).to(cfg.device)
    rm4d_load(args.rm4d_path, device=cfg.device)
    rm4d_pack_viz = torch.load(args.rm4d_path, map_location='cpu')


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

    # Create RM4D-backed energy that reads the *current* object pose each call
    global energy_cost
    energy_cost = EnergyCostRM4D(
        device=cfg.device,
        s_total=s_total,
        m=m, d=d,
        obj_body=obj_body,
        rm4d_energy_fn=_energy_cost_world  # from energy_cost_rm4d import ... as _energy_cost_world
    )

    # ---- SPACE: compute TCP from predicted object-centered grasp & move ----
    def move_to_grasp_once(guide_type):
        # (A) model grasp in object frame (torch 4x4 on device)
        T_obj_TCP = get_grasp_pose(model, obj_tensor, cfg.device, guide_type)
        print(f"Energy at predicted grasp: {energy_cost(T_obj_TCP).item():.4f}")
        if T_obj_TCP.ndim == 3 and T_obj_TCP.shape[0] == 1:
            T_obj_TCP = T_obj_TCP[0]
        assert T_obj_TCP.shape == (4, 4)

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

        set_tcp_target_marker(p, q_wxyz)
        desired_target["has"] = True
        desired_target["p"] = p
        desired_target["q"] = q_wxyz
        # ======= ADD inside move_to_grasp_once(...) after desired_target[...] is set =======
        _goal_ctx["tcp_p"] = p.copy()
        _goal_ctx["tcp_q"] = q_wxyz.copy()


        # (D) plan IK and schedule smooth motion (non-blocking)
        solved = move_panda_tcp_to_SE3(
            T_world_TCP, duration=args.move_duration,
            step_gain=0.55, lam=LAM_BASE
        )
        if solved:
            print("[SPACE ✅] IK solved; smooth motion scheduled.")
        else:
            print("[SPACE ❌] Grasp pose not feasible.")
            
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
    # ---- RM4D viz toggle state ----
    rm4d_show = False
    rm4d_markers = None   # cached list of markers when enabled

    def on_key(keycode):
        nonlocal running, paused, rm4d_show, rm4d_markers
        try:
            ch = chr(keycode)
        except ValueError:
            ch = ''
        if ch == ' ':
            move_to_grasp_once(args.guide_type)
        elif ch in ('p', 'P'):
            paused = not paused
        elif ch in ('o', 'O'):
            rm4d_show = not rm4d_show
            rm4d_markers = _build_boundary_ring_markers(
                            rm4d_pack_viz,
                            z_min=args.rm4d_z_min, z_max=args.rm4d_z_max,
                            slices=args.rm4d_boundary_slices,
                            phis=args.rm4d_boundary_phis,
                            radius=args.rm4d_boundary_radius,
                            a_min=args.rm4d_alpha_min, a_max=args.rm4d_alpha_max
                        )
            print(f"[RM4D BOUNDARY] built {len(rm4d_markers)} points "
                    f"({args.rm4d_boundary_slices} slices × {args.rm4d_boundary_phis} pts)")
            print(f"[RM4D BOUNDARY] {'ENABLED' if rm4d_show else 'DISABLED'}")
        elif ch in ('q', 'Q', '\x1b'):  # q or ESC
            running = False


    
    with viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.lookat[:] = (0.0, 0.0, 0.05)
        v.cam.distance = 1.2
        v.cam.azimuth = 60
        v.cam.elevation = -25

        dt = 1.0 / float(args.hz)
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
            if rm4d_show and rm4d_markers:
                _draw_markers_to_user_scn(v, rm4d_markers)
            else:
                v.user_scn.ngeom = 0

            v.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()
