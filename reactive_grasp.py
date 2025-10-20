import argparse
from datetime import datetime
import os, glob
from omegaconf import OmegaConf
import logging
import yaml
import random
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
import open3d as o3d
import time
import pybullet as p
import pybullet_data

from loaders import get_dataloader
from models import get_model
from metrics import get_metrics
from utils.visualization import PlotlySubplotsVisualizer
from envs.lib.LieGroup import *

def spawn_panda(base_pos=(-0.6, 0.0, 0.0), base_yaw_deg=90.0, urdf_rel_path=("envs", "models", "panda", "panda_hand.urdf")):
    """Load a fixed-base Panda, set a home pose, and open the gripper."""
    panda_urdf = os.path.join(*urdf_rel_path)
    base_orn = p.getQuaternionFromEuler([0, 0, np.deg2rad(base_yaw_deg)])

    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE 
    panda_id = p.loadURDF(
        panda_urdf,
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=True,
        flags=flags
    )

    # Build name -> index map
    name_to_j = {p.getJointInfo(panda_id, j)[1].decode(): j for j in range(p.getNumJoints(panda_id))}

    # Arm home pose (safe, slightly bent)
    home_q = [0.0, -0.6, 0.0, -2.2, 0.0, 2.2, 0.8]
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    for i, jn in enumerate(arm_names):
        if jn in name_to_j:
            p.resetJointState(panda_id, name_to_j[jn], home_q[i])

    # Open gripper (each finger ~0.02 m)
    for jn, tgt in [("panda_finger_joint1", 0.02), ("panda_finger_joint2", 0.02)]:
        if jn in name_to_j:
            j = name_to_j[jn]
            p.resetJointState(panda_id, j, tgt)
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, targetPosition=tgt, force=20.0)

    return panda_id, name_to_j

def move_panda_tcp_to_SE3(
    panda_id,
    name_to_j,
    T_world_TCP,
    tcp_offset_in_hand=(0.0, 0.0, 0.0),
    tcp_rot_offset_rpy=(0.0, 0.0, 0.0),
    steps=240,
    max_force=200.0,
    ik_iters=200,
    ik_thresh=1e-4,
    debug_draw=True,              # <<< draw axes/labels in the GUI
    debug_length=0.10,            # axis length (m)
    debug_width=2,                # line width
    debug_lifetime=0,             # 0 = persist until removed/overwritten
):
    import numpy as np

    def _fmt(v, prec=4): return [round(float(x), prec) for x in v]
    def _normalize_quat(q):
        q = np.asarray(q, float); n = np.linalg.norm(q)
        return [0,0,0,1] if n < 1e-9 else (q / n).tolist()

    # -- tiny helper to draw a pose triad + label, and remove prior drawing --
    def _draw_pose(name, pos, quat, prev_ids=None):
        ids = []
        R = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        p0 = np.array(pos, dtype=float)
        ends = [p0 + R[:,0]*debug_length,  # X (red)
                p0 + R[:,1]*debug_length,  # Y (green)
                p0 + R[:,2]*debug_length]  # Z (blue)
        colors = [[1,0,0],[0,1,0],[0,0,1]]

        # remove previous
        if prev_ids:
            for uid in prev_ids:
                p.removeUserDebugItem(uid)

        # draw new axes
        for k in range(3):
            uid = p.addUserDebugLine(p0.tolist(), ends[k].tolist(), colors[k],
                                     lineWidth=debug_width, lifeTime=debug_lifetime)
            ids.append(uid)
        # label
        uid_txt = p.addUserDebugText(name, p0.tolist(), textColorRGB=[1,1,1],
                                     textSize=1.2, lifeTime=debug_lifetime)
        ids.append(uid_txt)
        return ids

    # keep handles across calls so we overwrite instead of stacking
    if not hasattr(move_panda_tcp_to_SE3, "_dbg_ids"):
        move_panda_tcp_to_SE3._dbg_ids = {"desired": None, "reached": None}

    # ----- link name -> index -----
    link_name_to_idx = {p.getJointInfo(panda_id, ji)[12].decode(): ji
                        for ji in range(p.getNumJoints(panda_id))}
    if "panda_hand" not in link_name_to_idx:
        raise RuntimeError("EE link 'panda_hand' not found. Make sure you did NOT use URDF_MERGE_FIXED_LINKS.")
    ee_link = link_name_to_idx["panda_hand"]
    print(f"[IK] Using EE link index {ee_link} (name='panda_hand')")

    # ----- desired TCP -> desired HAND (remove TCP offset) -----
    target_tcp_pos, target_tcp_quat = _parse_se3(T_world_TCP)   # ensure quat is XYZW for Bullet
    target_tcp_quat = _normalize_quat(target_tcp_quat)

    if debug_draw:
        move_panda_tcp_to_SE3._dbg_ids["desired"] = _draw_pose(
            "TCP_desired", target_tcp_pos, target_tcp_quat, prev_ids=move_panda_tcp_to_SE3._dbg_ids["desired"]
        )

    tcp_off_quat = p.getQuaternionFromEuler(tcp_rot_offset_rpy)
    inv_off_pos, inv_off_quat = p.invertTransform(tcp_offset_in_hand, tcp_off_quat)
    hand_pos, hand_quat = p.multiplyTransforms(target_tcp_pos, target_tcp_quat, inv_off_pos, inv_off_quat)
    hand_quat = _normalize_quat(hand_quat)

    # ----- use ONLY the 7 arm joints for IK -----
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    arm_idxs = [name_to_j[n] for n in arm_names if n in name_to_j]
    if len(arm_idxs) != 7:
        raise RuntimeError(f"Expected 7 arm joints, got {len(arm_idxs)}")

    lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
    for ji in arm_idxs:
        jinfo = p.getJointInfo(panda_id, ji)
        ll, ul = float(jinfo[8]), float(jinfo[9])
        lower_limits.append(ll); upper_limits.append(ul)
        joint_ranges.append(ul - ll)
        rest_poses.append(p.getJointState(panda_id, ji)[0])

    # ----- IK -----
    q_sol = p.calculateInverseKinematics(
        panda_id, ee_link,
        hand_pos, hand_quat,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=ik_iters,
        residualThreshold=ik_thresh,
    )
    q_arm = q_sol[:7]

    # log desired (numbers)
    print("[TCP desired] pos:", _fmt(target_tcp_pos),
          " rpy:", _fmt(p.getEulerFromQuaternion(target_tcp_quat)),
          " quat:", _fmt(target_tcp_quat))

    # ----- command & step -----
    for _ in range(max(1, int(steps))):
        for ji, qi in zip(arm_idxs, q_arm):
            p.setJointMotorControl2(panda_id, ji, p.POSITION_CONTROL, targetPosition=qi, force=max_force)
        p.stepSimulation()

    return q_arm

def _parse_se3(T):
    """
    Returns (pos, quat) in world frame.
    Accepts:
      -Tensor (4x4)
    """
    # 4x4 homogeneous
    pos = T[:3, 3].tolist()
    Rm = T[:3, :3].unsqueeze(0)  # (1, 3, 3)
    quat = SO3_to_quaternion(Rm, ordering='xyzw')  # [x,y,z,w]
    quat = quat.tolist()
    return pos, quat

def get_single_pcd(obj_type: str,
                   root: str = "dataset/meshes",
                   num_pts: int = 1024,
                   scale: float = 8.0
                   ):
    """
    Load exactly one .obj from dataset/meshes/<obj_type>/, sample a point cloud,
    center it, (optionally) augment rotation, and return a (3, num_pts) torch tensor.
    """

    # Strictly stay in the category
    cat_dir = os.path.join(root, obj_type)
    if not os.path.isdir(cat_dir):
        raise ValueError(f"Unknown category '{obj_type}'. Expected folder: {cat_dir}")

    obj_files = sorted(glob.glob(os.path.join(cat_dir, "*.obj")))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in: {cat_dir}")

    # Deterministically choose one file
    mesh_path = obj_files[0]

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty or failed to load: {mesh_path}")

    if scale != 1.0:
        mesh.scale(1/scale, center=(0, 0, 0))

    # Sample point cloud, transpose to (3, N)
    pc = np.asarray(mesh.sample_points_uniformly(num_pts).points).T  # (3, N)

    # Center to mean (match your dataset centering)
    center = pc.mean(axis=1, keepdims=True)
    pc = pc - center

    return mesh_path, torch.tensor(pc, dtype=torch.float32)

def main(args, cfg):
    seed = cfg.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    # Setup model
    cfg.model.ode_solver.name = 'SE3_RK_mk_guide'
    model = get_model(cfg.model).to(cfg.device)
    obj_type = args.obj_type

    # Choose a single mesh and build a centered point cloud for the model
    obj_path, obj = get_single_pcd(obj_type)

    # ------------------ PyBullet simulator (robust load) ------------------
    try:
        p.connect(p.GUI); use_gui = True
    except Exception:
        p.connect(p.DIRECT); use_gui = False

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")

    # <<< Spawn Panda
    panda_id, name_to_j = spawn_panda(base_pos=(-0.6, 0.0, 0.0), base_yaw_deg=90.0)

    # Inspect mesh with Open3D to get center and scale
    mesh_o3d = o3d.io.read_triangle_mesh(obj_path)
    if mesh_o3d.is_empty():
        raise ValueError(f"Failed to load mesh for physics/visual: {obj_path}")

    aabb_min = mesh_o3d.get_min_bound()
    aabb_max = mesh_o3d.get_max_bound()
    size = (aabb_max - aabb_min)
    center = mesh_o3d.get_center()
    mesh_scale_factor = 0.2 / max(size)  # scale to fit in a 0.3 m cube
    mesh_scale = [mesh_scale_factor] * 3

    base_pos = [
        -center[0] * mesh_scale_factor,
        -center[1] * mesh_scale_factor,
        -center[2] * mesh_scale_factor + 0.2
    ]

    # --- VHACD convex decomposition ---
    out_prefix = os.path.splitext(obj_path)[0] + "_vhacd"
    p.vhacd(obj_path, out_prefix + ".obj", out_prefix + ".log",
            resolution=100000, concavity=0.0025, planeDownsampling=4,
            convexhullDownsampling=4, alpha=0.04, beta=0.05, maxNumVerticesPerCH=64)

    part_files = sorted(glob.glob(out_prefix + "_*.obj"))
    if not part_files:
        part_files = [out_prefix + ".obj"]

    col_id = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_MESH] * len(part_files),
        fileNames=part_files,
        meshScales=[[mesh_scale_factor]*3] * len(part_files)
    )

    vis_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=mesh_scale,
        rgbaColor=[0.9, 0.9, 0.9, 1.0]
    )
    if vis_id < 0:
        raise RuntimeError(f"createVisualShape failed for: {obj_path}")

    body_id = p.createMultiBody(
        baseMass=0, #Tune This GENE
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=base_pos,
        baseOrientation=[0, 0, 0, 1],
    )

    # (Optional) let the robot look at the object
    if use_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=0.9,
            cameraYaw=60,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.05]
        )

    guide_type = args.guide_type

    # Sim loop
    hz = 240
    SPACE = ord(' ')           # fallback if p.B3G_SPACE not present
    ESC   = 27                 # fallback if p.B3G_ESCAPE not present
    pc_scale = 8.0            # must match your dataset/model

    while True:
        p.stepSimulation()

        if use_gui:
            keys = p.getKeyboardEvents()

            # Fire once per press (KEY_WAS_TRIGGERED)
            space_pressed = (
                (SPACE in keys and (keys[SPACE] & p.KEY_WAS_TRIGGERED)) or
                (hasattr(p, "B3G_SPACE") and p.B3G_SPACE in keys and (keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED))
            )
            if space_pressed:
                # (A) object world pose (base)
                pos_base, quat_base = p.getBasePositionAndOrientation(body_id)
                R_base = np.array(p.getMatrixFromQuaternion(quat_base), dtype=float).reshape(3, 3)
                p_base = np.array(pos_base, dtype=float)

                # (B) object center in world
                c_local = np.array(center, dtype=float)                       # from Open3D, mesh local coords
                s_world = float(mesh_scale_factor)                            # world scale for the mesh in Bullet
                p_world_center = p_base + R_base @ (s_world * c_local)        # where the object-centered frame lives

                # (C) model grasp in object-centered frame (torch 4x4)
                results = get_grasp_pose(model, obj, cfg.device, guide_type)  # (1,4,4) or (4,4)
                T_objcenter_TCP = results.detach().cpu()
                if T_objcenter_TCP.ndim == 3 and T_objcenter_TCP.shape[0] == 1:
                    T_objcenter_TCP = T_objcenter_TCP[0]
                assert T_objcenter_TCP.shape == (4, 4)
                R_pred = T_objcenter_TCP[:3, :3].numpy()
                t_pred = T_objcenter_TCP[:3, 3].numpy()

                # (D) scale translation only
                # total scale from model's PC frame -> Bullet world is pc_scale (undo your 1/pc_scale shrink) then s_world
                s_total = s_world * pc_scale
                t_world_rel = s_total * t_pred

                # (E) compose world TCP
                R_world_TCP = R_base @ R_pred
                p_world_TCP = p_world_center + R_base @ t_world_rel

                # (F) build 4x4 (numpy) and send to IK mover
                T_world_TCP_np = np.eye(4, dtype=float)
                T_world_TCP_np[:3, :3] = R_world_TCP
                T_world_TCP_np[:3, 3] = p_world_TCP

                # if your _parse_se3 expects torch.Tensor:
                T_world_TCP_torch = torch.from_numpy(T_world_TCP_np).float()

                move_panda_tcp_to_SE3(
                    panda_id, name_to_j, T_world_TCP_torch,
                    tcp_offset_in_hand=(0.0, 0.0, 0.0),
                    tcp_rot_offset_rpy=(0.0, 0.0, 0.0),
                    steps=1
                )
                print("[space] moved TCP to new grasp pose")

            # optional: ESC to quit
            esc_pressed = (
                (ESC in keys and (keys[ESC] & p.KEY_WAS_TRIGGERED)) or
                (hasattr(p, "B3G_ESCAPE") and p.B3G_ESCAPE in keys and (keys[p.B3G_ESCAPE] & p.KEY_WAS_TRIGGERED))
            )
            if esc_pressed:
                break

            time.sleep(1.0 / hz)

    
    


def get_grasp_pose(model, obj, device, guide_type):
    # Initialize
    model.eval()
    with torch.no_grad():
        obj = obj.to(device)
        grasp_pos = model.single_guide_sample(obj, guide_type)
        return grasp_pos
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_result_path', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--guide_type', type=str, default='none')
    parser.add_argument('--device', default=1)
    parser.add_argument('--obj_type', default='Mug')

    args = parser.parse_args()

    # Load config
    config_filename = [file for file in os.listdir(args.train_result_path) if file.endswith('.yml')][0]

    cfg = OmegaConf.load(os.path.join(args.train_result_path, config_filename))
    
    #For Guided Sampling
    cfg.model.ode_solver.name = 'SE3_RK_mk_guide'
    # Setup checkpoint
    cfg.model.checkpoint = os.path.join(args.train_result_path, args.checkpoint)

    # Setup device
    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'
    main(args, cfg)
