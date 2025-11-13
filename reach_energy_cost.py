import numpy as np
import torch
import torch.nn as nn

# -------- feature helpers (torch, vectorized) --------
def _rot6d_torch(R: torch.Tensor) -> torch.Tensor:
    # R: (B, 3, 3) -> (B, 6) using first two columns
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)

def _feat_from_T_torch(T: torch.Tensor) -> torch.Tensor:
    # T: (B, 4, 4) -> feats: (B, 9) = (x,y,z) + rot6d
    t = T[:, :3, 3]               # (B, 3)
    R = T[:, :3, :3]              # (B, 3, 3)
    rot6 = _rot6d_torch(R)        # (B, 6)
    return torch.cat([t, rot6], dim=1)  # (B, 9)

# -------- tiny IRN --------
class _IRN(nn.Module):
    def __init__(self, h=128, depth=3, drop=0.1):
        super().__init__()
        layers = [nn.Linear(9, h), nn.ReLU(True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.ReLU(True), nn.Dropout(drop)]
        self.f = nn.Sequential(*layers)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        # x: (B, 9) -> (B, 1)  (keep last dim = 1; do NOT squeeze)
        return torch.sigmoid(self.out(self.f(x)))

class EnergyCostIRN:
    """
    energy_cost(T_obj_TCP):
        - Accepts shape (..., 4, 4)
        - Returns shape (..., 1)  where output = 1.0 (unreachable), 0.0 (reachable)

    compose(T_obj_TCP):
        - Scales translation by s_total (object->world) and preserves shape (..., 4, 4)
    """
    def __init__(self, ckpt_path, device, s_total, base_pos, distance=False, prev_pos=None):
        self.dev = torch.device(device if isinstance(device, str) else "cpu")
        self.net = _IRN().to(self.dev)
        state = torch.load(ckpt_path, map_location=self.dev)
        self.net.load_state_dict(state)
        self.net.eval()
        self.s_total = float(s_total)
        self.obj_pos = base_pos.to(self.dev)  # (4,4) torch.Tensor
        self.distance = distance
        self.prev_pos = torch.tensor(prev_pos, dtype=torch.float32, device=self.dev) if prev_pos is not None else None
        
    def _flatten_T(self, T_obj: torch.Tensor):
        """
        Input:  T_obj (..., 4, 4)
        Return: T_flat (B, 4, 4), orig_shape (tuple of leading dims), B (int)
        """
        if T_obj.ndim < 2 or T_obj.shape[-2:] != (4, 4):
            raise ValueError(f"Expected (..., 4, 4), got {tuple(T_obj.shape)}")
        orig_shape = T_obj.shape[:-2]
        B = int(np.prod(orig_shape)) if len(orig_shape) > 0 else 1
        T_flat = T_obj.reshape(B, 4, 4).to(self.dev)
        return T_flat, orig_shape, B

    def compose(self, T_obj: torch.Tensor) -> torch.Tensor:
        """
        Scale translation to world. Shape-preserving.
        """
        T_flat, orig, _ = self._flatten_T(T_obj)
        T_world = T_flat.clone()
        T_world[:, :3, 3] = T_world[:, :3, 3] * self.s_total
        T_world[:, :4, :4] = self.obj_pos @ T_world[:, :4, :4]
        return T_world.reshape(*orig, 4, 4)

    @torch.no_grad()
    def __call__(self, T_obj: torch.Tensor) -> torch.Tensor:
        """
        Returns (..., 1) with 1 = unreachable, 0 = reachable
        """
        # compose & flatten
        T_world = self.compose(T_obj)                 # (..., 4, 4)
        T_flat, orig, _ = self._flatten_T(T_world)    # (B, 4, 4)

        # features (B, 9)
        feats = _feat_from_T_torch(T_flat)            # (B, 9)

        # prob(unreachable) in (B, 1)
        probs_unreach = self.net(feats)               # (B, 1)

        # threshold to {0,1} but KEEP last dim = 1
        y = (probs_unreach >= 0.5).float()            # (B, 1)

        # reshape back to (..., 1)
        if self.distance:
            dist2 = ((T_world[...,:3,3] - self.prev_pos)**2).sum(dim=-1, keepdim=True) 
            # print(dist2.shape)# (B, 1)
            out = 100.0*dist2 + 100.0 * y                   # (B, 1)
            return out.reshape(*orig, 1)   
        else:
            return y.reshape(*orig, 1)*100

def init_energy_cost_irn(ckpt_path, cfg_device, s_total, base_pos, distance=False, current_pos=None):
    return EnergyCostIRN(ckpt_path, cfg_device, s_total, base_pos, distance=distance, prev_pos=current_pos)
