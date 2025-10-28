# reach_energy_cost.py
import numpy as np, torch, torch.nn as nn
# same 9D feature as in training: (x,y,z) + rot6D
def _rot6d(R): return np.concatenate([R[:,0], R[:,1]], 0)
def _feat_from_T(T): return np.asarray(np.concatenate([T[:3,3], _rot6d(T[:3,:3])],0), dtype=np.float32)

class _IRN(nn.Module):
    def __init__(self, h=128, depth=3, drop=0.1):
        super().__init__()
        layers=[nn.Linear(9,h), nn.ReLU(True)]
        for _ in range(depth-1):
            layers += [nn.Linear(h,h), nn.ReLU(True), nn.Dropout(drop)]
        self.f = nn.Sequential(*layers); self.out = nn.Linear(h,1)
    def forward(self,x): return torch.sigmoid(self.out(self.f(x))).squeeze(-1)

class EnergyCostIRN:
    """
    energy_cost:
        y = energy_cost(T_obj_TCP)         # torch scalar 0. (reachable) / 1. (unreachable)
        Tw = energy_cost.compose(T_obj_TCP) # (4,4) torch — object→world map
    """
    def __init__(self, ckpt_path, device, s_total):
        dev = torch.device(device if isinstance(device,str) else "cpu")
        self.net = _IRN().to(dev)
        state = torch.load(ckpt_path, map_location=dev)
        self.net.load_state_dict(state); self.net.eval()
        self.dev = dev
        self.s_total = float(s_total)

    def compose(self, T_obj: torch.Tensor) -> torch.Tensor:
        T = T_obj[0] if (T_obj.ndim==3 and T_obj.shape[0]==1) else T_obj
        Tw = T.clone().to(self.dev)
        Tw[:3,3] = Tw[:3,3]*self.s_total
        return Tw

    @torch.no_grad()
    def __call__(self, T_obj: torch.Tensor) -> torch.Tensor:
        Tw = self.compose(T_obj)
        x = torch.from_numpy(_feat_from_T(Tw.detach().cpu().numpy())).unsqueeze(0).to(self.dev)
        p = self.net(x)                         # prob(unreachable)
        y = (p >= 0.5).float()                  # 1 unreachable / 0 reachable
        return y[0]

def init_energy_cost_irn(ckpt_path, cfg_device, s_total):
    """1-liner for your main"""
    return EnergyCostIRN(ckpt_path, cfg_device, s_total)
