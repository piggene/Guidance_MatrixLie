import torch
import math
from copy import deepcopy

from utils.Lie import *


class EquiGraspFlow(torch.nn.Module):
    def __init__(self, p_uncond, guidance, init_dist, encoder, vector_field, ode_solver):
        super().__init__()

        self.p_uncond = p_uncond
        self.guidance = guidance

        self.init_dist = init_dist
        self.encoder = encoder
        self.vector_field = vector_field
        self.ode_solver = ode_solver
      


    def step(self, data, losses, split, optimizer=None):
        # Get data
        pc = data['pc']
        x_1 = data['Ts_grasp']

        # Get number of grasp poses in each batch and combine batched data
        nums_grasps = torch.tensor([len(Ts_grasp) for Ts_grasp in x_1], device=data['pc'].device)

        x_1 = torch.cat(x_1, dim=0)

        # Sample t and x_0
        t = torch.rand(len(x_1), 1).to(x_1.device)
        x_0 = self.init_dist(len(x_1), x_1.device)

        # Get x_t and u_t
        x_t, u_t = get_traj(x_0, x_1, t)

        # Forward
        v_t = self(pc, t, x_t, nums_grasps)

        # Calculate loss
        loss_mse = losses['mse'](v_t, u_t)

        loss = losses['mse'].weight * loss_mse

        # Backward
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        # Archive results
        results = {
            f'scalar/{split}/loss': loss.item(),
        }

        return results

    def forward(self, pc, t, x_t, nums_grasps):
        z = torch.zeros((len(pc), self.encoder.dims[-1], 3), device=pc.device)

        # Encode point cloud
        z = self.encoder(pc)

        # Repeat feature
        z = z.repeat_interleave(nums_grasps, dim=0)

        # Null condition
        mask_uncond = torch.bernoulli(torch.Tensor([self.p_uncond] * len(z))).to(bool)

        z[mask_uncond] = torch.zeros_like(z[mask_uncond])

        # Get vector
        v_t = self.vector_field(z, t, x_t)

        return v_t

    def sample(self, pc, nums_grasps):
        # Sample initial samples
        x_0 = self.init_dist(sum(nums_grasps), pc.device)
        self.X0SAMPLED = deepcopy(x_0)

        # Encode point cloud
        z = self.encoder(pc)

        # Repeat feature
        z = z.repeat_interleave(nums_grasps, dim=0)

        # Push-forward initial samples
        x_1_hat = self.ode_solver(z, x_0, self.guided_vector_field)[:, -1]

        # Batch x_1_hat
        x_1_hat = x_1_hat.split(nums_grasps.tolist())

        return x_1_hat

    def guided_vector_field(self, z, t, x_t):
        if torch.isnan(x_t).any():
            bad_idx = torch.isnan(x_t).nonzero(as_tuple=False)[0]  # first offendiNum_Grasp index
            bad_val = x_t.detach().cpu()[tuple(bad_idx.tolist())]
            raise FloatiNum_GraspPointError(
                f"x_t contains NaN at index {tuple(bad_idx.tolist())}; value={bad_val}."
            )
        v_t = (1 - self.guidance) * self.vector_field(torch.zeros_like(z), t, x_t) + self.guidance * self.vector_field(z, t, x_t)

        return v_t

    def guided_vector_field_with_guidance(self, z, x_1, t, x_t):
        # print(z.shape) #[N, 341, 3]
        # print(x_1.shape) #[D, 4, 4]
        # print(t.shape) #[N, 1]
        # print(x_t.shape) #[N,4,4]
        self.guidance = 2
        D = x_1.shape[0] if x_1 is not None else 100
        v_t = (1 - self.guidance) * self.vector_field(torch.zeros_like(z), t, x_t) + self.guidance * self.vector_field(z, t, x_t)
        # print(x_t[:,0,3].mean())
        if self.guide_type == 'mc':
            g_t =  self.guidance_vector_MC(x_1,t,x_t)
        elif self.guide_type == 'sim':
            g_t =  self.guidance_vector_sim_MC(D,t,x_t)
        elif self.guide_type == 'grad':
            g_t = self.guidance_vector_grad(t,x_t,v_t)
        elif self.guide_type == 'none':
            g_t = torch.zeros_like(v_t)
        else:   
            raise ValueError('Unknown guide type')
        norms = torch.linalg.vector_norm(g_t, dim=1, keepdim=True)  # (N, 1) L2 norms
        mean_norm = norms.mean()    
        print(f"Mean norm of g_t: {mean_norm.item()}")
        v_t = v_t + g_t
        return v_t
    
    # ===== inside your class =====

    def _tensor_fiNum_Grasperprint(self, T):
        # Fast identity-ish key; if you mutate T in-place this won't detect it.
        try:
            ptr = T.untyped_storage().data_ptr()
        except AttributeError:
            ptr = T.storage().data_ptr()  # older PyTorch
        return (T.shape, T.device.type, T.device.index, T.dtype, ptr)

    def _ensure_cache_for_x1(self, x_1):
        key = self._tensor_fiNum_Grasperprint(x_1)
        if getattr(self, "_x1_cache_key", None) == key:
            return  # cache is valid

        # (Re)build cache
        D = x_1.shape[0]
        device, dtype = x_1.device, x_1.dtype

        x_0 = self.init_dist(D, device)                          # (D,4,4)
        R0, R1 = x_0[:, :3, :3], x_1[:, :3, :3]                  # (D,3,3)
        p0, p1 = x_0[:, :3, 3],  x_1[:, :3, 3]                   # (D,3)

        # Precompute once per x_1:
        dR_log = log_SO3_fast(R0.transpose(-1, -2) @ R1)         # (D,3,3)
        w_b    = bracket_so3_fast(dR_log)                        # (D,3)
        dp     = p1 - p0                                         # (D,3)
        J_x1   = self.J(x_1).view(1, D)                          # (1,D)
        log_D  = math.log(D)

        # Stash
        self._x1_cache_key = key
        self._cache = {
            "D": D, "device": device, "dtype": dtype,
            "x_0": x_0, "R0": R0, "R1": R1, "p0": p0, "p1": p1,
            "dR_log": dR_log, "w_b": w_b, "dp": dp,
            "J_x1": J_x1, "log_D": log_D,
        }

    def _traj_from_cache(self, t):
        """
        Build x_t_mean and u_t for arbitrary t usiNum_Grasp cached x_0/x_1 parts.
        t: (N,1)
        returns: x_t_mean (N,D,4,4), u_t (N,D,6)
        """
        C = self._cache
        N, D = t.shape[0], C["D"]

        R0, p0 = C["R0"], C["p0"]                       # (D,3,3), (D,3)
        dR_log = C["dR_log"]                            # (D,3,3)
        dp     = C["dp"]                                # (D,3)
        w_b    = C["w_b"]                               # (D,3)

        # rotations at time t: R(t) = R0 * exp(t * log(R0^T R1))
        a   = (t.view(N,1,1,1) * dR_log.unsqueeze(0)).reshape(-1,3,3)
        R_t = (R0.unsqueeze(0) @ exp_so3_fast(a).view(N, D, 3, 3))   # (N,D,3,3)

        # translations (linear interp)
        p_t = p0.unsqueeze(0) + t.view(N,1,1) * dp.unsqueeze(0)      # (N,D,3)

        # assemble SE(3)
        x_t_mean = R_t.new_zeros((N, D, 4, 4))
        x_t_mean[..., :3, :3] = R_t
        x_t_mean[..., :3, 3]  = p_t
        x_t_mean[..., 3, 3]   = 1

        # u_t: rotate body twist w_b to space, reuse dp
        w_s = torch.einsum('dij,ndj->ndi', R0, w_b.unsqueeze(0).expand(N, -1, -1))
        u_t = x_t_mean.new_zeros((N, D, 6))
        u_t[..., :3] = w_s
        u_t[..., 3:] = dp.unsqueeze(0).expand(N, -1, -1)
        return x_t_mean, u_t

    def prime_x1_cache(self, x_1):
        """Optional: call once to build the cache explicitly."""
        self._ensure_cache_for_x1(x_1)

    def guidance_vector_MC(self, x_1, t, x_t):
        # Ensure we have cached pieces for this x_1 (build once, reuse later)
        self._ensure_cache_for_x1(x_1)
        C = self._cache
        D, log_D = C["D"], C["log_D"]

        # Use cached x_0/x_1-derived terms to build time-dependent trajectory
        x_t_mean, u_t = self._traj_from_cache(t)                    # (N,D,4,4), (N,D,6)

        # delta = inv(x_t_mean) * x_t
        inv_mean = inv_SE3_fast(x_t_mean.reshape(-1,4,4)).reshape(t.shape[0], D, 4, 4)
        delta = (inv_mean @ x_t.unsqueeze(1)).reshape(t.shape[0], D, 4, 4)

        xi = se3_log_to_vec_fast(delta)                             # (N,D,6)
        w = xi[..., :3]
        v = xi[..., 3:]

        sigma_w = 0.1
        sigma_v = 0.1
        inv_sw2 = 1.0 / (sigma_w * sigma_w)
        inv_sv2 = 1.0 / (sigma_v * sigma_v)

        #TODO could vuz some istability...
        # log_p_xt_z = -0.5 * (w.square().sum(-1) * inv_sw2 + v.square().sum(-1) * inv_sv2)  # (N,D)
        # log_p_xt   = torch.logsumexp(log_p_xt_z, dim=1) - log_D                             # (N,)
        # weights    = torch.exp(log_p_xt_z - log_p_xt.unsqueeze(1))                          # (N,D)

        # J = C["J_x1"]                                                                       # (1,D)

        # log_Z = torch.logsumexp(log_p_xt_z - log_p_xt.unsqueeze(1) - J, dim=1) - log_D      # (N,)
        # alpha = torch.exp(-J - log_Z.unsqueeze(1)) - 1                                       # (N,D)

        # g_t = ((alpha * weights).unsqueeze(-1) * u_t).mean(dim=1)                            # (N,6)
        
        log_p_xt_z = -0.5 * (w.square().sum(-1) * inv_sw2 + v.square().sum(-1) * inv_sv2)  # (N,D)

        # Numerically stable normalized weights: log_w = log softmax over D
        # (Sum over D of w_d == 1 exactly; avoids exploding sums)
        log_w = torch.log_softmax(log_p_xt_z, dim=1)                     # (N,D)
        w_norm = torch.exp(log_w)                                        # (N,D), sums to 1

        # Cached J(x_1) per particle
        # Expect shape (1,D) or (N,D) broadcastable to (N,D)
        J = C["J_x1"]                                                    # (1,D) or (N,D)

        # Compute log_Z = log ( E_{w}[ e^{-J} ] )
        # This is logsumexp(log_w - J) over D
        # Use clamping to keep it finite
        s = log_w - J                                                    # (N,D)
        s_max, _ = torch.max(s, dim=1, keepdim=True)                     # (N,1)
        s_shift = s - s_max                                              # (N,D)
        log_Z = s_max.squeeze(1) + torch.log(torch.sum(torch.exp(s_shift), dim=1) + 1e-12)  # (N,)

        # alpha_d = exp(-J_d - log_Z) - 1  (per particle)
        # Use expm1 for stability, clamp exponent to avoid overflow
        expo = (-J - log_Z.unsqueeze(1)).clamp(min=-60.0, max=60.0)      # (N,D)
        alpha = torch.expm1(expo)                                        # (N,D)  in (-1, +∞)

        # Guidance: sum_d (alpha_d * w_d * u_t_d)
        # (use sum, not mean; w_d already sums to 1)
        coeff = (alpha * w_norm).unsqueeze(-1)                            # (N,D,1)
        g_t = torch.sum(coeff * u_t, dim=1)                               # (N,6)

        # Final safety: replace any residual NaN/Inf (e.g., from degenerate inputs)
        g_t = torch.nan_to_num(g_t, nan=0.0, posinf=0.0, neginf=0.0)
        return g_t
        
    def guidance_vector_sim_MC(self, D, t, x_t):  
        # print(z.shape) #[N, 341, 3]
        # print(x_1.shape) #[D, 4, 4]
        # print(t.shape) #[N, 1]
        # print(x_t.shape) #[N,4,4]
        x_0 = self.init_dist(D, x_t.device)
        log_D  = math.log(D)
        x_1_sim, u_t = get_final(x_t, t, x_0) #N,D,4,4 / N,D,6
        J = self.J(x_1_sim)
        log_Z = torch.logsumexp(-J, dim=1) - log_D 
        alpha = torch.exp(-J - log_Z.unsqueeze(1)) - 1                                    
        g_t = (alpha * u_t).mean(dim=1) 
        return g_t  


    def left_triv_grad_from_matrix_grad_SE3(self, X, dJ_dX, alpha=1.0, beta=1.0, M=None, M_inv=None):
        """
        Convert Euclidean matrix gradient dJ/dX at X∈SE(3) into the left-trivialized se(3) gradient.

        Args:
            X:       (N,4,4) SE(3)
            dJ_dX:   (N,4,4) Euclidean matrix gradient wrt X (Frobenius inner product)
            alpha:   rotational weight (if using block-diag metric)
            beta:    translational weight (if using block-diag metric)
            M:       optional (6,6) SPD metric matrix at identity
            M_inv:   optional (6,6) inverse of M

        Returns:
            xi: (N,6) left-trivialized gradient in se(3) at X
        """
        # Transport Euclidean gradient to identity: Q = X^T * dJ/dX
        Q = X.transpose(-1, -2) @ dJ_dX  # (N,4,4)

        # Split into rotation/translation blocks
        Q_R = Q[..., :3, :3]           # (N,3,3)
        Q_t = Q[..., :3, 3]            # (N,3)

        # Only the skew part of Q_R couples to [ω]^
        skew = (Q_R - Q_R.transpose(-1, -2))          # (N,3,3), skew
        s = bracket_so3_fast(skew)                           # (N,3), vee(skew)

        # Frobenius identity: <Q_R, [ω]^>_F = 2 * vee(skew(Q_R)) · ω
        c = torch.cat([s, Q_t], dim=-1)               # (N,6)

        # Metric mapping: grad^L = M^{-1} c
        if M is None and M_inv is None:
            vals = torch.tensor([alpha, alpha, alpha, beta, beta, beta],
                                dtype=X.dtype, device=X.device)
            M = torch.diag(vals)
            M_inv = torch.diag(1.0 / vals)
        elif M_inv is None:
            M_inv = torch.inverse(M)

        # Broadcast M/M_inv over batch
        N = X.shape[0]
        Mb     = M.unsqueeze(0).expand(N, -1, -1)
        M_invb = M_inv.unsqueeze(0).expand(N, -1, -1)

        xi = (M_invb @ c.unsqueeze(-1)).squeeze(-1)         # (N,6)
        return xi


    def pullback_coadjoint(self, xi1, g, alpha=1.0, beta=1.0, M=None, M_inv=None):
        """
        xi_t = (Ad_{g^{-1}})^* xi1, coadjoint pullback with metric M.

        Args:
            xi1:  (N,6) left-trivialized grad at x1 = x_t @ g
            g:    (N,4,4) SE(3)
            alpha,beta or M,M_inv: metric

        Returns:
            xi_t: (N,6) left-trivialized grad wrt x_t
        """
        if M is None and M_inv is None:
            vals = torch.tensor([alpha, alpha, alpha, beta, beta, beta],
                                dtype=g.dtype, device=g.device)
            M = torch.diag(vals)
            M_inv = torch.diag(1.0 / vals)
        elif M_inv is None:
            M_inv = torch.inverse(M)

        N = g.shape[0]
        Mb     = M.unsqueeze(0).expand(N, -1, -1)
        M_invb = M_inv.unsqueeze(0).expand(N, -1, -1)

        g_inv = inv_SE3_fast(g)                 # (N,4,4)
        Ad_gi = large_adjoint(g_inv)            # (N,6,6)

        # (Ad_{g^{-1}})^* = M^{-1} Ad_{g^{-1}}^T M
        xi_t = (M_invb @ (Ad_gi.transpose(-1, -2) @ (Mb @ xi1.unsqueeze(-1)))).squeeze(-1)
        return xi_t


    def guidance_vector_grad(self, t, x_t, v_t, alpha=1.0, beta=1.0, M=None, M_inv=None):
        """
        (B) Differentiate w.r.t. x1 = x_t @ g, then pull back with the coadjoint.
        Returns: (N,6) left-trivialized guidance vector.
        """
        with torch.enable_grad():
            # 1 compose and freeze g (treated as constant for this grad)
            g  = exp_se3(v_t * (1.0 - t))          # (N,4,4)
            x1 = x_t @ g                           # (N,4,4)

            # 2 Re-leaf x1 so it definitely requires grad and does NOT backprop through x_t or g
            x1 = x1.detach().requires_grad_(True)
            # 3 Compute dJ/dx1
            J = self.J(x1)                         # shape (N,) or (N,1); must NOT call .item() or .detach() inside
            dJ_dx1, = torch.autograd.grad(
                outputs=J,
                inputs=x1,
                grad_outputs=torch.ones_like(J),   # same as J.sum(), but explicit
                create_graph=True,
                retain_graph=True
            )                                      # (N,4,4)
            # 4 Convert to left-trivialized grad at x1 (se(3))
            xi1 = self.left_triv_grad_from_matrix_grad_SE3(
                x1, dJ_dx1, alpha=alpha, beta=beta, M=M, M_inv=M_inv
            )                                      # (N,6)
            # 5 Pull back to x_t via coadjoint: xi_t = (Ad_{g^{-1}})^* xi1
            xi_t = self.pullback_coadjoint(
                xi1, g, alpha=alpha, beta=beta, M=M, M_inv=M_inv
            )                                      # (N,6)
            xi_t = xi_t/7000*(t)/(1-t).clip(0.05) 
        # 6 Guidance is negative gradient
        return -xi_t #-xi_t


    # def J(self, x):
    #     """
    #     Selects grasp poses with negative x only.
    #     Input: 
    #         x: (..., 4, 4) SE(3) transforms. Supports (D,4,4) or (N,D,4,4).
    #     Return:
    #         (..., 1) penalty. (D,1) if input is (D,4,4); (N,D,1) if (N,D,4,4).
    #     """
    #     import torch

    #     t = x[..., :3, 3]            # (..., 3)
    #     mask = t[..., 0] > 0         # (...)  True where x is positive (penalize)

    #     penalty = torch.where(
    #         mask,
    #         torch.full_like(t[..., 0], 40.0),   # penalty for positive x
    #         torch.zeros_like(t[..., 0])         # zero for x <= 0
    #     )
    #     return penalty.unsqueeze(-1)            # (..., 1)


    # # Alternative J function: cubic penalty for deviation from target x0    
    # def J(self, x, x0=0.1):
    #     """
    #     Penalize deviation of the grasp x-position from x0.

    #     Args:
    #         x: (..., 4, 4) SE(3) transforms. Supports (D,4,4) or (N,D,4,4).
    #         x0: target x-position (float or tensor), in the same units as t_x/8 below.

    #     Returns:
    #         penalty: (..., 1) tensor. (D,1) if input is (D,4,4), (N,D,1) if (N,D,4,4).
    #     """
    #     # translation vector
    #     t = x[..., :3, 3]  # (..., 3)

    #     # make x0 a tensor on the right device/dtype
    #     x0 = torch.as_tensor(x0, dtype=t.dtype, device=t.device)

    #     # deviation along x with the same scaling used before (scale = 8)
    #     dx = (t[..., 0] - x0 * 8).abs()  # (...)

    #     # cubic penalty
    #     penalty = (10.0 * dx).pow(3)  # (...)

    #     return penalty.unsqueeze(-1)  # (..., 1)




    def guided_sample(self, pc, grasp_pose, Num_Grasp, guide_type):
        #Run guided-sampliNum_Grasp
        x_1_hat_rot = torch.zeros((len(pc), Num_Grasp, 4, 4), device=pc.device)
        self.guide_type = guide_type
        for k, (mesh, grasp) in enumerate(zip(pc, grasp_pose)):
            #sample x_1
            num_samples = 10000 # 10000 this is number of sample used for g_t MC, and sim-MC methods
            indices = torch.randint(0, grasp.shape[0], (num_samples,), device=grasp.device)
            x_1 = grasp[indices]  # shape: [D, 4, 4]

            #get latent vector
            z = self.encoder(mesh.unsqueeze(0))
            z = z.repeat_interleave(Num_Grasp, dim=0)

            #sample x_0 (this is for sampling trajectory not guidance...)
            x_0 = self.init_dist(Num_Grasp, pc.device)
            self.X0SAMPLED = deepcopy(x_0)
            # Push-forward initial samples
            x_1_hat = self.ode_solver(z, x_0, x_1, self.guided_vector_field_with_guidance)[:, -1]

            # Batch x_1_hat
            # x_1_hat = x_1_hat.split(nums_grasps.tolist())
            x_1_hat_rot[k,:,:,:] = x_1_hat

        return x_1_hat_rot
    
    def single_guide_sample(self, pc, guide_type, energy_cost= None):
        z =self.encoder(pc.unsqueeze(0))
        self.guide_type = guide_type
        x_0 = self.init_dist(1, pc.device)
        if not callable(energy_cost):
            raise TypeError("fn must be callable")
        self.J = energy_cost
        x_1_ = self.ode_solver(z, x_0, None, self.guided_vector_field_with_guidance)[:, -1]
        return x_1_


def batch_covariance(w, v):
    """
    w, v: Tensors of shape [N, D, 3]
    Returns: covariance matrix of shape [N, 3, 3]
    """
    w_mean = w.mean(dim=1, keepdim=True)  # [N, 1, 3]
    v_mean = v.mean(dim=1, keepdim=True)  # [N, 1, 3]

    w_centered = w - w_mean               # [N, D, 3]
    v_centered = v - v_mean               # [N, D, 3]

    # Compute covariance: [N, 3, D] @ [N, D, 3] => [N, 3, 3]
    cov = torch.matmul(w_centered.transpose(1, 2), v_centered) / (w.shape[1] - 1)

    return cov.mean(dim=0)  # Shape: [N, 3, 3]


def get_traj(x_0, x_1, t):
    # Get rotations
    R_0 = x_0[:, :3, :3]
    R_1 = x_1[:, :3, :3]

    # Get translations
    p_0 = x_0[:, :3, 3]
    p_1 = x_1[:, :3, 3]

    # Get x_t
    x_t = torch.eye(4).repeat(len(x_1), 1, 1).to(x_1)
    x_t[:, :3, :3] = (R_0 @ exp_so3(t.unsqueeze(2) * log_SO3(inv_SO3(R_0) @ R_1)))
    x_t[:, :3, 3] = p_0 + t * (p_1 - p_0)

    # Get u_t
    u_t = torch.zeros(len(x_1), 6).to(x_1)
    u_t[:, :3] = bracket_so3(log_SO3(inv_SO3(R_0) @ R_1))
    u_t[:, :3] = torch.einsum('bij,bj->bi', R_0, u_t[:, :3])    # Convert w_b to w_s
    u_t[:, 3:] = p_1 - p_0

    return x_t, u_t

def get_traj2(x_0, x_1, t):
    N, D = t.shape[0], x_1.shape[0]
    device, dtype = x_1.device, x_1.dtype

    R0 = x_0[..., :3, :3]           # (D,3,3)
    R1 = x_1[..., :3, :3]
    p0 = x_0[..., :3, 3]            # (D,3)
    p1 = x_1[..., :3, 3]

    # rotations aloNum_Grasp the geodesic
    dR_log = log_SO3_fast(R0.transpose(-1,-2) @ R1)                # (D,3,3)
    a = (t.view(N,1,1,1) * dR_log.unsqueeze(0)).reshape(-1,3,3)    # (N*D,3,3)
    R_t = (R0.unsqueeze(0) @ exp_so3_fast(a).view(N, D, 3, 3))     # (N,D,3,3)

    # translations (linear interp)
    p_t = p0.unsqueeze(0) + t.view(N,1,1) * (p1.unsqueeze(0) - p0.unsqueeze(0))  # (N,D,3)

    # build x_t
    x_t = x_1.new_zeros((N, D, 4, 4))
    x_t[..., :3, :3] = R_t
    x_t[..., :3, 3]  = p_t
    x_t[..., 3, 3]   = 1

    # body twist -> space twist
    w_b = bracket_so3_fast(log_SO3_fast(R0.transpose(-1,-2) @ R1)).unsqueeze(0).expand(N, -1, -1)  # (N,D,3)
    w_s = torch.einsum('dij,ndj->ndi', R0, w_b)
    u_t = x_1.new_zeros((N, D, 6))
    u_t[..., :3] = w_s
    u_t[..., 3:] = (p1 - p0).unsqueeze(0).expand(N, -1, -1)
    return x_t, u_t

def get_final(x_t, t, x_0):
    # Shapes
    # x_0: (D,4,4), x_t: (N,4,4), t: (N,)
    N, D = x_t.shape[0], x_0.shape[0]
    R0 = x_0[..., :3, :3]           # (D,3,3)
    Rt = x_t[..., :3, :3]           # (N,3,3)
    p0 = x_0[..., :3, 3]            # (D,3)
    pt = x_t[..., :3, 3]            # (N,3)

    # rotations: log on pairwise R0^T @ Rt  → (N,D,3,3)
    dR = R0.transpose(-1, -2).unsqueeze(0) @ Rt.unsqueeze(1)          # (N,D,3,3)
    dR_log = log_SO3_fast(dR)                                         # (N,D,3,3)

    # geodesic step: scale the log by 1/t, then exp back and left-multiply by R0
    a = (1.0 / t.view(N, 1, 1, 1)) * dR_log                           # (N,D,3,3)
    R_1 = R0.unsqueeze(0) @ exp_so3_fast(a)                           # (N,D,3,3)

    # translations: linear interp per (N,D)
    p_1 = p0.unsqueeze(0) + (1.0 / (t.view(N, 1, 1)+0.025)) * (pt.unsqueeze(1) - p0.unsqueeze(0))  # (N,D,3)

    # build x_1
    x_1 = x_0.new_zeros((N, D, 4, 4))
    x_1[..., :3, :3] = R_1
    x_1[..., :3, 3]  = p_1
    x_1[..., 3, 3]   = 1


    # body twist -> space twist
    # log(R0^T @ R_1): (N,D,3,3) → bracket_so3_fast → (N,D,3)
    w_b = bracket_so3_fast(log_SO3_fast(R0.transpose(-1, -2).unsqueeze(0) @ R_1))  # (N,D,3)
    # rotate twists to space frame using R0 (D,3,3)
    w_s = torch.einsum('dij,ndj->ndi', R0, w_b)                                      # (N,D,3)

    u_t = x_0.new_zeros((N, D, 6))
    u_t[..., :3] = w_s
    u_t[..., 3:] = p_1 - p0.unsqueeze(0)                                            # (N,D,3)

    return x_1, u_t
