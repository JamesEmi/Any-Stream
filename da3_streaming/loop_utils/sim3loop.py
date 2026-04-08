# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import time
from typing import List, Optional, Tuple
import numpy as np
import pypose as pp
import torch
from fastloop.solve_python import solve_system_py
from scipy.spatial.transform import Rotation as R
import gtsam

cpp_version = False
try:
    import sim3solve

    cpp_version = True
except Exception:
    print("Sim3solve of C++ Version failed, Will using Python Version.")


class Sim3LoopOptimizer:
    """
    Loop closure optimizer for sequences of Sim3 transformations

    Input:
    - sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]
      Each element is (s, R, t), where s is scalar scale, R is [3,3] rotation matrix,
      t is [3,] translation vector
    - loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]
      Each element is (i, j, (s, R, t)), representing a loop closure constraint
      from frame i to frame j

    Output:
    - Optimized sequential_transforms
    """

    def __init__(self, config, device="cpu"):
        self.device = device
        self.config = config
        self.solve_system_version = self.config["Loop"]["SIM3_Optimizer"][
            "lang_version"
        ]  # choose between 'python' and 'cpp'

        if not cpp_version:
            self.solve_system_version = "python"

    def numpy_to_pypose_sim3(self, s: float, R_mat: np.ndarray, t_vec: np.ndarray) -> pp.Sim3:
        """Convert numpy s,R,t to pypose Sim3"""
        q = R.from_matrix(R_mat).as_quat()  # [x,y,z,w]
        # pypose requires [t, q, s] format
        data = np.concatenate([t_vec, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))

    def pypose_sim3_to_numpy(self, sim3: pp.Sim3) -> Tuple[float, np.ndarray, np.ndarray]:
        """Convert pypose Sim3 to numpy s,R,t"""
        data = sim3.data.cpu().numpy()
        t = data[:3]
        q = data[3:7]  # [x,y,z,w]
        s = data[7]
        R_mat = R.from_quat(q).as_matrix()
        return s, R_mat, t

    @staticmethod
    def _gtsam_pose3_from_rt(R_mat, t_vec):
        """
        Build a gtsam.Pose3 from a 3x3 rotation matrix and a 3-vector translation.
        """
        return gtsam.Pose3(gtsam.Rot3(R_mat), gtsam.Point3(*t_vec))

    @staticmethod
    def _rt_from_gtsam_pose3(pose):
        """Extract (R: 3x3, t: 3,) numpy arrays from a gtsam.Pose3."""
        R_mat = pose.rotation().matrix()
        t_vec = np.asarray(pose.translation()) #.reshape(3)
        return R_mat, t_vec
    
    @staticmethod
    def _model_sim3_to_gps_pose3(s_model, R_model, t_model, s_g, R_g, t_g):
        """
        Apply the Umeyama similarity (s_g, R_g, t_g) to a model-frame Sim3 pose,
        producing a GPS-frame (R, t) pair (scale absorbed into the metric translation).

        Composition rule:
            (s_g, R_g, t_g) ∘ (s_m, R_m, t_m)
                = (s_g*s_m, R_g R_m, s_g R_g t_m + t_g)
        """
        R_gps = R_g @ R_model
        t_gps = s_g * (R_g @ t_model) + t_g
        return R_gps, t_gps


    def sequential_to_absolute_poses(
        self, sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]
    ) -> torch.Tensor:
        """
        Convert sequential relative transforms to absolute pose sequence
        S_01, S_12, S_23, ... -> T_0, T_1, T_2, T_3, ...
        Where T_i is the transform from world coordinate to frame i
        """
        len(sequential_transforms) + 1
        poses = []

        identity = pp.Sim3(
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], device=self.device)
        )
        poses.append(identity)

        current_pose = identity
        for s, R_mat, t_vec in sequential_transforms:
            rel_transform = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            current_pose = current_pose @ rel_transform
            poses.append(current_pose)

        return torch.stack(poses)

    def absolute_to_sequential_transforms(
        self, absolute_poses: pp.Sim3
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Convert absolute pose sequence back to sequential relative transforms
        T_0, T_1, T_2, ... -> S_01, S_12, S_23, ...
        """
        sequential_transforms = []
        n = absolute_poses.shape[0]

        for i in range(n - 1):
            rel_transform = absolute_poses[i].Inv() @ absolute_poses[i + 1]
            s, R_mat, t_vec = self.pypose_sim3_to_numpy(rel_transform)
            sequential_transforms.append((s, R_mat, t_vec))

        return sequential_transforms

    def SE3_to_Sim3(self, x: torch.Tensor) -> pp.Sim3:
        """Convert SE3 to Sim3 (add unit scale)"""
        ones = torch.ones_like(x[..., :1])
        out = torch.cat((x, ones), dim=-1)
        return pp.Sim3(out)

    def build_loop_constraints(
        self, loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build loop closure constraints"""
        if not loop_constraints:
            return (
                torch.empty(0, 8, device=self.device),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        loop_transforms = []
        ii_loop = []
        jj_loop = []

        for i, j, (s, R_mat, t_vec) in loop_constraints:
            loop_sim3 = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            loop_transforms.append(loop_sim3.data)
            ii_loop.append(i)
            jj_loop.append(j)

        dSloop = pp.Sim3(torch.stack(loop_transforms))
        ii_loop = torch.tensor(ii_loop, dtype=torch.long, device=self.device)
        jj_loop = torch.tensor(jj_loop, dtype=torch.long, device=self.device)

        return dSloop, ii_loop, jj_loop

    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian=False):
        """Compute residuals (modified from original code)"""

        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()

        pred_inv_poses = pp.Sim3(input_poses).Inv()

        n, _ = pred_inv_poses.shape
        if n > 1:
            kk = torch.arange(1, n, device=self.device)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv()
        else:
            kk = torch.empty(0, dtype=torch.long, device=self.device)
            ll = torch.empty(0, dtype=torch.long, device=self.device)
            dSij = pp.Sim3(torch.empty(0, 8, device=self.device))

        constants = (
            torch.cat((dSij.data, dSloop.data), dim=0) if dSloop.shape[0] > 0 else dSij.data
        )
        if constants.shape[0] > 0:
            constants = pp.Sim3(constants)
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            iii = torch.empty(0, dtype=torch.long, device=self.device)
            jjj = torch.empty(0, dtype=torch.long, device=self.device)
            resid = torch.empty(0, device=self.device)

        if not jacobian:
            return resid

        if constants.shape[0] > 0:

            def batch_jacobian(func, x):
                def _func_sum(*x):
                    return func(*x).sum(dim=0)

                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange

                return rearrange(torch.stack((b, c)), "N O B I -> N B O I", N=2)

            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0, device=self.device)
            J_Ginv_j = torch.empty(0, device=self.device)

        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)

    def optimize(
        self,
        sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]],
        loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
        position_constraints=None,
        max_iterations: int = None,
        lambda_init: float = None,
        t0_prior: Tuple[float, np.ndarray, np.ndarray] | None = None,
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Main optimization function

        Args:
            sequential_transforms: Input sequence of transforms
            loop_constraints: List of loop closure constraints
            position_constraints: Optional list of (chunk_k: int, p_k: np.ndarray shape (3,))
                translation-only anchor constraints from GPS. Each constrains only the
                translation of chunk k's absolute pose; rotation and scale are left free.
            max_iterations: Maximum iterations
            lambda_init: Initial lambda for L-M algorithm
            t0_prior: Optional (s, R, t) GPS-derived prior for the first pose (T_0).
                If provided, T_0 is initialized to this pose and pinned (not updated)
                throughout the optimization, fixing the gauge to the GPS start position.

        Returns:
            Optimized sequence of transforms
        """
        if max_iterations is None:
            max_iterations = self.config["Loop"]["SIM3_Optimizer"]["max_iterations"]
        if lambda_init is None:
            lambda_init = eval(self.config["Loop"]["SIM3_Optimizer"]["lambda_init"])

        input_poses = self.sequential_to_absolute_poses(sequential_transforms)

        # Pin T_0 to GPS start: override the identity initialisation with the prior.
        if t0_prior is not None:
            s0, R0, t0 = t0_prior
            t0_sim3 = self.numpy_to_pypose_sim3(s0, R0, t0)
            input_poses = input_poses.clone()
            input_poses[0] = t0_sim3.data

        dSloop, ii_loop, jj_loop = self.build_loop_constraints(loop_constraints)

        if len(loop_constraints) == 0 and not position_constraints:
            print("Warning: No constraints provided, returning original transforms")
            return sequential_transforms

        # Pre-convert position constraints to tensors
        if position_constraints:
            k_pos = torch.tensor(
                [k for k, _ in position_constraints], dtype=torch.long, device=self.device
            )
            p_tgt = torch.tensor(
                np.array([p for _, p in position_constraints], dtype=np.float64),
                dtype=torch.float32,
                device=self.device,
            )  # (M, 3)
        else:
            k_pos = None
            p_tgt = None

        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []

        print(
            f"Starting optimization with {len(sequential_transforms)} poses, "
            f"{len(loop_constraints)} loop constraints"
            + (f", {len(position_constraints)} position constraints" if position_constraints else "")
            + (f" [T0 pinned]" if t0_prior is not None else "")
        )

        # L-M loop
        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True
            )

            if resid.numel() == 0 and k_pos is None:
                print("No residuals to optimize")
                break

            seq_cost = resid.square().mean().item() if resid.numel() > 0 else 0.0

            # Compute position residuals and Jacobians via autograd
            if k_pos is not None:
                with torch.enable_grad():
                    g_k = Ginv[k_pos].detach().requires_grad_(True)

                    def _pos_translation(g):
                        return pp.Exp(g).Inv().translation()  # (M, 3)

                    t_k = _pos_translation(g_k)
                    pos_r = t_k - p_tgt  # (M, 3)

                    # Jacobian of translation w.r.t. g: (3, M, 7) -> permute -> (M, 3, 7)
                    J_pos_raw = torch.autograd.functional.jacobian(
                        lambda g: _pos_translation(g).sum(0), g_k, vectorize=True
                    )
                    J_pos = J_pos_raw.permute(1, 0, 2).detach()  # (M, 3, 7)
                    pos_r = pos_r.detach()

                pos_cost = pos_r.square().mean().item()
            else:
                pos_r = None
                J_pos = None
                pos_cost = 0.0

            current_cost = seq_cost + pos_cost
            residual_history.append(current_cost)

            try:  # Solve linear system
                begin_time = time.time()
                # Force Python solver when position constraints are active (cpp path lacks support)
                use_python = (self.solve_system_version == "python") or (k_pos is not None)
                if use_python:
                    delta_pose = solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1,
                        J_pos=J_pos, k_pos=k_pos, resid_pos=pos_r,
                    )
                elif self.solve_system_version == "cpp":
                    (delta_pose,) = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
                else:
                    print("Solver version has not been chosen! ('python' or 'cpp')")
                end_time = time.time()
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break

            # Keep T_0 pinned: zero out its update so it stays at the GPS prior.
            if t0_prior is not None:
                delta_pose = delta_pose.clone()
                delta_pose[0] = 0

            Ginv_tmp = Ginv + delta_pose

            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_seq_cost = new_resid.square().mean().item() if new_resid.numel() > 0 else 0.0
            if k_pos is not None:
                with torch.no_grad():
                    pos_r_new = pp.Exp(Ginv_tmp[k_pos]).Inv().translation() - p_tgt
                new_pos_cost = pos_r_new.square().mean().item()
            else:
                new_pos_cost = 0.0
            new_cost = new_seq_cost + new_pos_cost

            solver_label = "python" if use_python else self.solve_system_version

            # L-M
            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
                print(
                    f"Iteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} (accepted)",
                    end=" | ",
                )
            else:
                lmbda *= 2
                print(
                    f"Iteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} (rej)     ",
                    end=" | ",
                )  # more readible to accepted

            print(
                f"Time of solver ({solver_label}): \
                    {(end_time - begin_time)*1000:.4f} ms"
            )

            if (current_cost < 1e-5) and (itr >= 4):
                if len(residual_history) >= 5:
                    improvement_ratio = residual_history[-5] / residual_history[-1]
                    if improvement_ratio < 1.5:
                        print(f"Converged at iteration {itr}")
                        break

        optimized_absolute_poses = pp.Exp(Ginv).Inv()

        optimized_sequential = self.absolute_to_sequential_transforms(optimized_absolute_poses)

        print(
            f"Optimization completed. Final cost: \
                {residual_history[-1] if residual_history else 'N/A'}"
        )

        return optimized_sequential

    def optimize_gps(
        self,
        sequential_transforms,
        position_constraints, # list of (k, p_k_gps) in metric GPS frame. include chunk 0 to ANCHOR
        umeyama,               # (s_g, R_g, t_g) model→GPS similarity 
        max_iterations: int = None,
        lambda_init: float = None,
    ):
        if max_iterations is None:
            max_iterations = self.config["Loop"]["SIM3_Optimizer"]["max_iterations"]
        if lambda_init is None:
            lambda_init = eval(self.config["Loop"]["SIM3_Optimizer"]["lambda_init"])
        
        s_g, R_g, t_g = umeyama

        # build init abs poses in model frame -> project to GPS
        abs_poses_model = self.sequential_to_absolute_poses(sequential_transforms)
        n_chunks = abs_poses_model.shape[0]

        initial_R = []
        initial_t = []

        for k in range(n_chunks):
            s_m, R_m, t_m = self.pypose_sim3_to_numpy(pp.Sim3(abs_poses_model[k])) # compose umeyama with the model sim3 (s_g * s_m → drop into Pose3 translation)
            R_gps, t_gps = self._model_sim3_to_gps_pose3(s_m, R_m, t_m, s_g, R_g, t_g)
            initial_R.append(R_gps)
            initial_t.append(t_gps)

        # graph + values + initial Pose3 insertion
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        X = lambda k: gtsam.symbol('x', k)

        for k in range(n_chunks):
            initial.insert(X(k), self._gtsam_pose3_from_rt(initial_R[k], initial_t[k]))

        # gps factors
        if position_constraints:
            gps_sigma_t = self.config["Loop"]["SIM3_Optimizer"].get("gps_sigma_t", 0.1)
            fallback_noise = gtsam.noiseModel.Isotropic.Sigma(3, gps_sigma_t)

            for k, p_k_gps, cov_k in position_constraints:
                if cov_k is None:
                    noise_k = fallback_noise
                else:
                    cov_k = np.asarray(cov_k, dtype=np.float64).reshape(3, 3)
                    # Defensive PD guards: degenerate altitude variance + jitter.
                    if cov_k[2, 2] < 1e-9:
                        cov_k[2, 2] = 100.0 * max(cov_k[0, 0], cov_k[1, 1], 1e-6)
                    cov_k = cov_k + 1e-9 * np.eye(3)
                    noise_k = gtsam.noiseModel.Gaussian.Covariance(cov_k)

                graph.add(gtsam.GPSFactor(
                    X(k),
                    np.asarray(p_k_gps, dtype=np.float64),
                    noise_k,
                ))

        # between factors
        seq_sigma = self.config["Loop"]["SIM3_Optimizer"].get("seq_sigma", 0.01) #TODO: Tune these params
        seq_noise = gtsam.noiseModel.Isotropic.Sigma(6, seq_sigma)
        for k in range(n_chunks - 1):
            T_k = initial.atPose3(X(k))
            T_kp1 = initial.atPose3(X(k+1))
            S_kl = T_k.between(T_kp1)
            graph.add(gtsam.BetweenFactorPose3(X(k), X(k+1), S_kl, seq_noise))
            
        # optim loop
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        params.setlambdaInitial(lambda_init)
        # params.setVerbosityLM("SUMMARY")

        lm = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = lm.optimize()

        # Return per-chunk ABSOLUTE GPS-frame Sim3s as plain numpy tuples.
        # The chunk's metric scale is OUTSIDE the Pose3: it lives in s_g * s_k^m,
        # where s_k^m is the depth-model per-chunk scale from _align_chunks. PGO
        # cannot touch it (Pose3 has no scale handle), so we pull it back out of
        # the input absolutes and bake (s_g * s_k^m) into the returned tuple.
        # The caller is responsible for using these to render acc_* in GPS frame.
        optimised_abs_gps = []
        for k in range(n_chunks):
            pose_gps = result.atPose3(X(k))
            R_gps, t_gps = self._rt_from_gtsam_pose3(pose_gps)

            s_model_k, _, _ = self.pypose_sim3_to_numpy(pp.Sim3(abs_poses_model[k]))
            s_k_gps = float(s_g) * float(s_model_k)

            optimised_abs_gps.append((s_k_gps, R_gps, t_gps))

        return optimised_abs_gps

        
# ======== TEST CODE ========


def create_ring_transforms(num_poses=6, radius=5.0, rot_noise_deg=2.0):
    """Generate a ring of Sim3 transforms with rotation, adding slight rotational noise"""
    transforms = []
    angle_step = 2 * np.pi / num_poses

    for i in range(num_poses):
        angle = angle_step

        # Main rotation (around Z-axis)
        R_z = R.from_euler("z", angle, degrees=False)

        # Add slight rotational noise (Gaussian noise in degrees)
        noise_angles_deg = np.random.normal(loc=0.0, scale=rot_noise_deg, size=3)
        R_noise = R.from_euler("xyz", noise_angles_deg, degrees=True)

        # Combine rotations
        R_mat = (R_noise * R_z).as_matrix()

        # Translation: simulate a circular trajectory
        t = np.array([radius * np.sin(angle), radius * (1 - np.cos(angle)), 0.0])

        s = np.random.uniform(0.8, 1.2)

        transforms.append((s, R_mat, t))

    return transforms


def example_usage():
    optimizer = Sim3LoopOptimizer(solve_system_version="cpp")

    # Build rotating ring
    sequential_transforms = create_ring_transforms(num_poses=20, radius=3.0)

    # Add loop closure constraint: from frame 5 back to frame 0
    loop_constraints = [
        (20, 0, (1.0, np.eye(3), np.zeros(3)))  # Temporary unit loop for simulation
    ]

    # Trajectory before/after optimization
    input_abs_poses = optimizer.sequential_to_absolute_poses(sequential_transforms)
    optimized_transforms = optimizer.optimize(sequential_transforms, loop_constraints)
    optimized_abs_poses = optimizer.sequential_to_absolute_poses(optimized_transforms)

    def extract_xyz(pose_tensor):
        poses = pose_tensor.cpu().numpy()
        return poses[:, 0], poses[:, 1], poses[:, 2]

    x0, y0, z0 = extract_xyz(input_abs_poses)
    x1, y1, z1 = extract_xyz(optimized_abs_poses)

    # Visualize trajectory
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    plt.figure(figsize=(8, 6))
    plt.plot(x0, y0, "o--", label="Before Optimization")
    plt.plot(x1, y1, "o-", label="After Optimization")
    for i, j, _ in loop_constraints:
        plt.plot([x0[i], x0[j]], [y0[i], y0[j]], "r--", label="Loop (Before)" if i == 5 else "")
        plt.plot([x1[i], x1[j]], [y1[i], y1[j]], "g-", label="Loop (After)" if i == 5 else "")
    plt.gca().set_aspect("equal")
    plt.title("Sim3 Loop Closure Optimization (Rotating Ring)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    return optimized_transforms


if __name__ == "__main__":
    example_usage()
