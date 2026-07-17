"""
debug_pgo.py — step-by-step PGO debugger

Run from da3_streaming/:
    python debug_pgo.py

Each STAGE can be uncommented / commented to isolate a component.
ipdb breakpoints are dropped inline with `import ipdb; ipdb.set_trace()`.

Stages:
  1. Sparse-matrix Hessian update (Bug 1) — does A_mat[sl, sl] += dense actually work?
  2. Jacobian via vectorize=True (Bug 3)  — is J_pos correct?  fd-check vs autograd.
  3. Full tiny PGO smoke-test             — 5 chunks, 1 GPS drift, verify convergence.
  4. Real data hook                       — drop into an actual run mid-optimize().
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import pypose as pp
from scipy.sparse import coo_matrix
import ipdb

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Does sparse in-place update actually modify A_mat?
# ─────────────────────────────────────────────────────────────────────────────
def stage1_sparse_hessian_update():
    print("\n" + "="*60)
    print("STAGE 1: sparse A_mat[sl,sl] += dense — does it stick?")
    print("="*60)

    n = 3   # 3 nodes → 21 DOFs
    # Build a tiny sequential Jacobian: edge 0→1 and 1→2, identity blocks for simplicity
    # J has shape (n_edges * 7, n * 7)  =  (14, 21)
    J_data = np.ones((14, 21), dtype=np.float64)
    from scipy.sparse import csc_matrix
    J = csc_matrix(J_data)

    A_mat = (J.T @ J)   # shape (21, 21), sparse
    print(f"A_mat type after J.T@J: {type(A_mat).__name__}, format: {A_mat.format}")

    Jk = np.random.randn(3, 7)   # GPS Jacobian for one constraint
    rk = np.random.randn(3)

    k = 1   # constrain chunk 1
    sl = slice(k * 7, (k + 1) * 7)

    diag_before = A_mat.diagonal()[sl].copy()
    print(f"\nA_mat diagonal at sl BEFORE update:\n  {diag_before}")

    A_mat[sl, sl] += Jk.T @ Jk   # ← the line under suspicion

    diag_after = A_mat.diagonal()[sl].copy()
    print(f"A_mat diagonal at sl AFTER  update:\n  {diag_after}")

    expected_increase = np.diag(Jk.T @ Jk)
    actual_increase   = diag_after - diag_before
    print(f"Expected increase (diag(Jk.T@Jk)):\n  {expected_increase}")
    print(f"Actual   increase:\n  {actual_increase}")

    ok = np.allclose(actual_increase, expected_increase, atol=1e-10)
    print(f"\n>>> Sparse update {'WORKS ✓' if ok else 'FAILS ✗ — this is Bug 1!'}")

    if not ok:
        print("\n--- safe alternative ---")
        A_dense = A_mat.toarray()
        A_dense[sl, sl] += Jk.T @ Jk
        diag_dense = A_dense[sl][:, sl].diagonal()
        print(f"Dense diagonal after .toarray() update:\n  {diag_dense}")
        print(f"Matches expected: {np.allclose(diag_dense - diag_before, expected_increase)}")

    ipdb.set_trace()   # ← inspect A_mat, Jk, diag_before, diag_after yourself
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Is the Jacobian from vectorize=True correct?
# ─────────────────────────────────────────────────────────────────────────────
def stage2_jacobian_check():
    print("\n" + "="*60)
    print("STAGE 2: J_pos via vectorize=True vs finite-difference")
    print("="*60)

    # Build a small Ginv: 3 chunks with GPS constraints on all of them
    torch.manual_seed(42)
    M = 3
    g_k = (torch.randn(M, 7) * 0.1).requires_grad_(False)

    def _pos_translation(g):
        return pp.Exp(g).Inv().translation()   # (M, 3)

    # ── autograd Jacobian ────────────────────────────────────────────────────
    g_k_ad = g_k.detach().requires_grad_(True)
    with torch.enable_grad():
        J_raw_vTrue = torch.autograd.functional.jacobian(
            lambda g: _pos_translation(g).sum(0), g_k_ad, vectorize=True
        )
    J_vTrue = J_raw_vTrue.permute(1, 0, 2).detach()   # (M, 3, 7)
    print(f"J_pos shape (vectorize=True):  {J_vTrue.shape}")
    print(f"J_pos max abs (vectorize=True): {J_vTrue.abs().max().item():.6f}")
    print(f"J_pos[0] =\n{J_vTrue[0].numpy().round(4)}")

    g_k_ad2 = g_k.detach().requires_grad_(True)
    with torch.enable_grad():
        J_raw_vFalse = torch.autograd.functional.jacobian(
            lambda g: _pos_translation(g).sum(0), g_k_ad2, vectorize=False
        )
    J_vFalse = J_raw_vFalse.permute(1, 0, 2).detach()  # (M, 3, 7)
    print(f"\nJ_pos shape (vectorize=False): {J_vFalse.shape}")
    print(f"J_pos max abs (vectorize=False): {J_vFalse.abs().max().item():.6f}")
    print(f"J_pos[0] =\n{J_vFalse[0].numpy().round(4)}")

    # ── finite-difference Jacobian (ground truth) ────────────────────────────
    eps = 1e-4
    J_fd = torch.zeros(M, 3, 7)
    g0 = g_k.detach().clone()
    with torch.no_grad():
        t0 = _pos_translation(g0)   # (M, 3)
        for m in range(M):
            for l in range(7):
                g_plus  = g0.clone(); g_plus[m, l]  += eps
                g_minus = g0.clone(); g_minus[m, l] -= eps
                t_plus  = _pos_translation(g_plus)[m]    # (3,)
                t_minus = _pos_translation(g_minus)[m]   # (3,)
                J_fd[m, :, l] = (t_plus - t_minus) / (2 * eps)
    print(f"\nJ_pos[0] (finite-diff) =\n{J_fd[0].numpy().round(4)}")

    err_vTrue  = (J_vTrue  - J_fd).abs().max().item()
    err_vFalse = (J_vFalse - J_fd).abs().max().item()
    print(f"\nMax |J_autograd(vTrue)  − J_fd|: {err_vTrue:.2e}  {'✓' if err_vTrue < 1e-3 else '✗ Bug 3!'}")
    print(f"Max |J_autograd(vFalse) − J_fd|: {err_vFalse:.2e}  {'✓' if err_vFalse < 1e-3 else '✗ Bug 3!'}")

    ipdb.set_trace()   # ← inspect J_vTrue, J_vFalse, J_fd
    return err_vTrue, err_vFalse


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Full tiny-PGO smoke test
#   5 chunks in a straight line, chunk 3 has simulated drift.
#   GPS anchors every 2 chunks.  Verify optimizer pulls chunk 3 back.
# ─────────────────────────────────────────────────────────────────────────────
def stage3_tiny_pgo():
    print("\n" + "="*60)
    print("STAGE 3: tiny synthetic PGO (5 chunks, 1 drifted chunk)")
    print("="*60)
    from scipy.spatial.transform import Rotation as R_scipy

    # True positions: straight line x=0,1,2,3,4
    true_pos = np.array([[i, 0., 0.] for i in range(5)])

    # sim3_list: noiseless sequential transforms (step=1 along x, identity rotation, scale=1)
    # Relative: chunk k → chunk k+1 moves by [1,0,0] in the chunk-k frame
    step = np.array([1., 0., 0.])
    sim3_list = [(1.0, np.eye(3), step.copy()) for _ in range(4)]

    # Inject drift: chunk 3 is pulled sideways by [0, 0.5, 0] in *relative* transform
    # (so chunk 3 and 4 are at wrong y positions)
    sim3_list[2] = (1.0, np.eye(3), np.array([1., 0.5, 0.]))   # chunk 2→3: drift 0.5 in y

    # GPS anchor targets: true positions (no noise for this test)
    # We'll anchor every 2 chunks: k=0,2,4
    gps_every_k = 2
    # Umeyama is identity here (model frame == metric frame, scale=1)
    s_g, R_g, t_g = 1.0, np.eye(3), np.zeros(3)

    def gps_target(k):
        return true_pos[k]   # perfect GPS

    position_constraints = [
        (k, gps_target(k)) for k in range(0, 5, gps_every_k)
    ]
    print(f"GPS anchors at chunks: {[k for k,_ in position_constraints]}")
    print(f"GPS targets:\n{np.array([p for _,p in position_constraints])}")

    # ── run optimizer ────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    from loop_utils.sim3loop import Sim3LoopOptimizer

    cfg = {
        "Loop": {
            "SIM3_Optimizer": {
                "lang_version": "python",
                "max_iterations": 50,
                "lambda_init": "1e-4",
            }
        }
    }
    opt = Sim3LoopOptimizer(cfg, device="cpu")

    # Absolute poses before
    abs_before = opt.sequential_to_absolute_poses(sim3_list)
    pos_before = abs_before[:, :3].numpy()
    print(f"\nAbsolute translations BEFORE PGO:\n{pos_before}")
    print(f"True positions:\n{true_pos}")
    print(f"Initial GPS residuals: {pos_before[::gps_every_k] - true_pos[::gps_every_k]}")

    ipdb.set_trace()   # ← inspect abs_before, position_constraints, sim3_list

    optimized = opt.optimize(sim3_list, [], position_constraints=position_constraints)

    abs_after = opt.sequential_to_absolute_poses(optimized)
    pos_after = abs_after[:, :3].numpy()
    print(f"\nAbsolute translations AFTER PGO:\n{pos_after}")
    print(f"True positions:\n{true_pos}")
    final_residuals = pos_after[::gps_every_k] - true_pos[::gps_every_k]
    print(f"Final GPS residuals: {final_residuals}")
    print(f"Max abs GPS residual: {np.abs(final_residuals).max():.6f}")

    ok = np.abs(final_residuals).max() < 0.05
    print(f"\n>>> PGO convergence {'OK ✓' if ok else 'FAILED ✗ — check Bugs 1/2/3!'}")

    ipdb.set_trace()   # ← inspect pos_before, pos_after, true_pos


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Hook into a real run — add this to sim3loop.py optimize() to drop
#          a breakpoint at the first LM iteration of an actual pipeline call.
# ─────────────────────────────────────────────────────────────────────────────
STAGE4_PATCH = """
# In Sim3LoopOptimizer.optimize(), inside the LM loop, FIRST iteration only:
if itr == 0:
    import ipdb
    # Check these things:
    #   1. pos_r.shape, pos_r         → GPS residuals (should be small if Umeyama good)
    #   2. J_pos.shape, J_pos[0]      → Jacobian of translation (should be non-zero ~I for 3x7)
    #   3. resid.shape, resid.abs().max() → sequential residuals (should be ~0 at init)
    #   4. After solve: delta_pose.abs().max()  → step size
    #   5. After: (Ginv + delta_pose — inspect where chunk positions end up
    ipdb.set_trace()
"""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1,
                    help="Which stage to run: 1=sparse, 2=jacobian, 3=pgo, 123=all")
    args = ap.parse_args()

    stages = list(str(args.stage))
    if "1" in stages:
        stage1_sparse_hessian_update()
    if "2" in stages:
        stage2_jacobian_check()
    if "3" in stages:
        stage3_tiny_pgo()
