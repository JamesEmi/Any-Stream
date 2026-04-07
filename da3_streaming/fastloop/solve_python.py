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

import numpy as np
import torch
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


def solve_sparse(A: csc_matrix, b: np.ndarray, freen: int) -> np.ndarray:
    """Solve linear system A * delta = b, supports submatrix solving"""
    if freen < 0:
        return spsolve(A, b)
    else:
        A_sub = A[:freen, :freen].tocsc()
        b_sub = b[:freen]
        delta_sub = spsolve(A_sub, b_sub)
        delta = np.zeros_like(b)
        delta[:freen] = delta_sub
        return delta


def solve_system_py(
    J_Ginv_i: torch.Tensor,
    J_Ginv_j: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    res: torch.Tensor,
    ep: float,
    lm: float,
    freen: int,
    J_pos: torch.Tensor = None,
    k_pos: torch.Tensor = None,
    resid_pos: torch.Tensor = None,
) -> torch.Tensor:
    # Ensure all tensors are on CPU
    device = res.device
    J_Ginv_i = J_Ginv_i.cpu()
    J_Ginv_j = J_Ginv_j.cpu()
    ii = ii.cpu()
    jj = jj.cpu()
    res = res.clone().cpu()

    r = res.size(0)  # Number of edges
    n_from_edges = (max(ii.max().item(), jj.max().item()) + 1) if r > 0 else 0
    n_from_pos = (k_pos.max().item() + 1) if (k_pos is not None and len(k_pos) > 0) else 0
    n = max(n_from_edges, n_from_pos)  # Number of nodes

    res_vec = res.view(-1).numpy().astype(np.float64)

    rows, cols, data = [], [], []
    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()

    for x in range(r):
        i = ii_np[x]
        j = jj_np[x]
        if i == j:
            raise ValueError("Self-edges are not allowed")

        for k in range(7):
            for l in range(7):
                row_idx = x * 7 + k
                col_idx_i = i * 7 + l
                val_i = J_Ginv_i_np[x, k, l]
                rows.append(row_idx)
                cols.append(col_idx_i)
                data.append(val_i)

                col_idx_j = j * 7 + l
                val_j = J_Ginv_j_np[x, k, l]
                rows.append(row_idx)
                cols.append(col_idx_j)
                data.append(val_j)

    J = coo_matrix((data, (rows, cols)), shape=(r * 7, n * 7)).tocsc()

    b_vec = -J.T @ res_vec

    A_mat = J.T @ J

    # Add position constraint contributions: each (k, p_k) contributes J_k^T J_k to H
    # and -J_k^T r_k to b, where J_k is (3,7) and r_k is (3,).
    if J_pos is not None and k_pos is not None and len(k_pos) > 0:
        J_pos_np    = J_pos.cpu().numpy().astype(np.float64)      # (M, 3, 7)
        k_pos_np    = k_pos.cpu().numpy()                         # (M,)
        resid_pos_np = resid_pos.cpu().numpy().astype(np.float64) # (M, 3)
        for m in range(len(k_pos_np)):
            k  = int(k_pos_np[m])
            Jk = J_pos_np[m]       # (3, 7)
            rk = resid_pos_np[m]   # (3,)
            sl = slice(k * 7, (k + 1) * 7)
            A_mat[sl, sl] += Jk.T @ Jk   # rank-3 (7,7) contribution
            b_vec[sl]     -= Jk.T @ rk   # (7,) contribution

    diag = A_mat.diagonal()
    new_diag = diag * (1.0 + lm) + ep
    A_mat.setdiag(new_diag)

    freen_total = freen * 7
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)

    delta_tensor = torch.from_numpy(delta.astype(np.float32)).view(n, 7).to(device)
    return delta_tensor
