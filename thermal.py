import jax
import jax.numpy as jnp
import numpy as np
from config import ModelCfg, TrainCfg

# ---- Attention entropy S ----

def attn_entropy(w, eps=1e-9):
    ent = -jnp.sum(w * jnp.log(w + eps), axis=-1)
    return jnp.mean(ent)

def attn_entropy_per_position(w, eps=1e-9):
    ent = -jnp.sum(w * jnp.log(w + eps), axis=-1)  # (B,H,L)
    return jnp.mean(ent, axis=1)  # (B,L)

def entropies_from_list(attn_list):
    return jnp.array([attn_entropy(w) for w in attn_list])

# ---- Surprisal U ----

def surprisal_per_position(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
    return -target_log_probs

# ---- Discretizer: continuous S,U,F → bins ----

class SUFDiscretizer:
    def __init__(self, n_bins: int = 8, momentum: float = 0.99):
        self.n_bins = n_bins
        self.momentum = momentum
        self.s_min, self.s_max = 0.0, 6.0
        self.u_min, self.u_max = 0.0, 10.0
        self.f_min, self.f_max = -6.0, 10.0

    def update_bounds(self, S, U, F):
        m = self.momentum
        self.s_min = m * self.s_min + (1 - m) * float(np.min(S))
        self.s_max = m * self.s_max + (1 - m) * float(np.max(S))
        self.u_min = m * self.u_min + (1 - m) * float(np.min(U))
        self.u_max = m * self.u_max + (1 - m) * float(np.max(U))
        self.f_min = m * self.f_min + (1 - m) * float(np.min(F))
        self.f_max = m * self.f_max + (1 - m) * float(np.max(F))

    def discretize(self, val, vmin, vmax):
        normed = (val - vmin) / (vmax - vmin + 1e-8)
        normed = np.clip(normed, 0.0, 1.0 - 1e-6)
        return (normed * self.n_bins).astype(np.int32)

    def __call__(self, S, U, F):
        S, U, F = np.asarray(S), np.asarray(U), np.asarray(F)
        self.update_bounds(S, U, F)
        return self.discretize(S, self.s_min, self.s_max), \
               self.discretize(U, self.u_min, self.u_max), \
               self.discretize(F, self.f_min, self.f_max)

# ---- Build thermal inputs for a batch ----

def build_thermal_inputs(token_ids, boundary_positions, window_starts,
                         seq_len, prev_suf, discretizer, n_bins=8):
    B = len(window_starts)
    binS = np.zeros((B, seq_len), dtype=np.int32)
    binU = np.zeros((B, seq_len), dtype=np.int32)
    binF = np.zeros((B, seq_len), dtype=np.int32)
    suf_prev_batch = np.zeros((B, 3), dtype=np.float32)
    boundary_info = []

    for bi in range(B):
        start = int(window_starts[bi])
        end = start + seq_len

        # boundaries in this window
        bpos_in_window = [int(bp) for bp in boundary_positions if start <= bp < end]
        boundary_info.append(bpos_in_window)

        # prev SUF from boundary just before this window
        mask = boundary_positions < start
        if np.any(mask):
            prev_bp = int(boundary_positions[mask][-1])
            if prev_bp in prev_suf:
                s, u, f = prev_suf[prev_bp]
                suf_prev_batch[bi] = [s, u, f]

        # segment boundaries → fill bins per segment
        segment_starts = [0]
        segment_sufs = [suf_prev_batch[bi].tolist()]

        for bp in bpos_in_window:
            rel_pos = bp - start
            if 0 < rel_pos < seq_len:
                segment_starts.append(rel_pos)
                if bp in prev_suf:
                    segment_sufs.append(list(prev_suf[bp]))
                else:
                    segment_sufs.append(segment_sufs[-1])

        segment_starts.append(seq_len)
        for si in range(len(segment_starts) - 1):
            s_start = segment_starts[si]
            s_end = segment_starts[si + 1]
            s_val, u_val, f_val = segment_sufs[min(si, len(segment_sufs) - 1)]

            bs, bu, bf = discretizer(np.array([s_val]), np.array([u_val]), np.array([f_val]))
            binS[bi, s_start:s_end] = bs[0]
            binU[bi, s_start:s_end] = bu[0]
            binF[bi, s_start:s_end] = bf[0]

    return binS, binU, binF, suf_prev_batch, boundary_info

# ---- Update SUF cache after forward pass ----

def update_suf_cache(prev_suf, boundary_info, window_starts, S_per_pos, U_per_pos, beta=1.0):
    CONTEXT = 32
    for bi, bpos_list in enumerate(boundary_info):
        start = int(window_starts[bi])
        for bp in bpos_list:
            rel = bp - start
            seg_start = max(0, rel - CONTEXT)
            if rel <= seg_start:
                continue
            s_val = float(np.mean(S_per_pos[bi, seg_start:rel]))
            u_val = float(np.mean(U_per_pos[bi, seg_start:rel]))
            f_val = u_val - beta * s_val
            prev_suf[bp] = (s_val, u_val, f_val)
    return prev_suf