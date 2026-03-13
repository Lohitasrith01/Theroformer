"""
Training loop v3: joint thermal learning with full gradient flow.
Vectorized boundary pairs, Huber loss, normalized targets, thermal warmup.
"""

import os
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from config import ModelCfg, TrainCfg
from model import ThermoTransformerLM, TauHead, ThermalPredictor
from thermal import (
    attn_entropy_per_position, surprisal_per_position, entropies_from_list,
    SUFDiscretizer, build_thermal_inputs, update_suf_cache,
)
from data import get_batch

# ---- Checkpoint ----

def save_checkpoint(ckpt_dir, step, params, opt_state, prev_suf, discretizer, logs):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:06d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "step": step,
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
            "prev_suf": prev_suf,
            "discretizer_state": {
                "s_min": discretizer.s_min, "s_max": discretizer.s_max,
                "u_min": discretizer.u_min, "u_max": discretizer.u_max,
                "f_min": discretizer.f_min, "f_max": discretizer.f_max,
            },
            "logs": logs,
        }, f)
    with open(os.path.join(ckpt_dir, "latest.txt"), "w") as f:
        f.write(f"step_{step:06d}.pkl")
    print(f"[ckpt] Saved step {step}")

def load_latest_checkpoint(ckpt_dir):
    latest_file = os.path.join(ckpt_dir, "latest.txt")
    if not os.path.exists(latest_file):
        return None
    with open(latest_file) as f:
        fname = f.read().strip()
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"[ckpt] Loaded step {ckpt['step']}")
    return ckpt

# ---- LR schedule (your version) ----

def make_lr_schedule(tcfg):
    warmup_steps = max(int(tcfg.warmup_steps), 1)
    total_steps = max(int(tcfg.total_steps), warmup_steps + 1)

    def lr_fn(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup_lr = tcfg.peak_lr * (step / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = jnp.clip(progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay_lr = tcfg.min_lr + (tcfg.peak_lr - tcfg.min_lr) * cosine
        return jnp.where(step < warmup_steps, warmup_lr, decay_lr)

    return lr_fn

# ---- Boundary pair pre-computation ----

MAX_PAIRS = 16

def build_boundary_pairs(x_np, seq_len, min_seg=5):
    B = x_np.shape[0]
    BOUNDARY_ID = 3

    bp_starts = np.full((B, MAX_PAIRS), -1, dtype=np.int32)
    bp_ends = np.full((B, MAX_PAIRS), -1, dtype=np.int32)
    pair_mask = np.zeros((B, MAX_PAIRS), dtype=bool)

    for bi in range(B):
        bpos = np.where(x_np[bi] == BOUNDARY_ID)[0]
        bpos = bpos[bpos < seq_len - min_seg]
        count = 0
        for i in range(len(bpos) - 1):
            if bpos[i + 1] - bpos[i] >= min_seg and count < MAX_PAIRS:
                bp_starts[bi, count] = bpos[i]
                bp_ends[bi, count] = bpos[i + 1]
                pair_mask[bi, count] = True
                count += 1

    return bp_starts, bp_ends, pair_mask

# ---- Training ----

def train(mcfg: ModelCfg, tcfg: TrainCfg, data: dict):
    token_ids = data["token_ids"]
    boundary_positions = data["boundary_positions"]
    train_starts = data["train_starts"]
    tok = data["tok"]

    mcfg.vocab_size = tok.get_vocab_size()

    rng = jax.random.PRNGKey(tcfg.seed)
    rng, k1, k2, k3 = jax.random.split(rng, 4)

    thermo_model = ThermoTransformerLM(cfg=mcfg, deterministic=False)
    tau_head = TauHead(n_heads=mcfg.n_heads)
    thermal_pred = ThermalPredictor()

    B_init = 2
    x_init = jnp.ones((B_init, tcfg.seq_len), dtype=jnp.int32)
    bin0 = jnp.zeros((B_init, tcfg.seq_len), dtype=jnp.int32)
    tau0 = jnp.ones((B_init, mcfg.n_heads), dtype=jnp.float32)

    params_m = thermo_model.init(k1, x_init, bin0, bin0, bin0, tau0)["params"]
    params_t = tau_head.init(k2, jnp.zeros((B_init, 3)))["params"]
    params_p = thermal_pred.init(k3, jnp.zeros((B_init, mcfg.d_model)))["params"]
    params = {"m": params_m, "t": params_t, "p": params_p}

    # ---- Optimizer (your version: real LR schedule + grad clipping) ----
    lr_schedule = make_lr_schedule(tcfg)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=tcfg.weight_decay,
        ),
    )
    opt_state = tx.init(params)

    discretizer = SUFDiscretizer(n_bins=mcfg.n_bins)
    prev_suf = {}

    logs = {"step": [], "lm_loss": [], "thermal_loss": [], "total_loss": [],
            "tau_mean": [], "S_layers": [], "lr": []}

    start_step = 0
    ckpt = load_latest_checkpoint(str(tcfg.ckpt_dir))
    if ckpt is not None:
        start_step = ckpt["step"] + 1
        params = jax.device_put(ckpt["params"])
        opt_state = jax.device_put(ckpt["opt_state"])
        prev_suf = ckpt["prev_suf"]
        ds = ckpt["discretizer_state"]
        discretizer.s_min, discretizer.s_max = ds["s_min"], ds["s_max"]
        discretizer.u_min, discretizer.u_max = ds["u_min"], ds["u_max"]
        discretizer.f_min, discretizer.f_max = ds["f_min"], ds["f_max"]
        logs = ckpt["logs"]
        print(f"[train] Resuming from step {start_step}")

    # ================================================================
    # Your JIT-compiled joint train step
    # ================================================================

    def masked_mean_std(x, mask, eps=1e-6):
        mask = mask[..., None]
        denom = jnp.maximum(mask.sum(axis=0), 1.0)
        mean = (x * mask).sum(axis=0) / denom
        var = ((x - mean) ** 2 * mask).sum(axis=0) / denom
        std = jnp.sqrt(var + eps)
        return mean, std

    @jax.jit
    def train_step(params, opt_state, rng, step, x, y,
                   binS, binU, binF, suf_prev_arr,
                   bp_starts, bp_ends, pair_mask):
        B_sz, L = x.shape
        P = bp_starts.shape[1]

        def loss_fn(p):
            pm, pt, pp = p["m"], p["t"], p["p"]

            tau = tau_head.apply({"params": pt}, suf_prev_arr)

            logits, attn_list, hidden = thermo_model.apply(
                {"params": pm}, x, binS, binU, binF, tau,
                rngs={"dropout": rng},
            )

            # LM loss
            lm_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, mcfg.vocab_size), y.reshape(-1),
            ).mean()

            # Positionwise thermal signals
            last_attn = attn_list[-1] if attn_list else None
            S_pos = (attn_entropy_per_position(last_attn)
                     if last_attn is not None
                     else jnp.zeros_like(x, dtype=jnp.float32))
            U_pos = surprisal_per_position(logits, y)

            # Safe boundary handling
            bp_s = jnp.clip(bp_starts, 0, L - 2)
            bp_e = jnp.clip(bp_ends, 1, L - 1)
            valid = pair_mask.astype(jnp.float32)

            # Hidden at boundary start
            batch_idx = jnp.arange(B_sz)[:, None]
            h_bp = hidden[batch_idx, bp_s, :]

            # Predictor
            h_bp_flat = h_bp.reshape(B_sz * P, h_bp.shape[-1])
            pred_flat = thermal_pred.apply({"params": pp}, h_bp_flat)
            pred = pred_flat.reshape(B_sz, P, 3)

            # Future segment masks
            positions = jnp.arange(L)[None, None, :]
            seg_mask = ((positions > bp_s[:, :, None]) &
                        (positions < bp_e[:, :, None]) &
                        pair_mask[:, :, None])
            seg_mask_f = seg_mask.astype(jnp.float32)
            seg_count = jnp.maximum(seg_mask_f.sum(axis=-1), 1.0)

            S_exp = S_pos[:, None, :]
            U_exp = U_pos[:, None, :]

            target_S = (S_exp * seg_mask_f).sum(axis=-1) / seg_count
            target_U = (U_exp * seg_mask_f).sum(axis=-1) / seg_count
            target_F = target_U - tcfg.beta * target_S

            target = jnp.stack([target_S, target_U, target_F], axis=-1)

            # Normalize targets
            target_flat = target.reshape(B_sz * P, 3)
            pred_flat_r = pred.reshape(B_sz * P, 3)
            valid_flat = valid.reshape(B_sz * P)

            tgt_mean, tgt_std = masked_mean_std(target_flat, valid_flat)
            target_norm = (target_flat - tgt_mean) / tgt_std
            pred_norm = (pred_flat_r - tgt_mean) / tgt_std

            # Huber loss
            huber = optax.huber_loss(pred_norm, jax.lax.stop_gradient(target_norm), delta=1.0)
            huber = huber.mean(axis=-1)
            thermal_loss = (huber * valid_flat).sum() / jnp.maximum(valid_flat.sum(), 1.0)

            # Thermal warmup
            thermal_warmup = getattr(tcfg, "thermal_warmup_steps", 0)
            if thermal_warmup > 0:
                therm_scale = tcfg.lambda_thermal * jnp.clip(
                    step / float(thermal_warmup), 0.0, 1.0)
            else:
                therm_scale = jnp.asarray(tcfg.lambda_thermal, dtype=jnp.float32)

            total_loss = lm_loss + therm_scale * thermal_loss

            S_layers = (entropies_from_list(attn_list)
                        if attn_list
                        else jnp.zeros(len(mcfg.capture_layers), dtype=jnp.float32))
            tau_mean = jnp.mean(tau, axis=0)

            aux = {
                "lm_loss": lm_loss,
                "thermal_loss": thermal_loss,
                "total_loss": total_loss,
                "tau_mean": tau_mean,
                "S_layers": S_layers,
                "S_pos": S_pos,
                "U_pos": U_pos,
                "lr": lr_schedule(step),
            }
            return total_loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, aux

    # ================================================================
    # Training loop
    # ================================================================

    print(f"[train] Starting from step {start_step} → {tcfg.total_steps}")
    tcfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in tqdm(range(start_step, tcfg.total_steps), initial=start_step, total=tcfg.total_steps):
        x_np, y_np, starts_np = get_batch(train_starts, tcfg.batch_size, token_ids, tcfg.seq_len)

        binS, binU, binF, suf_prev, boundary_info = build_thermal_inputs(
            token_ids, boundary_positions, starts_np, tcfg.seq_len,
            prev_suf, discretizer, mcfg.n_bins,
        )

        bp_starts_np, bp_ends_np, pair_mask_np = build_boundary_pairs(x_np, tcfg.seq_len)

        rng, sub = jax.random.split(rng)
        params, opt_state, aux = train_step(
            params, opt_state, sub,
            jnp.array(step, dtype=jnp.int32),
            jnp.array(x_np), jnp.array(y_np),
            jnp.array(binS), jnp.array(binU), jnp.array(binF),
            jnp.array(suf_prev),
            jnp.array(bp_starts_np), jnp.array(bp_ends_np), jnp.array(pair_mask_np),
        )

        # update SUF cache
        S_pos_np = np.array(jax.device_get(aux["S_pos"]))
        U_pos_np = np.array(jax.device_get(aux["U_pos"]))
        prev_suf = update_suf_cache(prev_suf, boundary_info, starts_np, S_pos_np, U_pos_np, beta=tcfg.beta)

        if step % tcfg.log_every == 0:
            logs["step"].append(step)
            logs["lm_loss"].append(float(aux["lm_loss"]))
            logs["thermal_loss"].append(float(aux["thermal_loss"]))
            logs["total_loss"].append(float(aux["total_loss"]))
            logs["tau_mean"].append(np.array(jax.device_get(aux["tau_mean"])).tolist())
            logs["S_layers"].append(np.array(jax.device_get(aux["S_layers"])).tolist())
            logs["lr"].append(float(aux["lr"]))

            if step % (tcfg.log_every * 10) == 0:
                print(f"  step {step:5d} | lm={float(aux['lm_loss']):.4f}  "
                      f"therm={float(aux['thermal_loss']):.4f}  "
                      f"tau={np.array(jax.device_get(aux['tau_mean']))}  "
                      f"lr={float(aux['lr']):.6f}")

        if step > 0 and step % tcfg.ckpt_every == 0:
            save_checkpoint(str(tcfg.ckpt_dir), step, params, opt_state, prev_suf, discretizer, logs)

    save_checkpoint(str(tcfg.ckpt_dir), tcfg.total_steps, params, opt_state, prev_suf, discretizer, logs)
    return params, logs