import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from config import ModelCfg, TrainCfg
from model import ThermoTransformerLM, TauHead
from data import get_batch
from thermal import (
    attn_entropy_per_position, surprisal_per_position, entropies_from_list,
    build_thermal_inputs, SUFDiscretizer,
)

def eval_loss(params, mcfg, tcfg, data, discretizer, prev_suf, n_batches=20):
    thermo_model = ThermoTransformerLM(cfg=mcfg, deterministic=True)
    tau_head_mod = TauHead(n_heads=mcfg.n_heads)

    val_starts = data["val_starts"]
    token_ids = data["token_ids"]
    boundary_positions = data["boundary_positions"]

    total_lm_loss, total_S, count = 0.0, 0.0, 0

    for _ in range(n_batches):
        x_np, y_np, starts_np = get_batch(val_starts, tcfg.batch_size, token_ids, tcfg.seq_len)
        binS, binU, binF, suf_prev_b, _ = build_thermal_inputs(
            token_ids, boundary_positions, starts_np, tcfg.seq_len, prev_suf, discretizer, mcfg.n_bins
        )
        tau = tau_head_mod.apply({"params": params["t"]}, jnp.array(suf_prev_b))
        logits, attn_list, _ = thermo_model.apply(
            {"params": params["m"]}, jnp.array(x_np), jnp.array(binS), jnp.array(binU), jnp.array(binF), tau
        )
        lm_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, mcfg.vocab_size), jnp.array(y_np).reshape(-1)
        ).mean()
        total_lm_loss += float(lm_loss)
        if attn_list:
            total_S += float(jnp.mean(jnp.array([jnp.mean(attn_entropy_per_position(w)) for w in attn_list])))
        count += 1

    return {"val_lm_loss": total_lm_loss / max(count, 1), "val_mean_S": total_S / max(count, 1)}

def compute_thermal_trajectory(params, mcfg, tcfg, data, discretizer, prev_suf):
    thermo_model = ThermoTransformerLM(cfg=mcfg, deterministic=True)
    tau_head_mod = TauHead(n_heads=mcfg.n_heads)
    token_ids = data["token_ids"]
    boundary_positions = data["boundary_positions"]

    trajectory = {"pos": [], "S": [], "U": [], "F": []}
    pos = 0
    while pos + tcfg.seq_len + 1 <= len(token_ids):
        starts_np = np.array([pos], dtype=np.int32)
        x = token_ids[pos:pos + tcfg.seq_len][None, :]
        y = token_ids[pos + 1:pos + tcfg.seq_len + 1][None, :]

        binS, binU, binF, suf_prev_b, boundary_info = build_thermal_inputs(
            token_ids, boundary_positions, starts_np, tcfg.seq_len, prev_suf, discretizer, mcfg.n_bins
        )
        tau = tau_head_mod.apply({"params": params["t"]}, jnp.array(suf_prev_b))
        logits, attn_list, _ = thermo_model.apply(
            {"params": params["m"]}, jnp.array(x), jnp.array(binS), jnp.array(binU), jnp.array(binF), tau
        )

        S_pos = np.array(jax.device_get(attn_entropy_per_position(attn_list[-1])))[0] if attn_list else np.zeros(tcfg.seq_len)
        U_pos = np.array(jax.device_get(surprisal_per_position(logits, jnp.array(y))))[0]

        for bp in boundary_info[0]:
            rel = bp - pos
            ctx_start = max(0, rel - 32)
            if rel > ctx_start:
                s_val = float(np.mean(S_pos[ctx_start:rel]))
                u_val = float(np.mean(U_pos[ctx_start:rel]))
                f_val = u_val - tcfg.beta * s_val
                trajectory["pos"].append(bp)
                trajectory["S"].append(s_val)
                trajectory["U"].append(u_val)
                trajectory["F"].append(f_val)
                prev_suf[bp] = (s_val, u_val, f_val)

        pos += tcfg.seq_len // 2
    return trajectory

# ---- Plots ----

def plot_training_logs(logs, save_dir="thermformer/plots"):
    os.makedirs(save_dir, exist_ok=True)
    steps = logs["step"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, logs["lm_loss"], label="LM loss", alpha=0.8)
    ax.plot(steps, logs["thermal_loss"], label="Thermal loss", alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("ThermFormer Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    tau_arr = np.array(logs["tau_mean"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for h in range(tau_arr.shape[1]):
        ax.plot(steps, tau_arr[:, h], label=f"head {h}", alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("tau"); ax.set_title("Per-Head Attention Temperature")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "tau_per_head.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    S_arr = np.array(logs["S_layers"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(S_arr.shape[1]):
        ax.plot(steps, S_arr[:, i], label=f"layer {i}", alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("S"); ax.set_title("Per-Layer Attention Entropy")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "entropy_per_layer.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"[plot] Saved to {save_dir}/")

def plot_thermal_trajectory(trajectory, save_dir="thermformer/plots"):
    os.makedirs(save_dir, exist_ok=True)
    pos = trajectory["pos"]
    if not pos:
        print("[plot] No trajectory data."); return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(pos, trajectory["S"], color="tab:blue", alpha=0.7, linewidth=0.8)
    axes[0].set_ylabel("S (Entropy)"); axes[0].set_title("Thermal Trajectory — Game of Thrones"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(pos, trajectory["U"], color="tab:orange", alpha=0.7, linewidth=0.8)
    axes[1].set_ylabel("U (Surprisal)"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(pos, trajectory["F"], color="tab:green", alpha=0.7, linewidth=0.8)
    axes[2].set_ylabel("F (Free Energy)"); axes[2].set_xlabel("Token Position"); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "thermal_trajectory.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] Saved thermal_trajectory.png")