"""
Text generation with live thermal feedback.
Tests whether S/U/F actually influence generation behavior.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from config import ModelCfg, TrainCfg
from model import ThermoTransformerLM, TauHead
from thermal import attn_entropy_per_position, surprisal_per_position, SUFDiscretizer
from data import load_tokenizer
from train import load_latest_checkpoint


def generate(
    params, mcfg, tcfg, tok, prompt_text,
    max_new_tokens=200,
    top_k=40,
    top_p=0.9,
    sample_temp=1.0,
    use_thermal=True,
    seed=42,
):
    """Autoregressive generation with live thermal feedback.

    Returns:
        text: generated string
        trajectory: dict with per-step S, U, F, tau values
    """
    thermo_model = ThermoTransformerLM(cfg=mcfg, deterministic=True)
    tau_head = TauHead(n_heads=mcfg.n_heads)
    discretizer = SUFDiscretizer(n_bins=mcfg.n_bins)

    # encode prompt
    prompt_ids = tok.encode(prompt_text).ids
    BOS_ID = tok.token_to_id("<BOS>")
    token_ids = [BOS_ID] + prompt_ids

    # thermal state starts neutral
    cur_S, cur_U, cur_F = 0.0, 0.0, 0.0

    rng = jax.random.PRNGKey(seed)
    trajectory = {"step": [], "S": [], "U": [], "F": [],
                  "tau": [], "token": [], "text_so_far": []}

    for step in range(max_new_tokens):
        # build input window (truncate to max_len)
        seq = token_ids[-tcfg.seq_len:]
        L = len(seq)
        x = jnp.array(seq, dtype=jnp.int32)[None, :]  # (1, L)

        # thermal inputs
        if use_thermal:
            binS, binU, binF = discretizer(
                np.array([cur_S]), np.array([cur_U]), np.array([cur_F])
            )
            binS_arr = jnp.full((1, L), int(binS[0]), dtype=jnp.int32)
            binU_arr = jnp.full((1, L), int(binU[0]), dtype=jnp.int32)
            binF_arr = jnp.full((1, L), int(binF[0]), dtype=jnp.int32)
            suf_prev = jnp.array([[cur_S, cur_U, cur_F]], dtype=jnp.float32)
        else:
            binS_arr = jnp.zeros((1, L), dtype=jnp.int32)
            binU_arr = jnp.zeros((1, L), dtype=jnp.int32)
            binF_arr = jnp.zeros((1, L), dtype=jnp.int32)
            suf_prev = jnp.zeros((1, 3), dtype=jnp.float32)

        tau = tau_head.apply({"params": params["t"]}, suf_prev)  # (1, H)

        logits, attn_list, _ = thermo_model.apply(
            {"params": params["m"]}, x, binS_arr, binU_arr, binF_arr, tau
        )

        # get logits for last position
        next_logits = logits[0, -1, :]  # (V,)

        # apply sampling temperature
        next_logits = next_logits / sample_temp

        # top-k filtering
        if top_k > 0:
            top_k_vals = jax.lax.top_k(next_logits, top_k)
            threshold = top_k_vals[0][-1]
            next_logits = jnp.where(next_logits < threshold, -1e9, next_logits)

        # top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = jnp.argsort(-next_logits)
            sorted_logits = next_logits[sorted_indices]
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
            cutoff_mask = cumulative_probs - jax.nn.softmax(sorted_logits) > top_p
            sorted_logits = jnp.where(cutoff_mask, -1e9, sorted_logits)
            next_logits = next_logits.at[sorted_indices].set(sorted_logits)

        # sample
        rng, sub = jax.random.split(rng)
        next_token = jax.random.categorical(sub, next_logits)
        next_token_id = int(next_token)

        # stop on EOS or BOUNDARY
        if next_token_id == tok.token_to_id("<EOS>"):
            break

        token_ids.append(next_token_id)

        # update thermal state from this step
        if use_thermal and attn_list:
            # S from attention entropy at last position
            S_pos = attn_entropy_per_position(attn_list[-1])  # (1, L)
            cur_S = float(S_pos[0, -1])

            # U from surprisal at last position
            target = jnp.array([[next_token_id]], dtype=jnp.int32)
            last_logits = logits[:, -1:, :]
            U_pos = surprisal_per_position(last_logits, target)
            cur_U = float(U_pos[0, 0])

            cur_F = cur_U - tcfg.beta * cur_S

        # log trajectory
        trajectory["step"].append(step)
        trajectory["S"].append(cur_S)
        trajectory["U"].append(cur_U)
        trajectory["F"].append(cur_F)
        trajectory["tau"].append(np.array(jax.device_get(tau[0])).tolist())
        trajectory["token"].append(next_token_id)

    # decode
    generated_ids = token_ids[len(prompt_ids) + 1:]
    generated_text = tok.decode(generated_ids)

    return generated_text, trajectory


def compare_thermal_vs_flat(params, mcfg, tcfg, tok, prompt, max_new_tokens=200, seed=42):
    """Generate with and without thermal feedback and compare."""

    print(f"Prompt: {prompt}\n")
    print("=" * 60)

    print("\n[WITH thermal feedback]")
    text_therm, traj_therm = generate(
        params, mcfg, tcfg, tok, prompt,
        max_new_tokens=max_new_tokens, use_thermal=True, seed=seed
    )
    print(text_therm)
    print(f"\n  mean S={np.mean(traj_therm['S']):.3f}  "
          f"mean U={np.mean(traj_therm['U']):.3f}  "
          f"mean F={np.mean(traj_therm['F']):.3f}")

    print("\n" + "=" * 60)

    print("\n[WITHOUT thermal feedback (flat zeros)]")
    text_flat, traj_flat = generate(
        params, mcfg, tcfg, tok, prompt,
        max_new_tokens=max_new_tokens, use_thermal=False, seed=seed
    )
    print(text_flat)
    print(f"\n  mean S={np.mean(traj_flat['S']):.3f}  "
          f"mean U={np.mean(traj_flat['U']):.3f}  "
          f"mean F={np.mean(traj_flat['F']):.3f}")

    return traj_therm, traj_flat


def plot_generation_trajectory(traj, save_path="plots/generation_trajectory.png"):
    """Plot S, U, F, tau during generation."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steps = traj["step"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(steps, traj["S"], color="tab:blue", alpha=0.8)
    axes[0].set_ylabel("S (Entropy)")
    axes[0].set_title("Thermal State During Generation")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, traj["U"], color="tab:orange", alpha=0.8)
    axes[1].set_ylabel("U (Surprisal)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, traj["F"], color="tab:green", alpha=0.8)
    axes[2].set_ylabel("F (Free Energy)")
    axes[2].grid(True, alpha=0.3)

    tau_arr = np.array(traj["tau"])
    for h in range(tau_arr.shape[1]):
        axes[3].plot(steps, tau_arr[:, h], label=f"head {h}", alpha=0.8)
    axes[3].set_ylabel("tau")
    axes[3].set_xlabel("Generation Step")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {save_path}")


# ---- CLI entry point ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ThermFormer generation")
    parser.add_argument("--prompt", type=str, default="The king sat upon the iron throne and",
                        help="Text prompt")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--compare", action="store_true",
                        help="Compare thermal vs flat generation")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mcfg = ModelCfg()
    tcfg = TrainCfg()

    tok = load_tokenizer(tcfg.tok_path)
    mcfg.vocab_size = tok.get_vocab_size()

    ckpt = load_latest_checkpoint(str(tcfg.ckpt_dir))
    if ckpt is None:
        print("No checkpoint found. Train first.")
        exit(1)

    params = jax.device_put(ckpt["params"])

    if args.compare:
        traj_t, traj_f = compare_thermal_vs_flat(
            params, mcfg, tcfg, tok, args.prompt,
            max_new_tokens=args.tokens, seed=args.seed
        )
        plot_generation_trajectory(traj_t, "plots/gen_thermal.png")
        plot_generation_trajectory(traj_f, "plots/gen_flat.png")
    else:
        text, traj = generate(
            params, mcfg, tcfg, tok, args.prompt,
            max_new_tokens=args.tokens, sample_temp=args.temp,
            use_thermal=True, seed=args.seed
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated:\n{text}")
        print(f"\nmean S={np.mean(traj['S']):.3f}  "
              f"mean U={np.mean(traj['U']):.3f}  "
              f"mean F={np.mean(traj['F']):.3f}")
        plot_generation_trajectory(traj)