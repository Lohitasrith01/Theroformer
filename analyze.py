"""
Chunk-level and chapter-level thermal analysis.
Shows whether S/U/F actually varies with narrative content.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp

from config import ModelCfg, TrainCfg
from model import ThermoTransformerLM, TauHead
from data import load_tokenizer, build_token_stream, make_windows, _split_into_chunks
from train import load_latest_checkpoint
from thermal import (
    attn_entropy_per_position, surprisal_per_position,
    SUFDiscretizer, build_thermal_inputs,
)


def compute_chunk_level_suf(params, mcfg, tcfg, data, discretizer, prev_suf):
    """Compute S, U, F for EACH chunk (not each boundary observation).
    Returns one S, U, F value per chunk with the chunk text."""

    thermo_model = ThermoTransformerLM(cfg=mcfg, deterministic=True)
    tau_head = TauHead(n_heads=mcfg.n_heads)
    token_ids = data["token_ids"]
    boundary_positions = data["boundary_positions"]
    chunks = data["chunks"]

    results = []
    pos = 0

    while pos + tcfg.seq_len + 1 <= len(token_ids):
        starts_np = np.array([pos], dtype=np.int32)
        x = token_ids[pos:pos + tcfg.seq_len][None, :]
        y = token_ids[pos + 1:pos + tcfg.seq_len + 1][None, :]

        binS, binU, binF, suf_prev_b, boundary_info = build_thermal_inputs(
            token_ids, boundary_positions, starts_np, tcfg.seq_len,
            prev_suf, discretizer, mcfg.n_bins
        )
        tau = tau_head.apply({"params": params["t"]}, jnp.array(suf_prev_b))
        logits, attn_list, _ = thermo_model.apply(
            {"params": params["m"]},
            jnp.array(x), jnp.array(binS), jnp.array(binU), jnp.array(binF), tau
        )

        S_pos = np.array(jax.device_get(
            attn_entropy_per_position(attn_list[-1]) if attn_list else jnp.zeros(tcfg.seq_len)
        ))[0]
        U_pos = np.array(jax.device_get(
            surprisal_per_position(logits, jnp.array(y))
        ))[0]

        # for each boundary in this window, record chunk-level stats
        for bp in boundary_info[0]:
            rel = bp - pos
            ctx_start = max(0, rel - 64)  # wider context for chunk-level
            if rel > ctx_start:
                s_val = float(np.mean(S_pos[ctx_start:rel]))
                u_val = float(np.mean(U_pos[ctx_start:rel]))
                f_val = u_val - tcfg.beta * s_val

                # find which chunk this boundary belongs to
                bp_idx = np.searchsorted(boundary_positions, bp)
                chunk_idx = min(bp_idx, len(chunks) - 1)

                results.append({
                    "chunk_idx": int(chunk_idx),
                    "bp_pos": int(bp),
                    "S": s_val,
                    "U": u_val,
                    "F": f_val,
                    "text_preview": chunks[chunk_idx][:100] if chunk_idx < len(chunks) else "",
                })
                prev_suf[bp] = (s_val, u_val, f_val)

        pos += tcfg.seq_len // 2

    return results


def detect_chapters(chunks):
    """Find chapter boundaries by looking for all-caps headings in chunks."""
    chapter_starts = []
    chapter_names = []

    for i, ch in enumerate(chunks):
        # check if chunk starts with an all-caps name (chapter heading)
        first_words = ch.strip().split()[:5]
        first_part = " ".join(first_words)
        if re.match(r'^[A-Z][A-Z\s\'.,-]{2,25}$', first_part.strip()):
            chapter_starts.append(i)
            chapter_names.append(first_part.strip())

    return chapter_starts, chapter_names


def plot_chapter_level(results, chunks, save_dir="plots"):
    """Plot S, U, F aggregated per chapter."""
    os.makedirs(save_dir, exist_ok=True)

    chapter_starts, chapter_names = detect_chapters(chunks)

    if not chapter_starts:
        print("[analyze] No chapters detected, falling back to fixed-window aggregation")
        plot_smoothed(results, save_dir)
        return

    # assign each result to a chapter
    chapter_sufs = {i: {"S": [], "U": [], "F": []} for i in range(len(chapter_starts))}

    for r in results:
        cidx = r["chunk_idx"]
        # find which chapter this chunk belongs to
        ch_num = 0
        for j, cs in enumerate(chapter_starts):
            if cidx >= cs:
                ch_num = j
        chapter_sufs[ch_num]["S"].append(r["S"])
        chapter_sufs[ch_num]["U"].append(r["U"])
        chapter_sufs[ch_num]["F"].append(r["F"])

    # compute per-chapter means
    ch_indices = []
    ch_S, ch_U, ch_F = [], [], []
    ch_labels = []

    for i in sorted(chapter_sufs.keys()):
        if chapter_sufs[i]["S"]:
            ch_indices.append(i)
            ch_S.append(np.mean(chapter_sufs[i]["S"]))
            ch_U.append(np.mean(chapter_sufs[i]["U"]))
            ch_F.append(np.mean(chapter_sufs[i]["F"]))
            ch_labels.append(chapter_names[i] if i < len(chapter_names) else f"Ch {i}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    x_pos = range(len(ch_indices))

    axes[0].bar(x_pos, ch_S, color="tab:blue", alpha=0.7)
    axes[0].set_ylabel("S (Entropy)")
    axes[0].set_title("Per-Chapter Thermal Profile — Game of Thrones")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x_pos, ch_U, color="tab:orange", alpha=0.7)
    axes[1].set_ylabel("U (Surprisal)")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(x_pos, ch_F, color="tab:green", alpha=0.7)
    axes[2].set_ylabel("F (Free Energy)")
    axes[2].set_xlabel("Chapter")
    axes[2].grid(True, alpha=0.3, axis="y")

    # rotate labels
    for ax in axes:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ch_labels, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "chapter_level_suf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Saved chapter_level_suf.png")

    # also print the data
    print(f"\n{'Chapter':<25} {'S':>8} {'U':>8} {'F':>8}")
    print("-" * 51)
    for i in range(len(ch_indices)):
        print(f"{ch_labels[i]:<25} {ch_S[i]:8.3f} {ch_U[i]:8.3f} {ch_F[i]:8.3f}")


def plot_smoothed(results, save_dir="plots", window=20):
    """Smoothed trajectory with rolling average."""
    os.makedirs(save_dir, exist_ok=True)

    positions = [r["bp_pos"] for r in results]
    S = np.array([r["S"] for r in results])
    U = np.array([r["U"] for r in results])
    F = np.array([r["F"] for r in results])

    def smooth(arr, w):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    S_sm = smooth(S, window)
    U_sm = smooth(U, window)
    F_sm = smooth(F, window)
    pos_sm = smooth(np.array(positions, dtype=float), window)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(pos_sm, S_sm, color="tab:blue", linewidth=1.2)
    axes[0].fill_between(pos_sm, S_sm - 0.3, S_sm + 0.3, alpha=0.15, color="tab:blue")
    axes[0].set_ylabel("S (Entropy)")
    axes[0].set_title(f"Smoothed Thermal Trajectory (window={window} chunks)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pos_sm, U_sm, color="tab:orange", linewidth=1.2)
    axes[1].fill_between(pos_sm, U_sm - 0.3, U_sm + 0.3, alpha=0.15, color="tab:orange")
    axes[1].set_ylabel("U (Surprisal)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pos_sm, F_sm, color="tab:green", linewidth=1.2)
    axes[2].fill_between(pos_sm, F_sm - 0.3, F_sm + 0.3, alpha=0.15, color="tab:green")
    axes[2].set_ylabel("F (Free Energy)")
    axes[2].set_xlabel("Token Position")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "smoothed_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Saved smoothed_trajectory.png")


def plot_extreme_chunks(results, save_dir="plots", n=10):
    """Show the highest and lowest F chunks — what's dramatic vs calm?"""
    os.makedirs(save_dir, exist_ok=True)

    sorted_by_F = sorted(results, key=lambda r: r["F"])

    print(f"\n{'='*60}")
    print(f"TOP {n} LOWEST FREE ENERGY (calm/predictable)")
    print(f"{'='*60}")
    for r in sorted_by_F[:n]:
        print(f"  F={r['F']:.3f}  S={r['S']:.3f}  U={r['U']:.3f}")
        print(f"  {r['text_preview']}...")
        print()

    print(f"{'='*60}")
    print(f"TOP {n} HIGHEST FREE ENERGY (dramatic/surprising)")
    print(f"{'='*60}")
    for r in sorted_by_F[-n:]:
        print(f"  F={r['F']:.3f}  S={r['S']:.3f}  U={r['U']:.3f}")
        print(f"  {r['text_preview']}...")
        print()


if __name__ == "__main__":
    mcfg = ModelCfg()
    tcfg = TrainCfg()

    tok = load_tokenizer(tcfg.tok_path)
    mcfg.vocab_size = tok.get_vocab_size()

    text = (tcfg.data_dir / "got_clean_chunked_v3.txt").read_text(errors="ignore")
    chunks = _split_into_chunks(text)
    token_ids, boundary_positions = build_token_stream(chunks, tok)
    from data import make_windows, train_val_split
    starts = make_windows(token_ids, tcfg.seq_len, tcfg.stride)
    train_starts, val_starts = train_val_split(starts, tcfg.val_frac)

    data = {"tok": tok, "chunks": chunks, "token_ids": token_ids,
            "boundary_positions": boundary_positions,
            "train_starts": train_starts, "val_starts": val_starts}

    ckpt = load_latest_checkpoint(str(tcfg.ckpt_dir))
    if ckpt is None:
        print("No checkpoint found.")
        exit(1)

    params = jax.device_put(ckpt["params"])
    discretizer = SUFDiscretizer(n_bins=mcfg.n_bins)
    ds = ckpt["discretizer_state"]
    discretizer.s_min, discretizer.s_max = ds["s_min"], ds["s_max"]
    discretizer.u_min, discretizer.u_max = ds["u_min"], ds["u_max"]
    discretizer.f_min, discretizer.f_max = ds["f_min"], ds["f_max"]
    prev_suf = ckpt["prev_suf"]

    print("[analyze] Computing chunk-level S/U/F...")
    results = compute_chunk_level_suf(params, mcfg, tcfg, data, discretizer, prev_suf)
    print(f"[analyze] {len(results)} chunk observations")

    plot_chapter_level(results, chunks)
    plot_smoothed(results)
    plot_extreme_chunks(results)

    print("\n[analyze] Done!")