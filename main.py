#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

from config import ModelCfg, TrainCfg
from data import prepare_all, load_tokenizer, build_token_stream, make_windows, train_val_split, _split_into_chunks
from train import train, load_latest_checkpoint
from eval import eval_loss, compute_thermal_trajectory, plot_training_logs, plot_thermal_trajectory
from thermal import SUFDiscretizer

def main():
    parser = argparse.ArgumentParser(description="ThermFormer")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    mcfg = ModelCfg()
    tcfg = TrainCfg()
    if args.steps: tcfg.total_steps = args.steps
    if args.batch_size: tcfg.batch_size = args.batch_size

    if args.skip_preprocess or args.eval_only:
        print("[main] Loading existing preprocessed data...")
        tok = load_tokenizer(tcfg.tok_path)
        text = (tcfg.data_dir / "got_clean_chunked_v3.txt").read_text(errors="ignore")
        chunks = _split_into_chunks(text)
        token_ids, boundary_positions = build_token_stream(chunks, tok)
        starts = make_windows(token_ids, tcfg.seq_len, tcfg.stride)
        train_starts, val_starts = train_val_split(starts, tcfg.val_frac)
        data = {"tok": tok, "chunks": chunks, "token_ids": token_ids,
                "boundary_positions": boundary_positions,
                "train_starts": train_starts, "val_starts": val_starts}
    else:
        data = prepare_all(tcfg)

    mcfg.vocab_size = data["tok"].get_vocab_size()
    print(f"[main] Vocab: {mcfg.vocab_size}  Tokens: {len(data['token_ids'])}")

    if args.eval_only:
        ckpt = load_latest_checkpoint(str(tcfg.ckpt_dir))
        if ckpt is None:
            print("[main] No checkpoint found."); sys.exit(1)
        params = jax.device_put(ckpt["params"])
        logs = ckpt["logs"]
        discretizer = SUFDiscretizer(n_bins=mcfg.n_bins)
        ds = ckpt["discretizer_state"]
        discretizer.s_min, discretizer.s_max = ds["s_min"], ds["s_max"]
        discretizer.u_min, discretizer.u_max = ds["u_min"], ds["u_max"]
        discretizer.f_min, discretizer.f_max = ds["f_min"], ds["f_max"]
        prev_suf = ckpt["prev_suf"]
    else:
        params, logs = train(mcfg, tcfg, data)
        discretizer = SUFDiscretizer(n_bins=mcfg.n_bins)
        prev_suf = {}
        ckpt = load_latest_checkpoint(str(tcfg.ckpt_dir))
        if ckpt:
            prev_suf = ckpt["prev_suf"]
            ds = ckpt["discretizer_state"]
            discretizer.s_min, discretizer.s_max = ds["s_min"], ds["s_max"]
            discretizer.u_min, discretizer.u_max = ds["u_min"], ds["u_max"]
            discretizer.f_min, discretizer.f_max = ds["f_min"], ds["f_max"]

    print("\n[main] Validation...")
    val_metrics = eval_loss(params, mcfg, tcfg, data, discretizer, prev_suf)
    print(f"  Val LM loss: {val_metrics['val_lm_loss']:.4f}")
    print(f"  Val mean S:  {val_metrics['val_mean_S']:.4f}")

    print("\n[main] Thermal trajectory...")
    trajectory = compute_thermal_trajectory(params, mcfg, tcfg, data, discretizer, prev_suf)
    print(f"  {len(trajectory['pos'])} boundary observations")

    print("\n[main] Plots...")
    plot_training_logs(logs)
    plot_thermal_trajectory(trajectory)
    print("\n[main] Done!")

if __name__ == "__main__":
    main()

