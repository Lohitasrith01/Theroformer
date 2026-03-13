from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelCfg:
    vocab_size: int = 12_000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 1024
    n_bins: int = 8
    capture_layers: tuple = (0, 1, 2, 3)

@dataclass
class TrainCfg:
    seq_len: int = 1024
    stride: int = 256
    target_words: int = 220
    min_tokens: int = 80
    val_frac: float = 0.1

    batch_size: int = 8
    total_steps: int = 2000
    warmup_steps: int = 100
    peak_lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-2

    beta: float = 1.0
    lambda_thermal: float = 0.15
    thermal_warmup_steps: int = 200 

    log_every: int = 5
    ckpt_every: int = 50
    eval_every: int = 50
    
    raw_text: Path = Path("data/1 - A Game of Thrones.txt")
    data_dir: Path = Path("data")
    ckpt_dir: Path = Path("checkpoint")
    tok_path: Path = Path("data/tokenizer_got_bpe.json")

    seed: int = 42