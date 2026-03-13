import re
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from config import TrainCfg

# ---- Stage 1: raw → sentence-aligned chunks (Cell 0) ----

def is_heading(s: str) -> bool:
    return len(s) <= 25 and re.fullmatch(r"[A-Z][A-Z '.-]*", s) is not None

def raw_to_chunks_v1(raw_text: str, target_words: int = 220) -> str:
    raw = re.sub(r'(?m)^\s*Page\s+\d+\s*$', '', raw_text)
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    raw = re.sub(r'[ \t]+', ' ', raw)
    lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]

    blocks, cur = [], []
    for ln in lines:
        if is_heading(ln):
            if cur:
                blocks.append(" ".join(cur))
                cur = []
            blocks.append(f"{ln}\n<BOUNDARY>")
        else:
            cur.append(ln)
    if cur:
        blocks.append(" ".join(cur))

    text = "\n".join(blocks)
    text = re.sub(r'\n(?!<BOUNDARY>)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunk, wc, out = [], 0, []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if "<BOUNDARY>" in s:
            parts = s.split("<BOUNDARY>")
            for part in parts:
                part = part.strip()
                if part:
                    chunk.append(part)
                    wc += len(part.split())
                if chunk:
                    out.append(" ".join(chunk).strip())
                    out.append("<BOUNDARY>")
                    chunk, wc = [], 0
            continue
        chunk.append(s)
        wc += len(s.split())
        if wc >= target_words:
            out.append(" ".join(chunk).strip())
            out.append("<BOUNDARY>")
            chunk, wc = [], 0

    if chunk:
        out.append(" ".join(chunk).strip())
        out.append("<BOUNDARY>")
    return "\n".join(out)

# ---- Stage 2: split-word join + merge (Cell 1) ----

def _split_into_chunks(text: str) -> List[str]:
    parts = re.split(r"\n?<BOUNDARY>\n?", text)
    return [p.strip() for p in parts if p.strip()]

def _tokenize_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", s)

def _n_tokens(s: str) -> int:
    return len(re.findall(r"\S+", s))

def postprocess_v2(text_v1: str, min_tokens: int = 80) -> str:
    BIGRAM_MAXCOUNT, JOINED_MINCOUNT, JOINED_RATIO, MAX_JOIN_RULES = 2, 5, 3, 200
    chunks = _split_into_chunks(text_v1)

    unigrams, bigrams = Counter(), Counter()
    for ch in chunks:
        toks = _tokenize_words(ch)
        unigrams.update(toks)
        bigrams.update(zip(toks, toks[1:]))

    vocab = set(unigrams.keys())
    candidates = []
    for (a, b), c in bigrams.items():
        if c > BIGRAM_MAXCOUNT: continue
        if len(a) < 2 or len(a) > 4 or len(b) < 3 or len(b) > 12: continue
        ab = a + b
        if ab not in vocab: continue
        if unigrams[ab] >= JOINED_MINCOUNT and unigrams[ab] >= JOINED_RATIO * c:
            candidates.append((a, b, ab, c, unigrams[ab]))

    candidates.sort(key=lambda x: (x[4], -x[3]), reverse=True)
    candidates = candidates[:MAX_JOIN_RULES]
    replacements = [(a, b, ab) for a, b, ab, _, _ in candidates]

    def apply_joins(t):
        for a, b, ab in replacements:
            t = re.sub(rf"\b{re.escape(a)}\s+{re.escape(b)}\b", ab, t)
        return t

    fixed = [apply_joins(ch) for ch in chunks]

    merged, i = [], 0
    while i < len(fixed):
        ch = fixed[i]
        if _n_tokens(ch) < min_tokens and i + 1 < len(fixed):
            merged.append((ch + " " + fixed[i + 1]).strip())
            i += 2
        else:
            merged.append(ch)
            i += 1

    out_lines = []
    for ch in merged:
        out_lines.append(ch.strip())
        out_lines.append("<BOUNDARY>")
    return "\n".join(out_lines).strip() + "\n"

# ---- Stage 3: drop frontmatter + final merge (Cell 3) ----

def postprocess_v3(text_v2: str, min_tokens: int = 80) -> str:
    chunks = _split_into_chunks(text_v2)

    if chunks:
        first = chunks[0]
        if _n_tokens(first) < min_tokens and (
            "A Game Of Thrones" in first or "A Song of Ice and Fire" in first or "George" in first
        ):
            chunks = chunks[1:]

    merged, i = [], 0
    while i < len(chunks):
        ch = chunks[i]
        if _n_tokens(ch) < min_tokens and i + 1 < len(chunks):
            merged.append((ch + " " + chunks[i + 1]).strip())
            i += 2
        else:
            merged.append(ch)
            i += 1

    if len(merged) >= 2 and _n_tokens(merged[-1]) < min_tokens:
        merged[-2] = (merged[-2] + " " + merged[-1]).strip()
        merged = merged[:-1]

    out_lines = []
    for ch in merged:
        out_lines.append(ch.strip())
        out_lines.append("<BOUNDARY>")
    return "\n".join(out_lines).strip() + "\n"

# ---- Full pipeline ----

def preprocess(cfg: TrainCfg) -> Tuple[str, List[str]]:
    print(f"[data] Reading {cfg.raw_text}")
    raw = cfg.raw_text.read_text(errors="ignore")
    print("[data] Stage 1: raw → sentence-aligned chunks")
    v1 = raw_to_chunks_v1(raw, target_words=cfg.target_words)
    print("[data] Stage 2: split-word join + merge")
    v2 = postprocess_v2(v1, min_tokens=cfg.min_tokens)
    print("[data] Stage 3: drop frontmatter + final merge")
    v3 = postprocess_v3(v2, min_tokens=cfg.min_tokens)

    out_path = cfg.data_dir / "got_clean_chunked_v3.txt"
    out_path.write_text(v3)
    print(f"[data] Wrote {out_path}  chars={len(v3)}")

    chunks = _split_into_chunks(v3)
    print(f"[data] {len(chunks)} chunks")
    return v3, chunks

# ---- BPE tokenizer (Cell 6) ----

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<BOUNDARY>"]

def train_tokenizer(chunks: List[str], vocab_size: int, save_path: Path) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<PAD>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2)
    tok.train_from_iterator(chunks, trainer=trainer)
    tok.save(str(save_path))
    print(f"[data] Saved tokenizer: {save_path}  vocab={tok.get_vocab_size()}")
    enc = tok.encode("Winter is coming. <BOUNDARY> The wolves howled.")
    print(f"[data] Sanity tokens: {enc.tokens[:15]}")
    return tok

def load_tokenizer(path: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    expected = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<BOUNDARY>": 3}
    for t, eid in expected.items():
        tid = tok.token_to_id(t)
        assert tid == eid, f"{t} id mismatch: got {tid}, expected {eid}"
    return tok

# ---- Token stream (Cell 7) ----

def build_token_stream(chunks: List[str], tok: Tokenizer):
    BOS_ID = tok.token_to_id("<BOS>")
    EOS_ID = tok.token_to_id("<EOS>")
    BOUNDARY_ID = tok.token_to_id("<BOUNDARY>")

    token_ids = [BOS_ID]
    boundary_positions = []
    for ch in chunks:
        ids = tok.encode(ch).ids
        token_ids.extend(ids)
        token_ids.append(BOUNDARY_ID)
        boundary_positions.append(len(token_ids) - 1)
    token_ids.append(EOS_ID)

    token_ids = np.array(token_ids, dtype=np.int32)
    boundary_positions = np.array(boundary_positions, dtype=np.int32)
    print(f"[data] Total tokens: {token_ids.shape[0]}  boundaries: {boundary_positions.shape[0]}")
    return token_ids, boundary_positions

# ---- Windowed dataset (Cell 8) ----

def make_windows(token_ids: np.ndarray, seq_len: int, stride: int):
    starts = []
    pos = 0
    while pos + seq_len + 1 <= len(token_ids):
        starts.append(pos)
        pos += stride
    return np.array(starts, dtype=np.int32)

def train_val_split(starts: np.ndarray, val_frac: float):
    split = int(len(starts) * (1.0 - val_frac))
    return starts[:split], starts[split:]

# ---- Batch (Cell 13) ----

def get_batch(starts, batch_size, token_ids, seq_len):
    idx = np.random.choice(len(starts), size=batch_size, replace=False)
    s = starts[idx]
    x = np.stack([token_ids[i:i + seq_len] for i in s], axis=0)
    y = np.stack([token_ids[i + 1:i + seq_len + 1] for i in s], axis=0)
    return x.astype(np.int32), y.astype(np.int32), s

# ---- One-shot prepare ----

def prepare_all(cfg: TrainCfg):
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    _, chunks = preprocess(cfg)

    if cfg.tok_path.exists():
        print(f"[data] Loading existing tokenizer: {cfg.tok_path}")
        tok = load_tokenizer(cfg.tok_path)
    else:
        tok = train_tokenizer(chunks, vocab_size=12_000, save_path=cfg.tok_path)

    token_ids, boundary_positions = build_token_stream(chunks, tok)
    starts = make_windows(token_ids, cfg.seq_len, cfg.stride)
    train_starts, val_starts = train_val_split(starts, cfg.val_frac)
    print(f"[data] Train windows: {len(train_starts)}  Val windows: {len(val_starts)}")

    return {
        "tok": tok, "chunks": chunks,
        "token_ids": token_ids, "boundary_positions": boundary_positions,
        "train_starts": train_starts, "val_starts": val_starts,
    }