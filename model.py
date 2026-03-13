import jax
import jax.numpy as jnp
import flax.linen as nn
from config import ModelCfg

def causal_mask(L: int):
    m = jnp.tril(jnp.ones((L, L), dtype=bool))
    return m[None, None, :, :]

class ThermalEmbed(nn.Module):
    d_model: int
    n_bins: int = 8

    @nn.compact
    def __call__(self, binS, binU, binF):
        eS = nn.Embed(self.n_bins, self.d_model, name="embS")(binS)
        eU = nn.Embed(self.n_bins, self.d_model, name="embU")(binU)
        eF = nn.Embed(self.n_bins, self.d_model, name="embF")(binF)
        return eS + eU + eF

class TauHead(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, SUF_prev):
        x = nn.Dense(32)(SUF_prev)
        x = nn.tanh(x)
        x = nn.Dense(self.n_heads)(x)
        return nn.softplus(x) + 1e-3

class ThermalPredictor(nn.Module):
    @nn.compact
    def __call__(self, h):
        x = nn.Dense(64)(h)
        x = nn.gelu(x)
        x = nn.Dense(3)(x)
        return x

class FFN(nn.Module):
    cfg: ModelCfg
    deterministic: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.cfg.d_ff)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.cfg.dropout)(x, deterministic=self.deterministic)
        x = nn.Dense(self.cfg.d_model)(x)
        x = nn.Dropout(self.cfg.dropout)(x, deterministic=self.deterministic)
        return x

class ThermoMHA(nn.Module):
    cfg: ModelCfg
    deterministic: bool

    @nn.compact
    def __call__(self, x, tau_per_head):
        B, L, D = x.shape
        H = self.cfg.n_heads
        Dh = D // H

        qkv = nn.Dense(3 * D, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def reshape(t):
            return t.reshape(B, L, H, Dh).transpose(0, 2, 1, 3)
        q, k, v = map(reshape, (q, k, v))

        att = jnp.einsum("bhld,bhmd->bhlm", q, k) / jnp.sqrt(Dh)
        att = att / tau_per_head[:, :, None, None]

        m = causal_mask(L)
        att = jnp.where(m, att, -1e9)

        w = nn.softmax(att.astype(jnp.float32), axis=-1) # Cast to float32 for softmax
        w = nn.Dropout(self.cfg.dropout)(w, deterministic=self.deterministic)

        out = jnp.einsum("bhlm,bhmd->bhld", w, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = nn.Dense(D, use_bias=False)(out)
        out = nn.Dropout(self.cfg.dropout)(out, deterministic=self.deterministic)
        return out, w

class ThermoBlock(nn.Module):
    cfg: ModelCfg
    deterministic: bool

    @nn.compact
    def __call__(self, x, tau_per_head):
        y, w = ThermoMHA(self.cfg, self.deterministic)(nn.LayerNorm()(x), tau_per_head)
        x = x + y
        x = x + FFN(self.cfg, self.deterministic)(nn.LayerNorm()(x))
        return x, w

class ThermoTransformerLM(nn.Module):
    cfg: ModelCfg
    deterministic: bool

    @nn.compact
    def __call__(self, input_ids, binS, binU, binF, tau_per_head):
        B, L = input_ids.shape
        tok_emb = nn.Embed(self.cfg.vocab_size, self.cfg.d_model)(input_ids)

        pos = jnp.arange(L)[None, :]
        pos_emb = nn.Embed(self.cfg.max_len, self.cfg.d_model)(pos)

        therm = ThermalEmbed(self.cfg.d_model, self.cfg.n_bins)(binS, binU, binF)

        x = tok_emb + pos_emb + therm
        x = nn.Dropout(self.cfg.dropout)(x, deterministic=self.deterministic)

        attn_list = []
        for li in range(self.cfg.n_layers):
            x, w = ThermoBlock(self.cfg, self.deterministic)(x, tau_per_head)
            if li in self.cfg.capture_layers:
                attn_list.append(w)
        
        # Ensure attn_list is always a list of jax arrays for consistent typing
        # If no layers are captured, ensure it's still a list that can be processed.
        # Here we'll ensure it's a non-empty list for consistent unpacking in train/eval
        if not attn_list:
             # Add a dummy zero array if attn_list is empty to maintain structure
            attn_list = [jnp.zeros((B, self.cfg.n_heads, L, L), dtype=jnp.float32)]

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.cfg.vocab_size, use_bias=False)(x)
        return logits, attn_list, x # x = hidden states for thermal predictor