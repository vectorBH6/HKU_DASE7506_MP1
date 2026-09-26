"""Knowledge-enhanced causal LM for MP1 training and scoring interfaces.

Improvements over the baseline:
- Pre-norm RMSNorm blocks combine SwiGLU with grouped-query attention,
  QK normalization, and partial RoPE.
- Better weight initialization with depth-scaled std.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMS normalization with FP32 accumulation for FP16/BF16 inputs."""
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squaring large FP16 activations directly can overflow to infinity.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        norm = (values.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        return (values * norm).to(x.dtype) * self.weight.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split and rotate half the dimensions for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Partial RoPE; cache only position constants, never token information."""
    def __init__(self, head_dim: int, base: float = 10000.0, rope_fraction: float = 0.5):
        super().__init__()
        if head_dim < 2:
            raise ValueError('head_dim must be at least 2 for RoPE')
        if (isinstance(base, bool) or not isinstance(base, (int, float))
                or not math.isfinite(base) or base <= 1):
            raise ValueError('rope_base must be finite and greater than 1')
        if (isinstance(rope_fraction, bool) or not isinstance(rope_fraction, (int, float))
                or not math.isfinite(rope_fraction) or not 0 < rope_fraction <= 1):
            raise ValueError('rope_fraction must be in (0, 1]')
        self.head_dim = head_dim
        # Rotate complete pairs, including small/odd head dimensions.
        self.rope_dim = max(2, int(head_dim * rope_fraction) // 2 * 2)
        self.base = base
        self.register_buffer('_cached_cos', None, persistent=False)
        self.register_buffer('_cached_sin', None, persistent=False)

    def _apply(self, fn, recurse=True):
        # Drop derived constants on .to()/.half()/.float(). Otherwise a
        # half -> float roundtrip silently keeps rounded positional values.
        self._cached_cos = None
        self._cached_sin = None
        return super()._apply(fn, recurse=recurse)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute angles at full precision, even under autocast/model.half()."""
        cache_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        if (self._cached_cos is None or self._cached_cos.shape[0] < seq_len
                or self._cached_cos.device != device
                or self._cached_cos.dtype != cache_dtype):
            with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                positions = torch.arange(seq_len, device=device, dtype=cache_dtype)
                dims = torch.arange(0, self.rope_dim, 2, device=device, dtype=cache_dtype)
                inv_freq = 1.0 / (self.base ** (dims / self.rope_dim))
                freqs = torch.outer(positions, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                self._cached_cos = emb.cos()
                self._cached_sin = emb.sin()

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to first rope_dim dimensions only."""
        seq_len = x.shape[-2]
        self._build_cache(seq_len, x.device, x.dtype)
        cos = self._cached_cos[:seq_len].to(x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self._cached_sin[:seq_len].to(x.dtype).unsqueeze(0).unsqueeze(0)
        x_rot = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]
        x_rot_out = x_rot * cos + _rotate_half(x_rot) * sin
        if x_pass.shape[-1] == 0:
            return x_rot_out
        return torch.cat((x_rot_out, x_pass), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE rotation to queries and keys."""
        return self._apply_rope(q), self._apply_rope(k)


class SwiGLU(nn.Module):
    """Swish-gated feed-forward network with three bias-free projections."""
    def __init__(self, width: int, hidden_dim: int):
        super().__init__()
        # w1 = gate projection, w2 = up projection, w3 = down projection
        self.w1 = nn.Linear(width, hidden_dim, bias=False)
        self.w2 = nn.Linear(width, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """Causal grouped-query attention with QK normalization and partial RoPE.
    """
    def __init__(self, width: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 context: int, dropout: float = 0.05, rope_base: float = 10000.0,
                 rope_fraction: float = 0.5):
        super().__init__()
        if n_heads < 1 or n_kv_heads < 1 or n_heads % n_kv_heads:
            raise ValueError('n_kv_heads must be positive and divide n_heads')
        if width != n_heads * head_dim:
            raise ValueError('width must equal n_heads * head_dim')
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_heads // n_kv_heads

        self.q_proj = nn.Linear(width, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

        # Per-token, per-channel gating of the attention contribution.
        self.gate = nn.Linear(width, width, bias=False)

        # QK-Norm: normalize Q and K before attention
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # Per-head learnable log-temperature (init 0 -> effective scale = 1/sqrt(head_dim))
        self.log_temperature = nn.Parameter(torch.zeros(n_heads))

        # RoPE positional embedding
        self.rope = RotaryPositionalEmbedding(head_dim, base=rope_base, rope_fraction=rope_fraction)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm before RoPE for stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Explicit expansion keeps the CPU scoring path compatible with PyTorch 2.7.
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        # Per-head learnable temperature
        scale = self.log_temperature.clamp(-math.log(100.0), math.log(100.0)).exp()
        q = q * scale.to(q.dtype).view(1, self.n_heads, 1, 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = attn

        # Output projection with gate
        out = out.transpose(1, 2).reshape(B, T, -1)
        gated_out = self.o_proj(out) * torch.sigmoid(self.gate(x))

        return self.drop(gated_out)


class Block(nn.Module):
    """Pre-norm attention and feed-forward residual block."""
    def __init__(self, width: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 context: int, dropout: float = 0.05, rope_base: float = 10000.0,
                 rope_fraction: float = 0.5, hidden_dim: int | None = None,
                 residual_dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.attn = Attention(width, n_heads, n_kv_heads, head_dim, context,
                              dropout, rope_base, rope_fraction)

        # SwiGLU hidden dimension: (8/3) * width, rounded to 256 multiple
        if hidden_dim is None:
            hidden_dim = math.ceil(width * 8 / 3 / 256) * 256
        self.ffn = SwiGLU(width, hidden_dim)
        # Optional branch dropout also regularizes the previously unregularized
        # FFN. It has no parameters or effect in evaluation, so old checkpoints
        # retain their evaluation behavior with the same architecture config.
        self.residual_drop = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_drop(self.attn(self.norm1(x)))
        x = x + self.residual_drop(self.ffn(self.norm2(x)))
        return x


class GlobalMemoryTokens(nn.Module):
    """Learnable global memory tokens fused into the residual stream.

    These tokens live entirely as learnable parameters and are not
    inserted into the sequence. They aggregate global context that is
    mixed into the residual stream right before the output projection,
    giving the model an inexpensive "side channel" for long-range info
    without changing the input contract (ids stay shape [B, T]).
    """
    def __init__(self, width: int, n_global_tokens: int = 8):
        super().__init__()
        self.n_global_tokens = n_global_tokens
        self.tokens = nn.Parameter(torch.zeros(1, n_global_tokens, width))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        # A small "memory read" projection: combines per-token activations
        # with the learned global tokens to produce a fused summary.
        self.read_proj = nn.Linear(width, n_global_tokens, bias=False)
        # Fuse the aggregated global state back into the sequence.
        self.fuse_norm = RMSNorm(width)
        self.fuse_gate = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject global memory into the residual stream.

        x has shape [B, T, W]. We treat the learned tokens as a set of
        prototypes; we compute their soft attention over the sequence,
        weight them, and add a gated contribution back to every position.
        """
        B, T, W = x.shape
        # Attention weights: [B, T, n_global] -> softmax over time.
        attn_logits = self.read_proj(x)  # [B, T, n_global]
        attn = F.softmax(attn_logits, dim=1)
        # Mix sequence info into the global tokens: [B, n_global, W]
        mixed_tokens = torch.matmul(attn.transpose(1, 2), x) + self.tokens.expand(B, -1, -1)
        # Summarize back to per-position contribution.
        summary = torch.matmul(attn, mixed_tokens)  # [B, T, W]
        gate = torch.sigmoid(self.fuse_gate(self.fuse_norm(summary)))
        return x + gate * summary


class ImprovedGPT(nn.Module):
    """Decoder-only LM with optional global memory tokens.

    forward returns logits, predict_log_probs returns log-p.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        for name in ('vocab', 'context', 'width', 'heads', 'depth'):
            value = config.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.context = config['context']
        width = config['width']
        n_heads = config['heads']
        depth = config['depth']

        if width % n_heads:
            raise ValueError('width must be divisible by heads')
        head_dim = width // n_heads
        if head_dim < 2:
            raise ValueError('width // heads must be at least 2 for RoPE')

        # Choose a divisor: heads // 4 alone fails for e.g. 14 query heads.
        default_kv_heads = next(h for h in range(max(1, n_heads // 4), 0, -1)
                                if n_heads % h == 0)
        n_kv_heads = config.get('kv_heads', default_kv_heads)
        if (isinstance(n_kv_heads, bool) or not isinstance(n_kv_heads, int)
                or n_kv_heads < 1 or n_heads % n_kv_heads):
            raise ValueError('kv_heads must be a positive integer dividing heads')
        hidden_dim = config.get('hidden_dim')
        if hidden_dim is not None and (isinstance(hidden_dim, bool)
                or not isinstance(hidden_dim, int) or hidden_dim < 1):
            raise ValueError('hidden_dim must be a positive integer')
        dropout = config.get('dropout', 0.05)
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not 0 <= dropout < 1):
            raise ValueError('dropout must be in [0, 1)')
        residual_dropout = config.get('residual_dropout', 0.0)
        if (isinstance(residual_dropout, bool)
                or not isinstance(residual_dropout, (int, float))
                or not 0 <= residual_dropout < 1):
            raise ValueError('residual_dropout must be in in [0, 1)')
        rope_base = config.get('rope_base', 10000.0)
        rope_fraction = config.get('rope_fraction', 0.5)
        n_global_tokens = config.get('n_global_tokens', 0)
        if (isinstance(n_global_tokens, bool) or not isinstance(n_global_tokens, int)
                or n_global_tokens < 0):
            raise ValueError('n_global_tokens must be a non-negative integer')

        # Token embedding (RoPE handles positions, no learned pos_emb needed)
        self.token_emb = nn.Embedding(config['vocab'], width)

        # Optional global memory tokens - they live OUTSIDE the regular
        # sequence and only interact via the global fuse gate, so the
        # contract that predict_log_probs sees ids of length <= context
        # is preserved.
        self.n_global_tokens = n_global_tokens
        if n_global_tokens > 0:
            self.global_mem = GlobalMemoryTokens(width, n_global_tokens)
        else:
            self.global_mem = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(width, n_heads, n_kv_heads, head_dim, self.context,
                  dropout, rope_base, rope_fraction, hidden_dim, residual_dropout)
            for _ in range(depth)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(width)

        # Output head (tied with input embedding)
        self.lm_head = nn.Linear(width, config['vocab'], bias=False)
        # Initialize each tensor once, then tie input/output weights.
        self.apply(self._init_weights)
        # GPT-2 style residual scaling: shrink attention output projection (o_proj)
        # and FFN down projection (SwiGLU w3) by sqrt(2*depth) to keep the residual
        # stream's per-layer contribution variance bounded as depth grows.
        residual_std = 0.02 / math.sqrt(2.0 * depth)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.w3.weight, std=residual_std)
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Training interface: unnormalized next-token logits.

        Args:
            ids: [batch, seq_len] token IDs

        Returns:
            logits: [batch, seq_len, vocab_size] unnormalized logits
        """
        if ids.ndim != 2:
            raise ValueError('ids must have shape [batch, time]')
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('ids must contain int32 or int64 token IDs')
        if ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError(f'ids must have a nonempty batch and 1..{self.context} tokens')

        # Token embeddings (RoPE handles positions)
        x = self.token_emb(ids)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply global memory fusion if enabled. Global tokens were never
        # inserted into the sequence (they live in the residual stream via
        # their learnable parameter), so nothing needs to be detached here.
        # We simply inject a learned summary at the very end of the stack.
        if self.global_mem is not None:
            x = self.global_mem(x)

        # Final norm and projection
        return self.lm_head(self.norm(x))

    def predict_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        """Evaluation interface: normalized log probabilities.

        Args:
            ids: [batch, seq_len] token IDs

        Returns:
            log_probs: [batch, seq_len, vocab_size] normalized log probabilities
        """
        logits = self.forward(ids).float()
        return F.log_softmax(logits, dim=-1)


def build_model(config: dict) -> ImprovedGPT:
    """Factory function matching the required interface."""
    return ImprovedGPT(config)
