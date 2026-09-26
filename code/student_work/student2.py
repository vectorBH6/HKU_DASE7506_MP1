"""MP1 提交模型：参考 baseline 逐步演进的 GPT 变体，配套 configs/student2.json。

演进路径（逐块理解）：
  1. baseline（model.py）                          : Embedding + LayerNorm + SDPA + GELU MLP
  2. + RMSNorm（替代 LayerNorm）                   : 无均值中心的归一化，bf16 下数值更稳定
  3. + RoPE（替代可学习位置编码）                 : 旋转位置编码，q/k 通过几何旋转注入位置信息
  4. + QK-Norm                                     : 对每个 head 的 q/k 做 RMSNorm，帮助注意力稳定
  5. + SwiGLU / GELU MLP 切换                     : SwiGLU: down(silu(gate)*up)，GELU: down(gelu(up))
  6. + dropout（训练时生效，eval 时关闭）          : attention + 残差分支
  7. + tied embedding head + 残差分支缩放初始化   : 减少参数量 + 改善训练初期梯度

最终架构（configs/student2.json）：
  vocab 2048, width 192, depth 8, heads 6 (head_dim=32), context 256,
  MLP hidden=512, dropout=0.15, tied embedding, 约 3.94M 参数。
  最终配置使用 mlp='gelu'（在此规模上比 SwiGLU 低 0.012 val BPB）。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


# ─── 组件层 ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """RMSNorm：无均值中心的归一化。

    y = x / sqrt(mean(x^2) + eps) * weight
    与 LayerNorm 的区别：不减均值，对梯度更友好；在 bf16 训练下数值更稳定。
    统计量全程在 fp32 中计算，再 cast 回输入 dtype，保证 bf16 安全。
    """

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放因子（对应 LayerNorm 的 weight/gamma）
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def rope_tables(context, head_dim, base=10000.0):
    """预计算 RoPE cos/sin 表，形状 [context, head_dim]。

    RoPE 的核心思想：给 q/k 的每对通道 (j, j+head_dim/2) 乘一个随位置 t 旋转的角度。
    角度公式：theta_j = base^(-2j/head_dim)，j 越大（高频通道）衰减越快。
    旋转后的 q/k 做点积时结果只与相对位置有关，自然获得长度外推能力。

    表结构：每行 t 包含所有通道的角度（cos, sin 交错拼在一起），
    与 apply_rope 中的 rotate-half 方式配合使用。
    """
    i = torch.arange(0, head_dim, 2).float()
    thetas = base ** (-i / head_dim)          # 各通道的角速度
    t = torch.arange(context).float()         # 位置 0..context-1
    angles = t[:, None] * thetas[None, :]     # [context, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)  # [context, head_dim] 偶/奇各一份
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """对 q/k 的最后两维做 rotate-half RoPE。

    公式：out = x * cos + rotate_half(x) * sin
    rotate_half：把向量后半部分取负放到前，前半部分放到后。
    效果：对调每个通道对 (j, j+half) 的相对顺序，等价于旋转 angle_j 角度。
    """
    d = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    return x * cos + x_rot * sin


# ─── Transformer Block ──────────────────────────────────────────────────────

class Block(nn.Module):
    """Pre-norm Transformer Block：每个残差分支前做 RMSNorm。

    数据流（x: [B, T, width]）：
      qkv = norm1(x) @ W_qkv
      q, k, v = qkv.split(width, dim=-1)
      q, k = q_norm(q), k_norm(k)          # QK-Norm（可选）
      q, k = apply_rope(q, k)               # RoPE（可选，注入相对位置）
      attn = SDPA(q, k, v, causal)         # Flash 风格高效注意力
      x = x + dropout(proj(attn))           # 注意力残差

      x = x + dropout(mlp(norm2(x)))        # MLP 残差

    所有 Linear 层不带 bias（与 GPT-2 一致），节省参数量和计算。
    RoPE 表用 non-persistent buffer，不进入 state_dict，但仍随模块迁移到目标设备。
    """

    def __init__(self, width, heads, mlp_hidden, context,
                 use_rope=True, mlp='swiglu', use_qk_norm=True, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.use_rope = use_rope
        self.mlp_kind = mlp
        self.dropout_p = dropout

        # Pre-norm：每个残差分支前归一化（比 post-norm 更利于 bf16 训练）
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)

        # QKV 投影 + 输出投影，均无 bias
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

        head_dim = width // heads
        # QK-Norm：在送入注意力前对每个 head 的 q/k 做 RMSNorm，
        # 让各 head 的激活尺度一致，防止某些 head 主导注意力
        self.q_norm = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if use_qk_norm else nn.Identity()

        # MLP：两种形式参数数量相近
        #   SwiGLU: gate(x) * up(x) 再 down，3 个 Linear
        #   GELU  : up(x) 再 gelu 再 down，2 个 Linear（hidden 扩大 3/2 以匹配参数量）
        if mlp == 'swiglu':
            self.gate = nn.Linear(width, mlp_hidden, bias=False)
            self.up   = nn.Linear(width, mlp_hidden, bias=False)
            self.down = nn.Linear(mlp_hidden, width, bias=False)
        else:  # gelu
            self.up   = nn.Linear(width, mlp_hidden * 3 // 2, bias=False)
            self.down = nn.Linear(mlp_hidden * 3 // 2, width, bias=False)

        # RoPE 表：预计算后注册为 buffer，persistent=False 不写入 state_dict
        if use_rope:
            cos, sin = rope_tables(context, head_dim)
            self.register_buffer('cos', cos, persistent=False)
            self.register_buffer('sin', sin, persistent=False)

        # Dropout 仅在训练时生效（eval 时 F.scaled_dot_product_attention 的 dropout_p=0）
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape

        # QKV 投影并拆分
        qkv = self.qkv(self.norm1(x))                     # [B, T, 3*width]
        q, k, v = qkv.view(batch, length, 3, self.heads,
                           width // self.heads).permute(2, 0, 3, 1, 4)
        # q, k, v: [3, B, heads, T, head_dim]

        # QK-Norm + RoPE
        if self.use_rope:
            q = apply_rope(self.q_norm(q), self.cos[:length].to(q.dtype),
                           self.sin[:length].to(q.dtype))
            k = apply_rope(self.k_norm(k), self.cos[:length].to(k.dtype),
                           self.sin[:length].to(k.dtype))
        else:
            q, k = self.q_norm(q), self.k_norm(k)

        # SDPA：PyTorch 2.0 内置 Flash Attention 实现，causal=True 自动做下三角掩码
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0)

        # 注意力残差
        x = x + self.drop(self.proj(attn.transpose(1, 2).reshape(batch, length, width)))

        # MLP 残差
        h = self.norm2(x)
        if self.mlp_kind == 'swiglu':
            return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))
        return x + self.drop(self.down(F.gelu(self.up(h))))


# ─── 模型组装 ──────────────────────────────────────────────────────────────

class StudentGPT(nn.Module):
    """GPT 主体：Embedding + N x Block + Norm + Head。

    初始化顺序（刻意设计）：
      1. 所有 Linear/Embedding 权重 ~ N(0, 0.02)
      2. 残差分支输出（proj/down）重新缩放为 std = 0.02/sqrt(2*depth)
         → GPT-2 风格的缩放初始化，保证每个残差分支初期贡献接近 0，训练更稳定
      3. Head 权重与 Embedding 表绑定（tied），放在最后避免被步骤 2 覆盖
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']

        use_rope    = config.get("use_rope", True)
        mlp         = config.get("mlp", 'swiglu')
        use_qk_norm = config.get("use_qk_norm", True)
        dropout      = config.get("dropout", 0.0)
        width       = config['width']
        depth       = config['depth']
        mlp_hidden  = config.get('mlp_hidden', 4 * width)

        # Token embedding；位置编码：RoPE 模式不需要可学习表
        self.token = nn.Embedding(config['vocab'], width)
        self.pos   = None if use_rope else nn.Embedding(self.context, width)

        # N 个 Transformer Block
        self.blocks = nn.ModuleList([
            Block(width, config["heads"], mlp_hidden, self.context,
                  use_rope=use_rope, mlp=mlp, use_qk_norm=use_qk_norm, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)

        # 步骤 1：所有权重初始化为 N(0, 0.02)
        self.apply(self._init_weights)

        # 步骤 2：残差分支输出缩放（GPT-2 scaled init）
        residual_std = 0.02 / math.sqrt(2 * depth)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.down.weight, std=residual_std)

        # 步骤 3：绑定 head 与 embedding 表（tied weight），放最后防止被覆盖
        self.head.weight = self.token.weight

    @staticmethod
    def _init_weights(module):
        """对所有 Linear / Embedding 层统一应用 N(0, 0.02) 初始化。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, ids):
        """提取 token 表征：Embedding → Blocks → RMSNorm → [B, T, width]."""
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """训练接口：返回未归一化的 logits [B, T, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """评估接口：fp32 归一化 log-probs；每次调用无状态（窗口之间不保留记忆）。"""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """入口函数，供 train.py 通过 make_model 调用。"""
    return StudentGPT(config)
