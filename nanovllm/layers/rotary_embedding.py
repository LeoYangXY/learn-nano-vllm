import torch
from torch import nn

# RoPE（Rotary Position Embedding）数学公式：
# 
# 设 x = [x_1, x_2, ..., x_d] 是一个 head 的向量，d 偶数。
# RoPE 对每对相邻分量 (x_{2i}, x_{2i+1}) 做二维旋转：
# 
#   [x'_{2i}  ]   [ cos(θ_p)     -sin(θ_p) ]   [ x_{2i}   ]
#   [x'_{2i+1}] = [ sin(θ_p)      cos(θ_p) ] * [ x_{2i+1} ]
# 
# 其中 θ_p = p * 1.0 / (base^{2i/d})，p 是位置，base 通常为 10000。
# 
# 代码实现等价于：
#   x1, x2 = chunk(x, 2)
#   y1 = x1 * cos - x2 * sin
#   y2 = x2 * cos + x1 * sin
#   y = cat(y1, y2)
#rope的数学函数
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    #这里主要是在做RoPE的预计算，预先把每个位置的cos和sin计算好，放在一个cache里，后续每次调用forward的时候直接取出来用就好了。
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)#register_buffer这个api的作用是把这个cache注册成模型的一个buffer，这样子它就会随着模型一起保存和加载，但它不会被当作模型的参数来更新。

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


_ROPE_CACHE: dict[tuple[int, int, int, float, tuple | None], RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if rope_scaling is None:
        scaling_key = None
    else:
        scaling_key = tuple(sorted(rope_scaling.items()))

    key = (head_size, rotary_dim, max_position, float(base), scaling_key)
    if key not in _ROPE_CACHE:
        # Current implementation only supports default RoPE behavior.
        # Keep runtime compatible even when newer HF configs include rope_scaling.
        _ROPE_CACHE[key] = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return _ROPE_CACHE[key]
