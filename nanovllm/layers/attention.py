import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

#里面直接调用了flash-attention的kernel，以及自己手动去管理了一下kv cache

# 这里传入的是当前step计算出来的key，value向量。
# 具体来说，传入的数据通常是以下两种情况之一：
# 🆕 预填充阶段
# 传入内容：用户输入的 Prompt 被切分成了很多个 Chunk，这里传入的是当前这一个 Chunk 的 K 和 V。
# 例子：你输入了“你好，我是人工智能”，系统可能一次只处理“你好，我是”这 4 个 token。
# 此时 N = 4。
# 🔄 解码阶段
# 传入内容：模型刚刚生成的那 1 个新 token 的 K 和 V。
# 例子：模型刚吐出一个“大”字。
# 此时 N = 1。
@triton.jit
def store_kvcache_kernel(
    key_ptr,#这一小段输入对应的 Key 向量，形状是[N,D]，其中 N 是当前 step 计算出来的 key 的数量（预填充阶段是一个 chunk 的 token 数量，解码阶段是 1），D 是每个 token 的词向量维度
    key_stride,#key中的2个token之间的跨度，一般是D
    value_ptr,
    value_stride,
    k_cache_ptr,#整个K缓存池的起始地址
    v_cache_ptr,
    slot_mapping_ptr,#槽位映射表的指针。这是一个数组，长度等于当前处理的 token 数量。slot_mapping_ptr[i] 的值表示第 i 个 token 在 KV cache 中对应的槽位（slot）
    D: tl.constexpr,#每个token的词向量维度
):
    #任务划分：一个block处理一个token。
    idx = tl.program_id(0)
    
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    cache_offsets = slot * D + tl.arange(0, D)#根据slot计算出这个token在KV cache中对应的位置
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)#取出这个block负责的token所对应的这个长度为D的向量
    value = tl.load(value_ptr + value_offsets)
    
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape#这里的N是动态获取的
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    #如果有N个token，那么就对应的启动N个block
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill: #这是处理用户输入的 Prompt 阶段的逻辑
            #TODO:xuanyiyang:这里为什么是使用k_cache和v_cache去覆盖掉当前的k和v呢？
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            #prefill阶段：传入根据prompt生成的完整的Q，K，V
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else: #这是模型生成文字阶段的逻辑
            #decode阶段：传入Q和缓存指针K_cache,V_cache
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
