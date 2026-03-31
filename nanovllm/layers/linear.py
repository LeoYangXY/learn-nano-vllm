import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

#每个chip上存一份完整的权重矩阵
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)



# 核心思想：将权重矩阵 W 按列拆分。
# 拆分方式：将 W 拆分为 [W1, W2]。
# 计算逻辑：每个 GPU 持有相同的输入 X，分别计算出一部分输出特征。
# 公式：Y = X[W1, W2] = [XW1, XW2]
# 通信需求：
#   - 前向传播：每个 GPU 计算出 Y 的一部分。如果下一层需要完整的 Y，则需要进行一次 All-Gather 操作。不过也可以保持拆分状态进入下一层（通常做法）。
#   - 反向传播：需要进行一次 All-Reduce 来同步梯度。
# 优点： 适合作为多层感知机（MLP）的第一层，因为它不需要对输入X进行复杂的切分。
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


#这里做的就是先合并，然后切分：
#我们一般做的是X*W^T，X的维度是[batch_size, in]，W的维度是[out,in]
#如果有W1^T为[in,out1],W2^T为[in,out2], W3^T为[in,out3]，我们要分别做做X*W1^T,X*W2^T,X*W3^T，可以考虑如下的方式去加速计算(将这几个权重矩阵合并成一个巨大的矩阵，一次性完成乘法操作)：
#我们先把它们合并成一个大矩阵W^T=[W1^T, W2^T, W3^T]，维度是[in,out1+out2+out3]，去算X*W^T即可。
#而对于此，我们又可以使用col方向的tp：切分成tp_size份，每份的维度是[in,(out1+out2+out3)/tp_size]，每个chip上存一份这样的切分后的矩阵。
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 用于应对MQA，GQA这样的复杂场景：Q矩阵的头数和K，V矩阵的头数不一样，以及每个chip上要存各自的Q，K，V
# 核心逻辑：先将Q、K、V权重合并为一个大矩阵，再按列把Q，K，V切分到不同GPU

# 场景设定：
#     hidden_size = 4096          # 隐藏层维度
#     head_size = 128             # 单头维度
#     total_num_heads = 32        # Q的头数（MHA的Q）
#     total_num_kv_heads = 8      # K/V的头数（GQA的K/V）
#     tp_size = 4                 # 张量并行切分数量

# 1. 宏观上构造[Q_part + K_part + V_part]的连续显存块（总宽度6144）
# 2. 微观上每个GPU存储[Q_slice + K_slice + V_slice]（宽度1536）

# 内存布局（逻辑上的大矩阵，总宽度6144）：
#     |<------- Q (32头=4096列) ------->|<--- K (8头=1024列) --->|<--- V (8头=1024列) --->|
#     |..................................................................................|
#     | 0                              4096                      5120                    6144 (列索引)
    
#     切分后每个GPU存储1536列（6144/4），包含Q、K、V的分片：
#     GPU 0: [Q:0-1024, K:1024-1280, V:1280-1536]
#     GPU 1: [Q:1536-2560, K:2560-2816, V:2816-3072]
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,#这个是指Q矩阵的头数
        total_num_kv_heads: int | None = None,#这个是指K，V矩阵的头数
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 行并行 (Row Parallelism)
# 核心思想：将权重矩阵 W 按行拆分。
# 拆分方式：将 W 拆分为上下两部分，即 W1 和 W2。
# 计算逻辑：为了能进行矩阵乘法，输入 X 也必须按列拆分为 [X1, X2]。
# 公式：Y = [X1, X2] * [[W1], 
#                     [W2]] = X1W1 + X2W2
# 通信需求：
#   - 前向传播：每个 GPU 分别计算出部分和 XiWi，最后必须进行一次 All-Reduce 操作将结果相加，得到最终的 Y。
#   - 反向传播：梯度通过 All-Reduce 后的算子自然分发。
# 优点：能够直接缩减中间特征图的维度，常作为 MLP 的第二层。
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对于含有bias的处理：
        # 行并行的数学公式计算为：
        # Y = [X1, X2] * [[W1], [W2]] + bias = (X1W1 + bias) + (X2W2 + bias) = X1W1 + X2W2 + 2 * bias
        # 问题描述：对于行并行，每个并行的 linear 都会加上 bias，统一相加的时候，就会多加 tp_size - 1 份 bias。
        # 解决方法：只让其中一个 linear 计算 bias，比如 tp_rank=0 的保留它原来的 bias 值，其他的全部置 None，然后用一个 all_reduce 来同步计算出来的 y。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
