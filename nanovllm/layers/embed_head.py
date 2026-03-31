import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

#embedding语义：Embedding 层其实就只是一个权重矩阵 W ，然后根据输入的 token_id 去取第 token_id 行
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    #最终目的是把那个“大的 loaded_weight”中属于当前 GPU 的那一部分，搬运到了 param 中。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)#去算出这个global token_id在每个chip上的local token_id
        y = F.embedding(x, self.weight)#每个chip用local token_id去embedding矩阵上取值，得到一个局部的embedding向量
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y#根据mask把不属于当前GPU的token的embedding向量置为0向量

            # tp:就是一个chip存权重的一部分。激活值来了之后,我们要考虑多个chip之间怎么进行配合
            # 原理：对于某个 token，只有一个 GPU 算出了它的真实 Embedding，其他 GPU 都是 0。
            # 使用all_reduce后，所有 GPU 都得到了该 token 的完整 Embedding 向量。
            dist.all_reduce(y)
        return y




#LMHead语义：LMHead 就是语言模型的输出层，拿到一个向量之后，与词表的各个向量做一个点积，得到每个词的得分，得分最高的那个词就是模型预测的下一个词
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)#根据自己传入的num_embeddings和embedding_dim,继承了VocabParallelEmbedding的__init__，做了初始化


#用户只发一个句子，但服务器会把几百个用户的句子凑成一个 Batch
# vLLM 的核心作用就是“拼单”。它会在后台维护一个全局的大 Batch。它是如何把“单句”变成“Batch”的？
# 假设此时此刻，服务器同时收到了 4 个用户的请求：
# 用户 A 输入：“你好”（长度 2）
# 用户 B 输入：“今天天气不错”（长度 5）
# 用户 C 输入：“写首诗”（长度 3）
# 用户 D 输入：“Python教程”（长度 4）
# vLLM 的调度器（Scheduler）会立刻行动：
# 1.收集：把这 4 个请求抓过来。
# 2.拍平：把它们拼成一个长度为 2+5+3+4=14 的长向量 x，里面存储了所有 token 的 id。
# 3.记录：生成 cu_seqlens_q = [0, 2, 7, 10, 14]，记下每个人的边界。
# 4.执行：把这 14 个 token 一起塞进 GPU 跑 Prefill。
# 这就是你代码里那个 batch_size==4（或者更大）的由来——它是多个用户的集合体。

# 🔄 生成阶段（Decode）的 Batch
# 等 Prefill 跑完了，进入生成阶段，这个 Batch 依然存在，但会动态变化：
# 第一轮生成：A, B, C, D 都在等结果，Batch 里还是这 4 个人。
# 第二轮生成：
# 用户 C 的句子短，生成完了（“床前明月光”），退出 Batch。
# 用户 E 刚发来新请求“你好吗”，插入 BATCH。
# 此时 Batch 里是：A, B, D, E。
# 这就是 vLLM 的连续批处理（Continuous Batching）技术

#对于含有多个sentence的一个batch的处理：比如batchsize==3，sentence_length分别是3，2，4
#在 vLLM 这种高性能框架中，为了极致的计算效率，它通常不喜欢处理参差不齐的“锯齿状”数据，而是喜欢把所有句子拍平（Flatten）成一条长长的直线
#在显存里，x 并不是一个[3,4]的矩阵（那样会浪费很多空间填充 0），而是一个长度为 3+2+4=9 的一维向量，里面存储了所有 token 的 id
#x=[sentence1_token1_id, sentence1_token2_id, sentence1_token3_id, sentence2_token1_id, sentence2_token2_id, sentence3_token1_id, sentence3_token2_id, sentence3_token3_id, sentence3_token4_id]
#cu_seqlens_q 的全称是 Cumulative Sequence Lengths（累积序列长度）。它就像是一个路标，记录了每个句子结束的位置,在上面的例子中，cu_seqlens_q 就是 [0, 3, 5, 9]，表示第一个句子结束在位置 3，第二个句子结束在位置 5，第三个句子结束在位置 9
    def forward(self, x: torch.Tensor):
        context = get_context()
        #prefill就是根据prompt生成首token，跟decode的区别就是要提取出每个sentence的最后一个token去进行生成。而decode本身的输入就是单个单个token的，所以不需要提取最后一个token了。
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1#切出每个句子最后一个 token 的位置，得到 last_indices=[2,4,8]
            x = x[last_indices].contiguous()#把这个batch中的多个sentence的最后一个token紧密排到一起
        #此时拿到了batchsize个token（prefill阶段提取出来，decode阶段的输入便是这样的），然后我们要去算出这个batch中每个token的下一个token的得分了。比如说，假设这个batch里有3个token，分别是“你”，“今”，“写”，我们要算出这3个token的下一个token的得分，也就是“你”的下一个token的得分，“今”的下一个token的得分，“写”的下一个token的得分。
        logits = F.linear(x, self.weight)#self.weight是从VocabParallelEmbedding继承来的权重矩阵，维度是 [num_embeddings_per_partition, embedding_dim]，x的维度是 [batch_size, embedding_dim]，这样子就根据每个chip的local词表，算出这个batch中的每个sentence后续要得到的下一个token了
        if self.tp_size > 1:
            # 每个 rank 只拥有一段词表，所以这里只能先得到“局部 logits”。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 汇总到 rank0：rank0 负责拼接完整词表 logits，并在后续执行采样的时候，只有rank0去生成（见model_runner.py），别的rank不生成
            # 非 rank0 作为计算 worker，不需要持有完整 logits。
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
