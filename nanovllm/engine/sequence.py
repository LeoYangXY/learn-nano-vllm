from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

# Sequence数据结构对应每个输入的prompt。
# 在后面的scheduler、ModelRunner、BlockManager和LLMEngine都不是直接处理经过tokenizer encode后得到的List[int]或者未处理的str，而是Sequence这个数据结构
class Sequence:
    block_size = 256 #一个block会存储256个token的kvcache
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]

        # 表示当前 Sequence 里所有 token 的总数，包括 prompt token 和后续生成的 token（即整个 token_ids 列表的长度）。
        # 在推理过程中，每生成一个新 token，这个值会递增。
        self.num_tokens = len(self.token_ids)

        # 表示最初 prompt 的 token 数量（即输入时的 token 数），只在初始化时设置，后续不会变化。
        # 只包含 prompt 部分，不包括后续生成的 token。
        self.num_prompt_tokens = len(token_ids)

        # kvcache已缓存的token数目
        self.num_cached_tokens = 0


        # 在大模型推理时，KV cache（注意力的 key/value 缓存）通常会被分成很多小块（block），每个 block 存储一段 token 的 KV 信息（比如 256 个 token）。
        # 一个 Sequence 可能很长，需要多个 block 来存储它的全部 KV cache。
        # self.block_table 就是一个列表，里面每个元素是一个 block 的编号（block_id），这些编号由 BlockManager 分配。
        self.block_table = []
        
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):#这个属性表示这个 Sequence 需要多少个 block 来存储它的 KV cache
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获取第 i 个逻辑 block（分段）的 token 列表。
    # 一个 Sequence 的所有 token 会被按照顺序分成若干 block，每个 block 最多 block_size 个 token。
    # 例如：
    #   假设 block_size=4，token_ids=[10,11,12,13,14,15,16,17,18]
    #   那么分块如下：
    #   block(0): [10,11,12,13]
    #   block(1): [14,15,16,17]
    #   block(2): [18]
    #
    #   ┌─────────────┬─────────────┬───────┐
    #   │ block(0)    │ block(1)    │block(2)
    #   └─────────────┴─────────────┴───────┘
    #   [10,11,12,13,14,15,16,17,18]
    #
    # 这样方便后续 KV cache 的分块管理和复用。
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 用于在分布式环境下的多进程间传递 Sequence 对象时，pickle 会调用 __getstate__ 和 __setstate__ 方法来序列化和反序列化对象。
    # pickle.dumps(obj)：把 Python 对象 obj 变成一串二进制数据（序列化）。pickle.loads(data)：把这串二进制数据还原成原来的 Python 对象（反序列化）
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
