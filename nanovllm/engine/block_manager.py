"""
BlockManager —— KV cache 物理块管理器（Paged Attention 的底座）
==================================================================

核心思想（Paged Attention）：
-------------------------------
传统连续 KV cache：每个 Sequence 预分配 [max_seq_len, num_layers, heads, dim] 的
连续显存 —— 绝大多数 token 没用到时就是浪费。
Paged Attention：把 KV 切成固定大小 block（这里是 256 token/block），
每个 Sequence 持有一个 **block_table**（逻辑块 → 物理块编号的映射）。
物理块按需分配、按需释放，显存利用率接近 100%。

这里多干的一件事：Prefix Caching

prefix cache是比如SequenceA在自己的prefill和不断decode中，积累的那些block，可以给SequenceB在prefill的时候使用复用
但是不能给SequenceB在decode的时候使用复用

-----------------------------------------------------------------
每个物理块上还挂了：
  - hash：由 `(prev_block_hash, token_ids)` 计算出来的链式哈希；
  - token_ids：这块里的原始 token 序列，用于哈希冲突兜底校验。

配合一个全局 `hash_to_block_id` dict，新请求来的时候会逐块算 hash 去查：
  - 命中且 token_ids 一致 → 复用（可能 ref_count++，也可能从 free 池"认领"回来）；
  - 没命中 / hash 冲突 → 分配新 block，GPU 那边算完再把 KV 写进去。

关键不变量：
-------------
1. block.ref_count == 0  ↔  block_id ∈ free_block_ids（互斥）
2. 活跃 block 的 hash 非 -1 时，必定能在 hash_to_block_id 里找到自己
3. 未满块的 hash 恒为 -1，且绝不进入 hash_to_block_id：
   - prefill（allocate）：仅当 len(token_ids) == block_size 才 compute_hash；
     末块不足一块时 h = -1，且只在 if h != -1 时才登记，故半满块不入表。
   - decode（may_append）：仅 len%block_size == 0（刚填满）那步才 compute_hash 并登记；
     ==1 新分配的块与 else 填充中的块均 assert hash == -1，不算哈希。
   - 原因：哈希服务于"复用"——其他请求靠"相同 token 序列 → 相同哈希"来命中本块 KV。
     半满块内容未定型，后续还会被 append_token 追加 token，此刻算出的哈希在补满后即失效。
     因此仅当 token 数 == block_size、内容最终确定时哈希才稳定、才有复用价值；
     半满块 hash = -1，意味着它无法被其他请求认领。

与其他组件的交互：
--------------------
- Scheduler.schedule() 调 can_allocate / allocate（prefill 入场）
- Scheduler.schedule() 调 can_append / may_append（decode 新写一个 token）
- Scheduler.postprocess() 在 seq finished 时调 deallocate
- ModelRunner.prepare_prefill/decode 用 seq.block_table 构造 slot_mapping
- Attention kernel 用 slot_mapping 知道 KV 写到哪、用 block_tables 知道 KV 读哪里

Phase 5 将替换为 Radix Trie 版本，支持 token 级（不是块级）partial prefix 匹配。
"""
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

#这是物理block，而不是Sequence里面的那个逻辑block
class Block:

    def __init__(self, block_id):
        self.block_id = block_id #全局唯一的block编号
        self.ref_count = 0 #引用计数，表示有多少个Sequence在使用这个Block
        self.hash = -1 #哈希值，prefix cache时用于匹配前缀符合的Block
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 列表中的每一个元素代表一个物理内存块。列表的索引就是块的ID
        self.hash_to_block_id: dict[int, int] = dict() # 创建哈希值->block_id对应的字典，在后续的前缀复用时找到可复用的Block 
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()


    # 计算当前 block 的哈希值，支持链式哈希：
    # - 如果 prefix != -1，说明有前缀上下文，需要把前一个 block 的哈希值也加入哈希计算，实现链式哈希。
    # - 这样可以保证只有“前缀完全一致且当前 block token 完全一致”的 block，才会得到相同的哈希值，实现严格的 KV cache 复用。
    # - token_ids 会被转成连续的二进制字节流，保证哈希的唯一性和可复现性。
    #
    # 举例（block_size=4，注意 hash 是【链式】的：第 i 块 = hash(第 i-1 块的 hash, 本块 token)）：
    #   序列A: [1,2,3,4 | 5,6,7,8]
    #   序列B: [1,2,3,4 | 5,6,7,8 | 9,10,11,12]
    #   序列C: [0,2,3,4 | 5,6,7,8 | 9,10,11,12]
    #
    #   序列A:  h0_A = hash([1,2,3,4])              (prefix=-1)
    #           h1_A = hash(h0_A, [5,6,7,8])
    #
    #   序列B:  h0_B = hash([1,2,3,4]) = h0_A       ← 第 0 块内容同 A
    #           h1_B = hash(h0_B, [5,6,7,8]) = h1_A ← 第 1 块前缀+内容同 A → hash 完全相等，KV 可复用
    #           h2_B = hash(h1_B, [9,10,11,12])     ← 新增块，cache miss，需重新算
    #           ⇒ B 前两块命中 A 的 KV cache，只需为第 3 块计算并存储
    #
    #   序列C:  h0_C = hash([0,2,3,4]) ≠ h0_A       ← 第 0 块第一个 token 就不同，hash 已分叉
    #           h1_C = hash(h0_C, [5,6,7,8])        ← 注意！本块【内容】和 A 的 [5,6,7,8] 一样，
    #                                                   但因为 prefix(=h0_C) ≠ h0_A，所以 h1_C ≠ h1_A
    #           ⇒ 链式哈希的精髓：仅凭"本块内容相同"不够，必须"前面所有块也都完全一致"才能复用。
    #              C 第 1 块虽然内容碰巧和 A 一样，但前缀对不上，因此无法复用 A 的 KV，必须重算。
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 这是分配一个空闲的物理block
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # 确保该block的引用次数为0
        assert block.ref_count == 0
        block.reset()
        # 从空闲队列中移除并添加到已使用的block集合中
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    #这里并不会擦除数据，只是逻辑上进行挪动，管理。被_deallocate的block就是可以被直接覆盖写数据
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # =========================================================================
    # 物理块的完整生命周期（理解下面 allocate / deallocate 的关键）
    # -------------------------------------------------------------------------
    # 一个物理块的"一生"通常经历三步，拿"序列A算出某块 → 序列B复用"举例：
    #
    #   ① 出生   : 序列A prefill，某逻辑块 cache miss
    #              → 抓 free_block_ids[0] 分配新块（allocate 138-141）
    #              → GPU 算出 KV 写进显存
    #
    #   ② 回收   : 序列A 结束 → deallocate（162-169）
    #              → 块 ref_count 归零 → 退回 free_block_ids
    #              ※ 只动账本，不擦显存里的 KV，也不清 hash/token_ids
    #              ※ 这正是"前缀还活着、但块已空闲"的来源
    #
    #   ③ 认领   : 序列B 来了，前缀哈希命中这个块
    #              → 走 allocate 的 else 情况2（150-156）
    #              → _allocate_block 把它从 free_block_ids "捞回" used_block_ids
    #              → GPU 无需重算，直接复用那份还有效的 KV
    #
    # 关键点：②"回收"不是 allocate 时分配物理块的方式之一，而是 ③ 的前置条件。
    #        块之所以躺在 free_block_ids 里，正是因为它上一个主人结束了、被回收了。
    #        本实现中序列存续期间不会被中途驱逐(eviction)，回收只发生在 deallocate。
    # =========================================================================

    # 负责为一个新序列（prefill 入场）分配物理块并尽可能复用已有缓存。
    # 复用边界：本函数既"消费"也"生产"——SequenceB 来 prefill 时，
    #   可复用 SequenceA 在 prefill+decode 中累积登记的全部块（前提：前缀块级对齐）；
    #   decode 阶段不调用本函数，故 SequenceB 在 decode 时不会复用 A 的块（见 may_append）。
    def allocate(self, seq: Sequence):
        assert not seq.block_table#确保该序列是一个全新的、尚未分配任何物理块的请求
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):#相当于把这个序列使用一个block滑窗去进行分析
            token_ids = seq.block(i)#当前block滑窗下框出的token_ids列表

            #计算这一堆token_ids的哈希值
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1#如果是-1的话就不要去算了，因为-1的话代表这个block还没填满，因此先不考虑它的复用问题

            # 去看之前是否已经有过这个哈希值对应的block（但是可能存在的问题是：只是有这个hash值对应的block，但是实际上是出现了hash碰撞，导致两个不同的token_ids对应了同一个哈希值）
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 判断条件：
            # 1. block_id == -1：字典里根本没这个哈希值（彻底没算过）。
            # 2. self.blocks[block_id].token_ids != token_ids：哈希值撞车了（哈希冲突）。虽然哈希值一样，但里面的内容不一样，说明不是同一个东西。
            # 只有哈希值存在 且 内容完全一致，才算 Cache Hit
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # 这里的 Cache 指的是：这个 Block 里的 Token 对应的 KV 矩阵（键值对）已经在 GPU 显存里算好了，并且数据是现成的。
            # 如果不复用（Cache Miss）：GPU 需要拿着 Prompt 里的 token IDs，跑一遍模型（矩阵乘法），算出 Q、K、V，然后把 K 和 V 写入 到显存的 Block 里。
            # 如果复用（Cache Hit）：GPU 跳过 计算过程，直接把这个 Block 里的 K 和 V 拿来用。因为之前的请求已经算过并写入过了，现在只需要读取。
            if cache_miss:
                # 说明这串 Token 没人算过，必须从 free_block_ids 里取一个，分配新的显存，稍后 GPU 需要重新计算 KV 值填进去。随便拿个空闲块（不管里面原来有啥，反正都要被覆盖）
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else: # 我们在哈希表里找到了这个块的内容，现在要把这个物理块‘认领’过来给当前序列用
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 【情况1：块正被占用】/mode
                    # 别的序列（或同一个批处理里的其他请求）正在使用这个物理块。
                    # 我们只需要增加引用计数，表示“我也要加入共享这个块”。
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 【情况2：块在空闲池】
                    # 这个块被之前的一个Sequence算过，这个Sequence结束了，因此这个block被释放了，现在躺在 free_block_ids 里，可以被覆盖。
                    # 虽然数据还在，但状态是“空闲”。
                    # 我们需要调用 _allocate_block 把它从空闲池里“捞出来”，
                    # 标记为“正在使用”，防止后续分配把它覆盖掉。
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 倒序遍历（因为一个seq的后面的block可能复用前面的block） seq的block table，block的引用次数-1，
        # 如果引用次数为0则这个block已经没有被使用了，就直接释放
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # -------------------------------------------------------------------------
    # decode 阶段：每生成一个 token，len(seq) 增加 1，故 len(seq) % block_size
    # 随生成过程周期性变化，决定当前步是否需要新物理块：
    #   == 1 : 上一个块刚填满，当前 token 必须落到"新的一块" → 消耗 1 个 free 块
    #   == 0 : 当前块刚好填满 → 不消耗块，只需登记其哈希
    #   其他 : 仍在填充最后一个半块 → 不消耗块
    #
    # 单个物理块在 decode 中的生命周期（以 block_size=4 为例，一开始的prompt是4个token，因此 prefill 后已占满 B0）：
    #   decode步 | 新token | len | len%4 | block_manager 动作                          | 消耗free块
    #   ---------+---------+-----+-------+---------------------------------------------+-----------
    #     1      |   t5    |  5  |   1   | 从 free 池取新块 B1 追加，t5 写入(B1 半满)  |   是(1)
    #     2      |   t6    |  6  |   2   | 无动作，t6 写入 B1                          |   否
    #     3      |   t7    |  7  |   3   | 无动作，t7 写入 B1                          |   否
    #     4      |   t8    |  8  |   0   | B1 刚好填满，算哈希登记进 hash_to_block_id  |   否
    #     5      |   t9    |  9  |   1   | 从 free 池取新块 B2 追加，t9 写入(B2 半满)  |   是(1)
    #     6      |   t10   | 10  |   2   | 无动作，t10 写入 B2                         |   否
    #     ...    |   ...   | ... |  ...  | ...                                         |  ...
    # 即：块在 len%4==1 步被分配（半满，hash=-1）→ 之后每步仅写入 token、block_manager 不改账本
    #    → 在 len%4==0 步填满并登记哈希（转为可复用缓存）→ 归属本序列直至 deallocate 回收。
    # -------------------------------------------------------------------------


    # 判断这一步 decode 是否有足够 free block：
    # 如果当前的Sequence长度是block_size的整数倍+1，那么就需要一个新的block来存储这个新token，因此需要检查free_block_ids是否足够。这一步只是检查，没有实际分配block；
    # 如果是别的情况，说明当前的Sequence的最后一个token还没有填满当前的block，因此不需要新的block，直接写入即可。
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 执行体：根据 len(seq) % block_size 做对应处理。
    # 注意 decode 不查哈希表、不认领复用（本函数是缓存的生产者，非消费者）：
    #   新块属于本序列独有，不被 SequenceB 的 decode 复用——自回归续写几乎无法与前缀
    #   严格对齐，故 decode 阶段只把填满的块登记进 hash_to_block_id，供未来 prefill 消费。
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 跨块边界：从 free 池取一个新块追加到 block_table，承接下一个 token。
            # 该块此时半满，hash == -1，尚不可被复用。
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 当前块刚好填满：计算链式哈希并登记进 hash_to_block_id，
            # 使其从"半满不可复用"转为"完整可复用"的缓存块。
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 仍在填充最后一个半块：本步不分配、不登记，token 直接写入现有 last_block。
            assert last_block.hash == -1
