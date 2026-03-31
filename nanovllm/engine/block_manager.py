from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


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
    # 举例：
    #   序列A: [1,2,3,4 | 5,6,7,8]
    #   序列B: [1,2,3,4 | 5,6,7,8 | 9,10]
    #   序列C: [0,2,3,4 | 5,6,7,8 | 9,10]
    #   block_size=4
    # 对于序列A的第一块 hash0=hash([1,2,3,4])hash0=hash([1, 2, 3, 4])hash0=hash([1,2,3,4])，第二块hash1=hash([5,6,7,8])hash1=hash([5,6,7,8])hash1=hash([5,6,7,8])
    # 对于序列B的第一块 hash0=hash([1,2,3,4])hash0=hash([1, 2, 3, 4])hash0=hash([1,2,3,4])，第二块hash1=hash([5,6,7,8])hash1=hash([5,6,7,8])hash1=hash([5,6,7,8])，第三块hash2=([9,10])hash2=([9, 10])hash2=([9,10])，可以发现序列B的前两块内容和A完全相同，这个时候就可以复用A的KV cache，只需要计算和存储后面的内容
    # 对于序列C，从第一个token开始就和A和B对不上，就完全无法复用他们的KV cache，只能重新计算并存储
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # 确保该block的引用次数为0
        assert block.ref_count == 0
        block.reset()
        # 从空闲队列中移除并添加到已使用的block集合中
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    #负责为一个新的序列分配物理内存块，并在这个过程中尽可能地复用已有的缓存
    def allocate(self, seq: Sequence):
        assert not seq.block_table#确保该序列是一个全新的、尚未分配任何物理块的请求
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):#相当于把这个序列使用一个block滑窗去进行分析
            token_ids = seq.block(i)#当前block滑窗下框出的token_ids列表

            #计算这一堆token_ids的哈希值
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

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
                    # 【情况1：块正被占用】
                    # 别的序列（或同一个批处理里的其他请求）正在使用这个物理块。
                    # 我们只需要增加引用计数，表示“我也要加入共享这个块”。
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 【情况2：块在空闲池】
                    # 这个块以前算过，后来被释放了，现在躺在 free_block_ids 里。
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

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
