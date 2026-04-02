"""
Scheduler —— 请求调度器
=========================

整体职责：
----------
决定"这一个 step，哪些 Sequence 能上 GPU、上 GPU 跑 prefill 还是 decode、
谁要被抢占让出 KV cache 显存块"。它是整个 engine 里唯一知道"系统当前状态"
的人（谁在排队、谁在跑、谁缓存命中了多少）。

核心数据：
----------
- self.waiting: deque[Sequence]   —— 等待被首次调度跑 prefill 的请求队列。
- self.running: deque[Sequence]   —— 已经 prefill 完、正在逐 token decode 的请求队列。
- self.block_manager: BlockManager —— 物理 KV block 池，负责 allocate / free / prefix hit。

schedule() 返回什么：
--------------------
返回 (seqs, is_prefill)：
  - 如果 waiting 里有可以跑的请求 → 本 step 做 prefill，返回一批 waiting 里的 seq；
  - 否则从 running 里拿出能继续 decode 的 seq，返回一批 running 里的 seq。

  这是 vLLM v0 的"二选一"模型 —— 同一个 step **要么 prefill 要么 decode，不混批**。
  这也是为什么长 prefill 会把 decode 卡死（head-of-line blocking，见 docs/principles.md §2）。
  Phase 1 会把这里改成 **chunked prefill 混批**。

两个关键 budget：
-----------------
- max_num_seqs            —— 并发 seq 上限（控制 batch size）。
- max_num_batched_tokens  —— 单 step 处理的 token 总数上限（控制 compute 量）。

抢占（preempt）：
-----------------
decode 阶段某个 seq 要写 KV 但 block 池空了 → 把 running 队尾的倒霉蛋
踢回 waiting 并释放其 block，让当前 seq 继续跑。被抢占的 seq 下次从
prefill 状态重跑（Phase 3 会做 SLO-aware 的优先级抢占）。
"""
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs#决定了同时运行的最大序列数量，是控制并发的重要指标。
        self.max_num_batched_tokens = config.max_num_batched_tokens#限制了批处理中 tokens 的最大数量，避免资源过载。
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()#存放等待运行的Sequence对象
        self.running: deque[Sequence] = deque()#存放正在运行的Sequence对象

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # 先取出一个正在运行的序列，尝试给它安排本轮 decode。
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # can_append=False 代表当前 seq 若继续生成，所需 block 资源不足。
                # 这时需要“抢占”其他序列释放显存块，优先牺牲队尾（较晚被调度）的序列。
                if self.running:
                    # 抢占 running 队列尾部的序列的block资源：把它回退到 waiting，并释放其 block。
                    self.preempt(self.running.pop())
                else:
                    # 如果没有其他序列可抢占，只能抢占自己（当前 seq）：也就是放弃本轮调度。
                    self.preempt(seq)
                    break
            else:
                # 只有 while 没有被 break 时才会进入这里：
                # 说明当前 seq 已经满足 can_append，可以安全推进一步 decode。
                num_seqs += 1
                # may_append 会在“需要新块”的边界时分配新 block，或在块填满时计算/登记 hash。
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本轮成功调度的序列放回 running 队列头部，维持下一轮调度顺序。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    #用于抢占资源
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
