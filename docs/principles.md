# nano-vllm 高阶推理技术原理与落地

> 目标：把工业级推理引擎（vLLM v0.6+、SGLang、TensorRT-LLM、Mooncake）真正难的点都讲清楚，
> 并把每一项如何在本仓库落地的路线图写出来。
> 每一节都按照：**(1) 要解决的问题 / (2) 原理 / (3) 工业实现细节 / (4) 在 nano-vllm 里的落地**  四段式来写。

## 目录
1. [Prefix Caching（Hash / RadixAttention）](#1-prefix-caching)
2. [Chunked Prefill（Sarathi-Serve）](#2-chunked-prefill)
3. [Disaggregated Prefill/Decode](#3-disaggregated-pd)
4. [Speculative Decoding（EAGLE / Medusa / n-gram）](#4-speculative-decoding)
5. [CUDA Graph](#5-cuda-graph)
6. [Multi-LoRA（Punica / SGMV）](#6-multi-lora)
7. [FP8 / INT4 Kernel（Marlin / Machete / FlashInfer）](#7-fp8--int4)
8. [复杂调度（Priority / Fairness / SLO）](#8-complex-scheduling)
9. [TP / PP 切分下的调度](#9-tp--pp-scheduling)
10. [附录：nano-vllm 与 vLLM 名词对照表](#10-appendix)

---

## 1. Prefix Caching

### 1.1 要解决的问题
LLM 推理的 prefill 阶段，对 `prompt` 里每个 token 都要跑一遍 Attention 的 K/V 投影，
**复杂度 O(L²)**（L 为 prompt 长度）。
但多轮对话、Few-shot、system prompt 等场景，**大量请求共享同一段前缀**：

```
req1: "<system>You are a helpful assistant</system><user>1+1=?</user>"
req2: "<system>You are a helpful assistant</system><user>中国首都?</user>"
```

前 10~1000 个 token 完全一样。如果每次都重新算一遍 K/V，**浪费 90% 的 prefill 算力**。

**Prefix Caching 的目标**：**同一段前缀的 KV 只算一次，后面请求直接复用缓存中的 KV，prefill 跳过**。

### 1.2 原理：两种主流实现

#### 方式 A：Hash-based Block Cache（vLLM v0.5+、nano-vllm）
- KV cache 按固定大小 `block_size`（常见 16/256）切块；
- 每个 block 计算 hash：`hash(block_token_ids, prev_block_hash)` → **链式哈希**；
- 只有整块且 token_ids 完全一致时才复用（粒度粗，但简单）；
- 用 `hash → block_id` 的 dict 做查找。

**优点**：实现简单、命中判定 O(1)、与 PagedAttention 的 block 结构天然对齐。
**缺点**：
- 粒度是 block（默认 16 token），**token 级 partial 匹配无法复用**；
- hash 冲突要防，需要存 `token_ids` 做二次校验（本仓库 `block_manager.py` 第 93 行就是在做这件事）；
- Dict 本质是扁平结构，**无法捕获"多请求共享 + 多分支分叉"的树状结构**。

#### 方式 B：RadixAttention（SGLang 主推）
- 把所有正在使用 / 已缓存的 prefix 组织成一棵 **Radix Trie**（压缩字典树）；
- Trie 每个节点存一段连续 token（可变长），子节点代表分叉的可能 continuation；
- 查询：在 trie 上走最长前缀匹配，返回命中长度 + 对应的 KV blocks；
- 驱逐：LRU + 引用计数，root-to-leaf 的叶子 LRU 先被驱逐。

**优点**：
- **Token 级别前缀匹配**（不受 block 边界限制）；
- 多请求共享的 system prompt / few-shot 自动聚拢到 trie 的公共前缀节点上；
- 多轮对话场景下每轮都能增量复用 → **SGLang 在多轮 benchmark 上碾压 vLLM 的本质**。

**缺点**：
- 实现复杂（分裂节点、合并、引用计数 GC）；
- 遍历 trie 有 O(prefix_len) 的 CPU 开销（不算大，但非零）；
- 与 PagedAttention 混用时需要细心对齐 block 边界。

### 1.3 工业实现细节（面试必问）

1. **防冲突**：xxhash / CityHash 比 Python `hash` 快且冲突率低；真要防，还要存 token_ids 做 memcmp。
2. **内存复用**：hit 的 block 如果已被驱逐回 free 池，需要**重新"认领"回来**（nano-vllm `_allocate_block` 分支 2）。
3. **引用计数 GC**：多个 req 共享同一 block，`ref_count` 归零才真正释放。
4. **Copy-on-Write**：两个 req 共享 block，某个 req 到 block 末尾要 decode 一个分叉 token 时，**必须 fork 一份新 block 出来**（vLLM 有实现，nano-vllm 没有，因为 decode 阶段每个 req 都独占末尾块）。
5. **KV 数据未必真写**：Prefix hit 意味着**根本不跑 prefill 的 attention**，K/V 压根没重新算过，直接走 cache 里的旧值。这也是为什么 `attention.py` 第 82 行 `k, v = k_cache, v_cache` —— hit 时 Q 来自新算的，但 K/V 全从 cache 拿（包括当前这一段）。

### 1.4 nano-vllm 里的落地
- **已有**：`block_manager.py` 的 hash-based 版本，`compute_hash` 做链式哈希，`hash_to_block_id` dict，支持重新认领 + 冲突兜底校验。
- **Phase 5 要做**：把 `hash_to_block_id` dict 替换成真正的 `RadixTrie`，支持 token 级 partial prefix 匹配、LRU 驱逐、引用计数。
- 代码入口：`nanovllm/engine/radix_cache.py`（Phase 5 新建），Scheduler 的 `block_manager` 换成 `RadixBlockManager`。

---

## 2. Chunked Prefill

### 2.1 要解决的问题
Prefill 和 decode 的计算特征完全不同：

| 阶段 | 计算量 | 显存访问 | Kernel 特征 |
|------|--------|----------|-------------|
| Prefill | 极大（O(L²) attention，O(L·d²) linear） | 较少 | **Compute-bound** |
| Decode（每步） | 极小（单 token） | 全量读 KV cache | **Memory-bound** |

**问题 1**：一个 8K token 的 prefill 请求进来，batch 里所有 decode 请求的 TTFT（Time To First Token）被拖到几秒甚至十几秒——**head-of-line blocking**。
**问题 2**：如果串行跑，prefill 阶段 GPU 算力吃不满 batch（bs=1 或 2），decode 阶段带宽吃不满（单 token 传 GB 级 KV）——**两个阶段各自的硬件特性都没利用到**。

### 2.2 原理：Sarathi / vLLM v0.6+ 方案
**核心思想**：
1. 把长 prefill **切成 chunk**（常见 512~2048 token/chunk）；
2. Scheduler 每个 step **混批**：`1 个 prefill chunk + N 个 decode token` 一起塞给 model；
3. Attention kernel 支持不同 req 的 query 长度不同（`varlen`）；
4. **Budget 约束**：`total_tokens_this_step ≤ max_num_batched_tokens`（典型值 2048）。

**收益**：
- Prefill 的 compute-bound 和 decode 的 memory-bound **天然互补**，硬件利用率从 40% → 80%；
- decode 请求的 TTFT / ITL（Inter-Token Latency）**不会被长 prefill 卡死**；
- 在 SLO-aware 场景下非常关键。

### 2.3 工业实现细节

1. **Attention kernel**：必须支持 varlen，query 长度可以是 1（decode）或 N（prefill chunk）。FlashAttention-2 的 `flash_attn_varlen_func` 天然支持。
2. **Slot mapping**：prefill chunk 的每个 token 都要写入自己对应的 KV slot；decode 只写 1 个 slot。混批时拼成一个大的 `slot_mapping`。
3. **Prefix cache 集成**：chunk prefill 要和 prefix cache 协同——chunk 内部命中的 block 不跑计算，只读；chunk 末尾的不完整 block 走正常 prefill。
4. **Chunk 大小权衡**：
   - 大 chunk → prefill 效率高，但拖慢 decode；
   - 小 chunk → decode 响应快，但 prefill 总开销增加（每个 chunk 都有 kernel launch 开销 + 读前面所有 KV 的开销）；
   - 经验值：2048 token / step，或按算力动态调整。
5. **调度器的难点**：每个 step 既要拿 decode 请求（budget 很松），也要拿 prefill chunk（budget 很紧），还要避免饿死。vLLM v0.6 的 scheduler 就是为此重写的。

### 2.4 nano-vllm 里的落地
- **Phase 1**：
  - `config.py` 加 `max_chunk_size`（默认 2048）；
  - `sequence.py` 加 `num_prompt_processed` 字段，记录 prompt 里已经 prefill 了多少 token；
  - `scheduler.py` 的 `schedule()` 改成**混批**：先拿 decode，剩余 budget 塞 prefill chunk；
  - `model_runner.py` 的 `prepare_prefill` 改成处理"部分 prompt"；新加 `prepare_mixed` 同时处理两类；
  - `attention.py` 不用动（`flash_attn_varlen_func` 已经支持 varlen）。

---

## 3. Disaggregated P/D

### 3.1 要解决的问题
即便有了 Chunked Prefill，**prefill 和 decode 仍然在同一个 GPU 上抢资源**。但它们的硬件需求差异巨大：

- Prefill：需要**大算力**（FLOPS），对显存带宽要求一般；
- Decode：需要**大显存 + 大带宽**（读 KV cache），对算力要求一般。

把它们堆在一起等于：**decode 请求不断打断 prefill 的 compute-bound kernel，导致算力利用率受损；prefill 的显存突发申请又把 decode 的 batch size 拉低**。

### 3.2 原理：DeepSeek / Mooncake / 月之暗面方案

**核心思想**：**物理分离**。
- **Prefill 集群**：专门跑 prefill，算完 KV 就把结果传给 Decode 集群；
- **Decode 集群**：只跑 decode，从 Prefill 集群接收 KV，持续生成 token；
- 中间通过 **RDMA / NVLink / NCCL send-recv** 传 KV（Mooncake 用 RDMA + 分层存储，还能把 KV 落到 DRAM / SSD）。

**收益**：
- Prefill 集群可以用**计算卡**（H100 SXM 压榨算力）；
- Decode 集群可以用**显存卡**（H200 / MI300X，更多 HBM）；
- 各自 batch size、调度策略、SLO 目标都独立；
- **长 prompt 场景收益最大**（比如代码生成、RAG、长文本总结）。

### 3.3 工业实现细节

1. **KV 传输**：
   - 量级很大：`2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_size`。
   - 比如 LLaMA-70B，32 层，8 KV heads，128 dim，2K prompt，BF16 → **2 GB/请求**。
   - RDMA 能打到 200+ GB/s，NVLink 能打到 900 GB/s，用对硬件才不会成为瓶颈。
2. **传输与计算 overlap**：
   - Mooncake 核心：**layer-wise 传输**——第 1 层算完就开始传，同时算第 2 层，把传输藏进计算；
   - 需要自定义 NCCL 通信流 + CUDA event 同步。
3. **KV 布局兼容**：
   - 两端的 TP / PP / head 切分必须匹配，否则要做 all-to-all 重排；
   - Paged KV 要按 block 传，对方收到后直接塞进自己的 block pool。
4. **全局调度器**：
   - 统一接收请求 → 路由到 prefill 集群 → 等 KV 就绪 → 路由到 decode 集群；
   - 需要考虑负载均衡（prefill 集群满了怎么办？decode 集群满了怎么办？）。
5. **容错**：一个 prefill 节点挂了，如果 KV 已经传到 decode 节点，decode 可以继续；如果没传完，要重跑 prefill。

### 3.4 nano-vllm 里的落地
- **Phase 8**（工作量最大，会改动整体架构）：
  - 新建 `nanovllm/engine/disagg/` 目录；
  - `prefill_engine.py`：魔改版 LLMEngine，只跑 prefill，返回 KV tensor；
  - `decode_engine.py`：接收 KV tensor，塞进 block pool，从 decode 阶段开始；
  - `kv_transfer.py`：本地用 NCCL send/recv（单机多卡模拟），跨机用 RDMA（选做）；
  - 顶层 `DisaggLLM` 封装两个 engine；
  - 只做**单机两卡模拟**：卡 0 跑 prefill，卡 1 跑 decode，KV 通过 NCCL P2P 传递。

---

## 4. Speculative Decoding

### 4.1 要解决的问题
Decode 阶段每一步只生成 1 个 token，**内存带宽瓶颈明显**（读全量 KV cache + 权重只为生成 1 个 token）。GPU 算力完全浪费（MFU < 5%）。

### 4.2 原理：草稿 + 验证

**核心**：用一个**便宜的模型**（draft）一次性生成 N 个 token，然后用**目标模型**（target）**一次 forward 并行验证**这 N 个 token，接受到第一个"不一致"为止。

**关键观察**：target 模型的一次 forward，**算 N 个 token 和算 1 个 token 耗时差不多**（都是 memory-bound），所以"1 次验证 = N 次 decode"。

#### 主流草稿方案对比

| 方案 | draft 来源 | 优点 | 缺点 | 典型加速 |
|------|-----------|------|------|---------|
| **Draft model**（SpecDec 论文） | 另一个小模型（7B 配 0.5B） | 通用、实现简单 | 需要额外模型、接受率一般 | 1.5-2x |
| **n-gram / prompt-lookup** | 从 prompt / 历史生成 token 里做 n-gram 匹配 | 零额外模型、代码生成场景极强 | 通用对话场景接受率很低 | 代码 2-3x，对话 1.1x |
| **Medusa** | 在 target 模型顶层训练多个 "头"（每个头预测第 k 个未来 token） | 不需要单独模型、推理流水清爽 | 需要训练、tree attention 实现复杂 | 2-3x |
| **EAGLE / EAGLE-2** | 用 target 模型的**倒数第二层隐藏状态**训练一个小 autoregressive head | 接受率最高（>80%）、业界主流 | 训练复杂、tree verify 实现复杂 | 3x+ |

### 4.3 工业实现细节

1. **Rejection Sampling**：
   - draft 给出每个 token 的概率 `q_i`，target 给出 `p_i`；
   - 以概率 `min(1, p_i/q_i)` 接受 token i；
   - 第一个被拒绝的位置，从修正分布 `norm(max(0, p - q))` 采样 1 个 token；
   - 这套数学保证**输出分布与直接从 target 采样完全一致**。
2. **Tree Attention**：
   - 不再只发一条线，而是发一棵 token 树（比如"the"后面分叉成 "cat", "dog", "a"）；
   - 一次 forward 并行验证整棵树，大幅提升接受 token 数；
   - 需要自定义 attention mask，把 tree 结构转成稀疏 mask。
3. **KV Cache 管理**：
   - 被拒绝的 token 对应的 KV 必须**回滚**（rollback / truncate）；
   - Paged KV 下就是把末尾几个 slot 的 K/V 丢弃，block pool 不动；
   - 如果做错了 KV 回滚，下一步 decode 就会用错 context → 静默错误，非常难 debug。
4. **与 CUDA Graph 的冲突**：
   - 接受的 token 数是**动态的**（0~N），CUDA Graph 要求静态 shape；
   - 解决方案：按接受长度录多份 Graph / 用 padding 补齐后 mask。

### 4.4 nano-vllm 里的落地
- **Phase 2a**：n-gram / prompt-lookup（纯 CPU 侧匹配，零额外模型依赖）：
  - 在 `sequence.py` 或新建 `nanovllm/engine/ngram.py` 做 n-gram 匹配；
  - `model_runner.run_model` 接受 `draft_tokens`，一次性 forward 验证；
  - 采样层加 `greedy_verify`（先用贪心版本，保证正确性）；
- **Phase 2b**：小 draft 模型版（可以复用 Qwen3-0.6B 自己给自己做 draft 示例，用一个更小的配置或截断前几层模拟 "小模型"）；
- EAGLE/Medusa 需要预训练 head 权重，nano-vllm 里只展示**架构接口**（加 `speculator` 抽象），不做完整训练。

---

## 5. CUDA Graph

### 5.1 要解决的问题
Decode 阶段每一步计算量很小，但**Python 层 + PyTorch dispatch + CUDA kernel launch** 的开销是固定的（每个 kernel 几 μs）。一次 decode 涉及几十上百个 kernel，**累计开销可能占整个 decode 时间的 30-50%**。

### 5.2 原理
CUDA Graph 把"一次完整的 forward"录制成一张**静态的 kernel DAG**，之后 replay 就是**一次 driver 调用**，把所有 kernel 批量提交给 GPU。开销从几百 μs 降到几 μs。

**约束**：
- 所有输入 tensor 的**地址、shape、dtype 都必须固定**；
- 所有算子都必须是 CUDA kernel（不能有 CPU→GPU 同步、不能有 Python 控制流）；
- kernel 里不能有动态 shape。

### 5.3 工业实现细节

1. **多档 batch size**：实际 bs 是动态的，**提前录一批常见 bs 的 graph**（比如 [1,2,4,8,16,32,...]），replay 时选"≥ bs 的最小档"。
2. **占位 buffer**：提前分配一组 `max_bs` 的 tensor，replay 前把数据 copy 进去（nano-vllm 的 `graph_vars` 就是这个）。
3. **Prefill 不录图**：prefill 的 token 数千变万化，录图成本过大，直接 eager 跑。
4. **KV cache 地址**：KV cache 是 graph 捕获时就绑定的 tensor，运行时不能重分配。这也是为什么 nano-vllm 要先 `allocate_kv_cache` 再 `capture_cudagraph`。
5. **多 stream / 多 pool**：多份 graph 共享同一个 memory pool 以节省显存（nano-vllm `self.graph_pool`）。

### 5.4 nano-vllm 里的落地
- **已有**：`model_runner.capture_cudagraph()` 录制 `[1, 2, 4, 8, 16, 32, ..., 512]` 的 decode graph；`run_model` 里 replay。
- **可增强**：
  - 加 metrics：统计每个档位的 replay 次数 + 耗时；
  - 为 spec decode 的 `verify` 单独录图（验证长度 N 固定时可行）。

---

## 6. Multi-LoRA

### 6.1 要解决的问题
SaaS 场景下，一个 base model（比如 Qwen-7B）要同时服务 100+ 个客户，每个客户有自己微调好的 LoRA adapter（几十 MB），但 base 共享。
**不能为每个 LoRA 都跑一个独立进程**（显存 × 100 炸掉），**不能 merge 进 base**（merge 了就没法多租户）。

### 6.2 原理
LoRA：`y = Wx + (BA)x`，其中 `W` 是 base 权重（冻结，shape `[out, in]`），`A/B` 是低秩矩阵（`A: [r, in]`, `B: [out, r]`，r 通常 8/16/64）。

**多 LoRA 服务**：batch 里每个请求可能用**不同的 adapter**。
- `y = Wx` → 所有请求共享，用普通 GEMM；
- `(BA)x` → 每个请求用自己的 `A_k, B_k`；

**naive 做法**：逐请求循环 `for k in batch: y[k] += B[k] @ A[k] @ x[k]` → batch 退化成串行，**完全没有 batch gain**。

### 6.3 工业实现：SGMV / Punica kernel

**SGMV (Segmented Gather Matrix-Vector Multiplication)**：
- 把同 adapter 的 request 聚在一起成**段（segment）**；
- 一个 GEMM kernel 内部对不同段用不同的权重指针；
- 等价于 `for seg: y[seg] = B[seg_id] @ A[seg_id] @ x[seg]`，但在 GPU 上并行。

**实现要点**：
1. 调度器按 adapter 把 batch 排序，生成 `segment_starts, segment_ids`；
2. Kernel 读 adapter 权重数组 `A_all[adapter_id], B_all[adapter_id]`；
3. 每个 CTA 处理一个 segment 的一部分；
4. 进阶：BGMV（Batched GMV，每个 request 一段），Punica paper 的做法。

### 6.4 nano-vllm 里的落地
- **Phase 4**：
  - 新建 `nanovllm/adapters/lora.py`：`LoRAAdapter` + `LoRAManager`（加载 / 卸载 / 查找）；
  - `Sequence` 加 `lora_id` 字段；
  - `Scheduler` 按 `lora_id` 排序 batch；
  - `linear.py` 加 `LoRALinear`：在 base `F.linear` 后面加 LoRA 增量；
  - 先实现**朴素版**（bmm 按 adapter 分组），再选做 Triton SGMV kernel；
  - 提供 demo：加载 2 个 LoRA，同一 batch 跑出不同风格输出。

---

## 7. FP8 / INT4

### 7.1 要解决的问题
- 权重占显存大（7B 模型 BF16 = 14 GB）；
- KV cache 也占显存（2K 序列能占几 GB）；
- Decode 阶段是 **memory-bound**，直接瓶颈在 HBM 带宽。

**量化把权重 / KV cache 用更少比特表示**，直接把显存 + 带宽需求降一半甚至 1/4。

### 7.2 原理

| 格式 | 权重 | activation | 硬件支持 |
|------|------|-----------|---------|
| BF16 baseline | 16 bit | 16 bit | 所有 Tensor Core |
| **FP8 (E4M3/E5M2)** | 8 bit | 8 bit | Hopper+、Blackwell、Ada Lovelace |
| **INT8** | 8 bit | 8 bit（SmoothQuant） | 所有 Tensor Core |
| **INT4 (Marlin)** | 4 bit | 16 bit | Ampere+ |
| **NF4 (GPTQ)** | 4 bit | 16 bit | 所有 Tensor Core |

**关键 kernel**：
- **Marlin**：INT4 weight × BF16 activation → BF16 output，A100 上能跑满带宽；
- **Machete**：Marlin 的 Hopper 升级版，支持 FP8；
- **FlashInfer**：同时支持多种量化 + attention 融合；
- **cutlass + CUTe**：自己写 FP8 GEMM 的底座。

### 7.3 工业实现细节

1. **Calibration**：做 INT8/INT4 前要跑一批校准数据，量化 activation 分布。
2. **Per-channel vs per-tensor**：按输出通道分别算 scale，精度损失小得多。
3. **Group-wise quantization**：把权重分组（每 128 个一组），每组一个 scale，是 GPTQ / AWQ 的标配。
4. **KV cache quantization**：K/V 单独量化成 FP8 / INT8，对输出分布影响比权重量化更大，但显存 / 带宽收益更明显。
5. **SmoothQuant / AWQ**：量化前把"难量化"的 activation outlier 转嫁到权重上，提升量化精度。

### 7.4 nano-vllm 里的落地
- **Phase 7**：
  - 新建 `nanovllm/layers/quantization/`；
  - `fp8.py`：FP8 线性层（RTX 5050 Blackwell 原生支持 FP8 E4M3 的 `torch._scaled_mm`）；
  - `int4_marlin.py`：INT4 linear（接 `vllm` 的 Marlin kernel 或自己写 Triton 版）；
  - 加 `config.quantization = "fp8" | "int4" | None`；
  - `loader.py` 根据 config 选权重转换路径；
  - 只做**权重量化**（weight-only），KV cache 量化可作为扩展。

---

## 8. Complex Scheduling

### 8.1 要解决的问题
10 个请求同时到来，延迟要求、优先级、SLO 都不一样：
- 付费用户 vs 免费用户（priority）；
- 实时对话（低 TTFT） vs 批量推理（高吞吐）；
- 有 deadline 的请求（比如 API 超时 60s）；
- 不同租户的公平性（一个 tenant 不能把 GPU 吃独食）。

**简单 FCFS 完全扛不住**。

### 8.2 原理：主流调度策略

| 策略 | 目标 | 实现 |
|------|------|------|
| **FCFS** | 简单 | 队列 |
| **Priority** | 付费用户优先 | 堆（按 priority 排） |
| **SJF (Shortest Job First)** | 最小化平均延迟 | 按 `max_tokens` 排 |
| **EDF (Earliest Deadline First)** | 命中 SLO | 按 deadline 排 |
| **Fair-share / DRF** | 多租户公平 | 每个 tenant 记 `used_tokens`，round-robin |
| **LSF (Least Slack First)** | SLO-aware | 按 `(deadline - ETA)` 排，slack 小的优先 |

**还有抢占策略**：
- 已经在 running 的请求，遇到高优先级请求来时**要不要抢**？
- 抢走的 request 要不要 swap KV 到 CPU 内存（recover 时省得重算）？

### 8.3 工业实现

1. **vLLM v0.6 scheduler**：基于 budget + priority + SLA constraints，代码 2K 行。
2. **SGLang scheduler**：按"谁能最快 finish" 调度，配合 radix cache 的"公共前缀请求聚簇"。
3. **TGI**：用 continuous batching + max-latency 保证。

### 8.4 nano-vllm 里的落地
- **Phase 3**：
  - `Sequence` 加 `priority`、`arrival_time`、`deadline`、`tenant_id`；
  - `Scheduler.waiting` 改 `heapq`；
  - 实现两种 policy：`priority` 和 `slo`（Least-Slack-First）；
  - 抢占策略：高优先级抢低优先级；
  - demo：同时发 10 个不同 priority 的请求，观察完成顺序。

---

## 9. TP / PP Scheduling

### 9.1 要解决的问题
单卡放不下模型（100B 以上模型必须切），就要用 Tensor Parallel 和 Pipeline Parallel。
但切完后，**调度器能做的决策都变了**：

- **TP**：所有卡同步跑同一个 batch（SIMD），调度没变化，**主要挑战是通信开销**（all-reduce 融入 attention/FFN）；
- **PP**：卡 0 算前 N 层，卡 1 算后 N 层，**每个 step 的 pipeline 有气泡**。

### 9.2 原理：Pipeline Parallel

假设模型分 4 stage，每个 stage 一张 GPU：

```
                 step0    step1    step2    step3    step4    step5
GPU0 (stage 0): [mb0] -> [mb1] -> [mb2] -> [mb3] -> [mb4] ...
GPU1 (stage 1):          [mb0] -> [mb1] -> [mb2] -> [mb3] -> [mb4] ...
GPU2 (stage 2):                   [mb0] -> [mb1] -> [mb2] -> [mb3] ...
GPU3 (stage 3):                            [mb0] -> [mb1] -> [mb2] ...
```

**气泡**：前 `num_stages - 1` 个 step，后面的卡在等——**利用率只有 `1 - (P-1)/(P+M-1)`**，M 是 micro-batch 数。

**解决方案**：把一个 batch 切成多个 micro-batch，pipeline 起来跑，减小气泡占比。
训练用 1F1B（一前向一后向），推理场景**连续地 forward micro-batch**。

### 9.3 工业实现

1. **Model 切分**：按 layer 切，同一个 layer 要在同一个 stage（切 layer 内会爆通信）。
2. **KV cache 分布**：每个 stage 存自己那几层的 KV，**请求的 KV 分布在多个 GPU 上**。
3. **调度器**：要把"这个请求第 3 层算完了，该送到 stage 1"这个状态记下来，scheduler 变成流水线工单系统。
4. **TP+PP 组合**：3D parallel（TP × PP × DP），DeepSeek / Llama 70B 常见配置 TP=8, PP=2, DP=1。
5. **通信**：stage 间传 hidden states（B × L × H），不是 KV。用 NCCL send/recv 或 P2P。

### 9.4 nano-vllm 里的落地
- **Phase 6**：
  - `config.py` 加 `pipeline_parallel_size`；
  - 模型按 layer 切成 stage，每个 stage 一个进程；
  - 同一个 request 的 forward 走 stage 0 → 1 → ... → 最后一 stage；
  - 只做 2-stage 的演示（单机 2 卡）；
  - Scheduler 加 `pipeline_step()` 逻辑，micro-batch 轮转。

---

## 10. Appendix

### nano-vllm ↔ vLLM 名词对照

| 概念 | nano-vllm | vLLM |
|------|----------|------|
| 请求 | `Sequence` | `Sequence` / `SequenceGroup` |
| KV cache 块 | `Block` (`block_size=256`) | `PhysicalBlock` (`block_size=16`) |
| KV cache 管理 | `BlockManager` | `BlockManager` + `BlockAllocator` |
| 调度器 | `Scheduler` | `Scheduler` |
| 模型执行 | `ModelRunner` | `Worker` + `ModelRunner` |
| KV tensor | `kv_cache[2, L, N_blk, blk_sz, heads, dim]` | `kv_cache[L][2]` 列表 |
| prefix cache | hash dict | hash-based / prefix-cached |
| CUDA Graph | `capture_cudagraph` | `CUDAGraphRunner` |
| TP | `tensor_parallel_size` + NCCL + SharedMemory | Ray / multiprocessing |

### 面试高频追问清单

按这个清单逐条能答上来，工业级 LLM 推理的深度就够了：

1. Prefix caching 的 hash 冲突怎么处理？为什么 nano-vllm 会存一份 `token_ids` 做二次校验？
2. RadixAttention 相比 hash-based 的优势场景是什么？为什么 SGLang 在多轮对话上更快？
3. Chunked prefill 的 chunk size 怎么选？大了小了各有什么问题？
4. Disaggregated P/D 里 KV 传输怎么和计算 overlap？
5. 投机采样为什么输出分布能保证一致？证明一下 rejection sampling 的数学。
6. EAGLE 和 Medusa 的区别？为什么 EAGLE 接受率更高？
7. CUDA Graph 为什么不录 prefill？录了会怎样？
8. SGMV / Punica 为什么能把 multi-LoRA 做到接近原生 batch 吞吐？
9. FP8 E4M3 和 E5M2 分别用在哪？为什么权重用 E4M3、梯度用 E5M2？
10. Marlin kernel 为什么比朴素的 INT4 dequant + GEMM 快很多？
11. SLO-aware 调度里，"slack" 怎么算？为什么不是简单按 deadline 排？
12. PP 的气泡怎么降？推理场景有哪些训练用不上的优化？
13. TP+PP 混用时，通信开销怎么估？在什么场景下 TP 比 PP 划算？
14. 抢占式 preemption 里，被抢的请求的 KV 要 swap 到哪？为什么不是直接丢掉？

---

> 文档持续更新中。每实现完一个 Phase，会把对应章节的"落地"部分补充成"已实现：代码入口 + 使用示例 + benchmark"。
