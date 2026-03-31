import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        #把device设置成cuda之后做相关的model load，warmup，kv cache分配等操作，确保这些操作都在GPU上进行，避免了不必要的CPU-GPU数据传输，提高效率。
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        #最后换回cpu
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 这个multiprocessing库的SharedMemory是让不同进程共享cpu的内存（注意和nvSharedMemory相区分）
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):#子进程循环的方法。子进程会在创建好SharedMemory后进入loop循环不断读取SharedMemory来获取主进程分发的任务，当收到的任务为exit时则结束循环
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # pickle 是 Python 的序列化工具，event 是 multiprocessing 的进程间同步工具，SharedMemory 是 multiprocessing 的进程间共享内存工具
    #    pickle负责把 Python 对象（比如 list、dict、tuple、类实例）转换成一串字节。
    #    之所以需要它，是因为 SharedMemory 本质上只是一块“裸内存/原始字节缓冲区”，
    #    它不认识 Python 对象，只认识 bytes。
    # 两个进程间交互的整套流程可以理解成：
    # Python 对象 -> pickle.dumps -> bytes -> SharedMemory -> bytes -> pickle.loads -> Python 对象
    def read_shm(self):
        # read_shm 是子进程的“收货方”。
        #    - 先 self.event.wait()，阻塞等待主进程通知。
        #    - 再从 self.shm.buf[0:4] 读出长度 n，知道 payload 有多长。
        #    - 然后从 self.shm.buf[4:n+4] 取出那段字节流。
        #    - 最后 pickle.loads(...) 把字节流还原成 Python 对象，拿到 method_name 和 args
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        # write_shm 是主进程的“发货方”。
        #    - 先执行 data = pickle.dumps([method_name, *args])，把方法名和参数打包成字节流。
        #    - 再执行 self.shm.buf[0:4] = n.to_bytes(4, "little")，写入 4 字节长度，
        #      这样接收方就知道后面要读多少字节。
        #    - 再执行 self.shm.buf[4:n+4] = data，把真正的字节内容写进共享内存。
        #    - 最后 event.set()，通知其他进程：“数据已经写好了，可以来读了”。
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 在python中，*这个修饰符可以用作实现编包和拆包的语法糖，极大地提升了函数参数传递的灵活性和可读性。
    def call(self, method_name, *args):
        # 入参这里的 *args 是“编包”：在函数定义里，*args 会把多出来的多个位置参数
        # 自动收集成一个 tuple。比如 call("foo", 1, 2) 时，args 就是 (1, 2)。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)#从 self 这个对象里，按字符串 method_name 去找同名属性或方法。比如：method_name 是 "exit"，那就等价于拿到 self.exit 这个方法
        # 下面 method(*args) 是“拆包”：把这个 tuple 再展开成独立参数传给目标函数。
        # 比如 args = (1, 2) 时，method(*args) 等价于 method(1, 2)。
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()

        # pytorch内存池行为：
        # PyTorch 不是每次都直接向 CUDA 申请/释放显存，它有 Caching Allocator（缓存池），避免了重复调用cudaMalloc的系统开销
        # 张量释放后，显存块通常先回到池子里，等后续复用，不会立刻还给驱动。
        # 这会导致：你代码里很多张量已经释放了，但 nvidia-smi 看到占用还很高。因为这些显存还在 PyTorch 池里，属于可复用但未归还状态。
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]#历史峰值显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]#当前显存

        num_kv_heads = hf_config.num_key_value_heads // self.world_size#做了tensor parallel，每个chip负责一部分head
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        #一个block的KV cache占用的显存大小 = 这一个block中可以存的token的数量 * 每个token占用的 KV cache大小
        #一个token占用的KV cache大小：因为一个token要流经整个model的每一层，因此会有num_hidden_layers个attention层，也就是要存num_hidden_layers份KV cache；
        #每份KV cache的大小又等于 num_kv_heads * head_dim（每个token在每一层的KV cache中占用的空间）* 2（因为要同时存K和V）* 每个向量的单个元素的字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 每个序列 seq 都有一个 block_table，这个代码是先找到所有Sequence里最长的那个 block_table 的长度 max_len，然后把每个 Sequence 的 block_table 都补齐到这个长度进行对齐
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens #seq.num_cached_tokens 之前的 tokens 被缓存过，因此从 num_cached_tokens 开始计算
            seqlen_k = seqlen

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)#把多个序列的进行flatten。这个前缀和数组表示每个序列在这个批次里的位置边界。比如：如果有三个序列，seqlen_q 分别是 5、7、6，那么 cu_seqlens_q 就是 [0, 5, 12, 18]，表示第一个序列占据位置 [0, 5)，第二个序列占据位置 [5, 12)，第三个序列占据位置 [12, 18)。
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            
            # 一个Sequence有一个block_table。一个Sequence会被从0开始编号的划分为多个logical block，每个logical block会被分配一个物理block（block_table里存的就是这个映射关系）。
            # 每个logical block又包含多个token（比如block_size=4），这些token在KV cache里占用连续的slot（比如slot 8,9,10,11）。当多个Sequence被放到一个batch里时，batch里的每个Sequence都会有一个slot_mapping，把这个Sequence里每个要写入KV cache的token映射到具体的slot上。
            # 
            # 例如：block_size = 4
            # sequence1.block_table = [2, 7]  => 物理 slot 分别是 [8, 9, 10, 11, 28, 29, 30, 31]
            # sequence2.block_table = [5]     => 物理 slot 分别是 [20, 21, 22, 23]
            # 如果把这两个 sequence 放到同一个 batch 里，这些 slot 会按 batch 顺序拼成一个大列表。这便是 slot_mapping。比如上面这个例子，slot_mapping 就是 [8, 9, 10, 11, 28, 29, 30, 31, 20, 21, 22, 23]。
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # 在prefill阶段就把这些准备好的输入数据（input_ids、positions、cu_seqlens_q、cu_seqlens_k、slot_mapping、block_tables）都一次性放到GPU上，并且调用 set_context 把这些数据放到全局上下文里，这样后续的模型计算就可以直接从上下文里拿到这些数据，避免了重复的数据传输和准备，提高效率。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:#都是把每个序列的最后一个token作为输入，positions是这个末尾token在这个序列里的位置，slot_mapping是这个末尾token在KV cache里对应的slot位置，context_lens是这个序列的长度
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)#获取当前batch大小，根据这个batch大小选择合适的CUDA Graph来回放：选择刚好大于等于这个batch大小的最小档位来回放
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()#在prefill以及每一次decode阶段结束后都调用reset_context把全局上下文重置掉
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # CUDA Graph 不是把“kernel的所有 shape”都录一遍，而是挑一组常见的 batch size 档位先录好。
        # 后面真正推理时，如果来了 bs 条请求，就找一个“不小于 bs 的最近档位”来回放，
        # 这样既避免了每次都重新走 Python eager 前向，也不用为每个可能的 bs 都单独录图。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # graph_vars 保存录图和回放时共用的固定张量。
        # 后面 decode 阶段不会重新创建这些 tensor，而是先把新数据写进这些占位张量，
        # 再直接 graph.replay()，这样就能复用已经录好的 CUDA Graph。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
