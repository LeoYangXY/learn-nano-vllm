import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    # *args: 用于接收任意数量的位置参数（没有参数名的），并将它们打包成一个tuple，以及后续使用的时候可以用*args来解包这个tuple传递给函数
    # **kwargs: 用于接收任意数量的关键字参数（key=value 形式的），并将它们打包成一个dict，以及后续使用的时候可以用**kwargs来解包这个dict传递给函数
    # 例：
    # def example(a, b=10, *args, **kwargs):
    #     print(f"a: {a}")
    #     print(f"b: {b}")
    #     print(f"args: {args}")
    #     print(f"kwargs: {kwargs}")
    #
    # 调用example(1, 2, 3, 4, x=5, y=6)
    #
    # 输出：
    # a: 1
    # b: 2
    # args: (3, 4)
    # kwargs: {'x': 5, 'y': 6}


    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()#这是特殊创建出来的跨进程通信的事件对象，可以在不同进程之间进行同步和通信。每个子进程都会有一个对应的事件对象，通过这个事件对象，父进程可以通知子进程执行某些操作，或者子进程可以通知父进程它们已经完成了某些任务。因此传入子进程的不是所谓的副本
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()# 调用schedule得到需要处理的序列以及是否为prefill。
        token_ids = self.model_runner.call("run", seqs, is_prefill)# 调用ModelRunner的run得到输出的token_ids。
        self.scheduler.postprocess(seqs, token_ids)# 使用postprocess把模型生成的token更新到对应的seqs。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]# 收集已经完成生成的序列并存入outputs中
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 计算生成的token数量，如果是prefill阶段就返回所有Sequence的长度之和，如果是decode阶段就返回负的当前处理的Sequence数量（因为decode阶段每个Sequence只生成一个token，所以直接返回数量即可）
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()#只要还有请求没完成，就不断调用 step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
