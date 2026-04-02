## TODO:

engine部分还需要仔细看


attention算子部分的实现

## Installation

```bash
conda create -n nano-vllm python=3.12 -y
conda activate nano-vllm
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128 #只要驱动的版本能用，那么这个pytorch就能跑。我们这个版本是下载了一个自带cu12.8版本toolkit的pytorch
pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl #下载对应的flash-attn（看准pytorch的版本是2.9以及他对应的toolkit是cu12，然后那个cp312指的是cpython为3.12）
pip install tqdm transformers xxhash

python ./example.py
```

## Model Download

To download the model weights manually, use the following command:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-0.6B', local_dir='./models/huggingface/Qwen3-0.6B/', local_dir_use_symlinks=False, repo_type='model')"
```
