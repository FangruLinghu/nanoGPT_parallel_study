"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
from pathlib import Path
import pynvml

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import deepspeed
from model import GPTConfig, GPT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import ParallelRunLogger
from comm_monitor import monitor
import subprocess

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
block_size = 256 # context of up to 256 previous characters
gradient_accumulation_steps = 4 # 1 gpu: 1 # 1 node: 4 # 2 nodes: 8
batch_size = 16 # 1 gpu: 64 # 1 node: 16 # 2 nodes: 8

# model
# baby GPT model :)
n_layer = 48
n_head = 32 # to make it balance with tp
n_embd = 2048
dropout = 0.2

learning_rate = 1e-4 # with baby networks can afford to go a bit higher # 1e-3 initially
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 #
beta1 = 0.9

bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer

weight_decay = 1e-1

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100

model_parallel = 4
tensor_parallel = 4

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# config_path = Path(__file__).resolve().parent.parent / "configurator.py"
# exec(open(config_path).read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# >>> LOGGER: init NVML
try:
    pynvml.nvmlInit()
    _NVML_DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _NVML_DEVICE_COUNT = 0
# <<< LOGGER

ddp = int(os.environ.get('RANK', -1)) != -1  # are we in a distributed launch?
if ddp:
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = ddp_rank == 0  # logging / checkpointing
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
#model.to(device)

raw_model_for_logger = model # for logger

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# --------------------------------------
# DeepSpeed init
# --------------------------------------
# initialize DeepSpeed engine
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    config="ds_config.json",
)

# after DeepSpeed init, model is now a DeepSpeed engine
device = model.device
checkpoint = None 

# --------------------------------------
# logger init（depend on raw_model_for_logger）
# --------------------------------------
parallel_mode = "zero1"  # single / dp / tp / pp / zero1 / zero2 / zero3
run_id = f"{parallel_mode}_nlayer{n_layer}_emb{n_embd}_g{ddp_world_size}"
param_dtype = next(raw_model_for_logger.parameters()).dtype
param_dtype_str = str(param_dtype).replace("torch.", "")

logger = ParallelRunLogger(
    raw_model=raw_model_for_logger, 
    dtype_str=param_dtype_str,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    parallel_mode=parallel_mode,
    n_gpu=ddp_world_size,
    micro_batch_size_per_gpu=batch_size,
    grad_accum_steps=1,
    run_id=run_id,
    results_path="/pscratch/sd/e/es_lh/nanoGPT/results.csv",
    master_process=master_process,
    notes=f"{dataset}, ds_{parallel_mode}",
)

# --------------------------------------
# helpers
# --------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# >>> LOGGER: GPU util helper
def get_gpu_util():
    try:
        if _NVML_DEVICE_COUNT == 0:
            return -1.0
        utils = []
        for i in range(_NVML_DEVICE_COUNT):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils.append(util.gpu)
        if not utils:
            return -1.0
        return sum(utils) / len(utils)
    except Exception:
        return -1.0
# <<< LOGGER

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

# --------------------------------------
# training loop
# --------------------------------------
wall_clock_start = time.time()

X, Y = get_batch('train')  # first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if hasattr(model, "module") else model
running_mfu = -1.0

# stats init
tokens_per_iter_theoretical = batch_size * block_size * ddp_world_size 
total_time = 0.0
measured_iters = 0
warmup_iters = 10  
gpu_util_sum = 0.0
gpu_util_count = 0

comm_time_sum = 0.0
bandwidth_sum = 0.0
collectives_sum = 0.0
comm_measured_iters = 0

if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats()

while True:

    monitor.reset()   # reset communication tracking per iteration

    # LR
    lr = get_lr(iter_num) if decay_lr else learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # eval
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward + backward + step
    with ctx:
        logits, loss = model(X, Y)  # model is DeepSpeed engine
    model.backward(loss)
    model.step()
    X, Y = get_batch('train')

    # timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num >= warmup_iters:
        total_time += dt
        measured_iters += 1

    # communication stats from monitor
    comm_time = monitor.comm_time
    comm_ratio = comm_time / dt if dt > 0 else 0.0
    bandwidth = monitor.bandwidth
    collectives = monitor.collectives

    # GPU util
    gpu_util = get_gpu_util()
    if gpu_util >= 0:
        gpu_util_sum += gpu_util
        gpu_util_count += 1

    # comm
    if iter_num >= warmup_iters:
        comm_time_sum += comm_time
        bandwidth_sum += bandwidth
        collectives_sum += collectives
        comm_measured_iters += 1

    # tokens
    tokens_this_iter = tokens_per_iter_theoretical
    total_tokens = tokens_this_iter * max(measured_iters, 1)
    tokens_per_sec = tokens_this_iter / dt if dt > 0 else 0.0

    # logging per iter
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        print(
            f"iter {iter_num}: loss {lossf:.4f}, "
            f"time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%, "
            f"comm {comm_time:.4f}s ({comm_ratio*100:.1f}%), "
            f"BW {bandwidth:.2f} GB/s, "
            f"collectives {collectives}, "
            f"tokens/s {tokens_per_sec:.1f}, "
            f"gpu_util {gpu_util:.1f}%"
        )

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        if ddp:
            torch.distributed.barrier()
        break

if ddp:
    torch.distributed.barrier()

wall_clock_end = time.time() 
total_runtime = wall_clock_end - wall_clock_start

# --------------------------------------
# end training, write logger
# --------------------------------------
if measured_iters > 0:
    avg_time = total_time / measured_iters
    avg_tokens_per_sec = tokens_per_iter_theoretical / avg_time
else:
    avg_time = None
    avg_tokens_per_sec = None

if device_type == "cuda":
    peak_mem_bytes = torch.cuda.max_memory_allocated()
else:
    peak_mem_bytes = None

avg_gpu_util = gpu_util_sum / gpu_util_count if gpu_util_count > 0 else None

final_train_loss = None
final_val_loss = None

if 'losses' in locals():
    try:
        final_train_loss = float(losses['train'])
        final_val_loss = float(losses['val'])
    except Exception:
        pass

if comm_measured_iters > 0:
    avg_comm_time = comm_time_sum / comm_measured_iters
    avg_bandwidth = bandwidth_sum / comm_measured_iters
    avg_collectives = collectives_sum / comm_measured_iters
    avg_comm_ratio = avg_comm_time / avg_time if avg_time else None
else:
    avg_comm_time = None
    avg_bandwidth = None
    avg_collectives = None
    avg_comm_ratio = None

logger.log_summary(
    avg_time_per_iter=avg_time,
    tokens_per_sec=avg_tokens_per_sec,
    peak_mem_rank0_bytes=peak_mem_bytes,
    final_train_loss=final_train_loss,
    final_val_loss=final_val_loss,
    mfu=running_mfu if running_mfu >= 0 else None,
    avg_gpu_util_pct=avg_gpu_util,
    extra_fields={
        "dataset": dataset,
        "total_runtime_sec": total_runtime,
        "avg_comm_time": avg_comm_time, 
        "avg_comm_ratio": avg_comm_ratio, 
        "avg_bandwidth_GBs": avg_bandwidth,
        "avg_collectives": avg_collectives,    
    },
)

try:
    pynvml.nvmlShutdown()
except Exception:
    pass

sys.exit(0)
