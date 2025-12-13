import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from model import GPTConfig, GPT
from model_tp_v2 import apply_tensor_parallel_to_gpt

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 8  # for TP 
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# TP settings
tensor_parallel = 1  # tp_size On single node, set as GPU count. On multi-nodes, world_size = dp_size * tensor_parallel
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
dtype = 'float32'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a distributed run (torchrun) ?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed

    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# count dp_size based on tensor_parallel. Used for multiple nodes
if tensor_parallel > 1 and ddp:
    assert ddp_world_size % tensor_parallel == 0, \
        f"world_size={ddp_world_size} must be divisible by tensor_parallel={tensor_parallel}"
    dp_size = ddp_world_size // tensor_parallel
else:
    dp_size = ddp_world_size

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"ddp_world_size={ddp_world_size}, dp_size={dp_size}, tp_size={tensor_parallel}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ---------------------- model init ----------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    base_model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    base_model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    base_model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    base_model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(base_model.config, k)

# crop down the model block size if desired
if block_size < base_model.config.block_size:
    base_model.crop_block_size(block_size)
    model_args['block_size'] = block_size

# put the model on current rank
base_model.to(device=device, dtype=ptdtype)
raw_model = base_model  # The base nanoGPT model, for estimate_mfu and others

# --------------- Tensor Parallel + FSDP / DDP ----------------
use_tp = (tensor_parallel > 1)

if use_tp:
    if not ddp:
        raise RuntimeError(
            "Tensor parallel > 1. Use torchrun to start multi-process（ddp_world_size = dp_size * tensor_parallel）"
        )
    print(f"[Rank {ddp_rank}] Building 2D device mesh: dp_size={dp_size}, tp_size={tensor_parallel}")
    mesh_2d = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, tensor_parallel),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

    print(f"[Rank {ddp_rank}] Applying tensor parallel on tp mesh")
    apply_tensor_parallel_to_gpt(raw_model, tp_mesh)

    print(f"[Rank {ddp_rank}] Wrapping with FSDP over dp dimension")
    model = FSDP(
        raw_model,
        device_mesh=dp_mesh,
    )

else:
    model = raw_model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

#-----

if use_tp:
    print(f"[Rank {ddp_rank}] Checking model after TP...")
    
    # 1. check the params
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[ERROR Rank {ddp_rank}] NaN/Inf in parameter: {name}")
            has_nan = True
    
    if not has_nan and ddp_rank == 0:
        print(f"✓ All parameters initialized correctly")
    
    # 2. test forward pass (have some issue on multiple nodes)
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, min(1000, model_args['vocab_size']), (2, 16), device=device)
        try:
            # use raw_model for test
            test_output = raw_model(test_input, test_input)
            
            if isinstance(test_output, tuple) and len(test_output) == 2:
                test_logits, test_loss = test_output
                
                if ddp_rank == 0:
                    print(f"✓ Test forward pass successful")
                    print(f"  Logits shape: {test_logits.shape}")
                    print(f"  Loss: {test_loss.item():.4f}")
                
                # check numeric
                if torch.isnan(test_loss) or torch.isinf(test_loss):
                    print(f"[ERROR Rank {ddp_rank}] Test loss is NaN/Inf!")
            else:
                if ddp_rank == 0:
                    print(f"[WARNING] Unexpected forward output format: {type(test_output)}")
                
        except Exception as e:
            print(f"[ERROR Rank {ddp_rank}] Test forward failed: {e}")
            if ddp_rank == 0:
                import traceback
                traceback.print_exc()
    
    model.train()
    
    # 3. barrier the ranks
    if ddp:
        dist.barrier()

# ---------------------- optimizer / scaler ----------------------
# GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer：TP+FSDP use model.parameters()
if use_tp:
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        **extra_args,
    )
    print(f"using fused AdamW (TP+FSDP): {use_fused}")
else:
    # optimizer of the base nanoGPT 
    optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# ---------------------- eval helper ----------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)

            if ddp and (ddp_rank == 0) and k == 0 and split == 'train':
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("[DEBUG] logits contain NaN/Inf in estimate_loss()", flush=True)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("[DEBUG] loss is NaN/Inf in estimate_loss()", flush=True)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# lr scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ---------------------- training loop ----------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # eval & ckpt
    if iter_num % eval_interval == 0:）
        losses = estimate_loss()

        # only master_process print log and save ckpt
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
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
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # 3. barrier all the ranks
        if ddp:
            dist.barrier()
    if iter_num == 0 and eval_only:
        break

    # train step
    for micro_step in range(gradient_accumulation_steps):

        with ctx:
            logits, loss = model(X, Y)
            
            # check numeric
            if iter_num == 0 and micro_step == 0 and ddp_rank == 0:
                print(f"[DEBUG] First forward pass:")
                print(f"  logits: min={logits.min():.4f}, max={logits.max():.4f}, "
                    f"mean={logits.mean():.4f}, std={logits.std():.4f}")
                print(f"  loss: {loss.item():.4f}")
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("[ERROR] logits contain NaN/Inf at iter 0!")
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("[ERROR] loss is NaN/Inf at iter 0!")
            
            loss = loss / gradient_accumulation_steps
        
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if ddp and ddp_rank == 0 and iter_num % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"[WARNING] NaN/Inf in gradients at iter {iter_num}")
            total_norm = total_norm ** 0.5
            if iter_num % 10 == 0:
                print(f"Gradient norm: {total_norm:.4f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # log
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        if ddp:
            dist.barrier()
        break

if ddp:
    dist.barrier()
    destroy_process_group()

sys.exit(0)
