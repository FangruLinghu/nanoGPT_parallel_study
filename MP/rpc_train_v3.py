# train_rpc_v3.py
import os
import time
import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed.rpc as rpc

from rpc_mp import GPTConfig, RPCPipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_addr", type=str, default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    ap.add_argument("--master_port", type=str, default=os.environ.get("MASTER_PORT", "29500"))
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    ap.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", "8")))

    ap.add_argument("--out_dir", type=str, default="out-shakespeare-char")
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--always_save_checkpoint", action="store_true", default=False)

    ap.add_argument("--dataset", type=str, default="shakespeare_char")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--learning_rate", type=float, default=4e-3)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--lr_decay_iters", type=int, default=5000)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--warmup_iters", type=int, default=100)

    ap.add_argument("--model_parallel", type=int, default=4)

    return ap.parse_args()


# RPC init
def init_rpc(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    opts = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=256,
        init_method=f"tcp://{master_addr}:{master_port}",
    )
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=opts
    )


# Data loader
def _memmap_split(data_dir: str, split: str):
    fn = "train.bin" if split == "train" else "val.bin"
    arr = np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode="r")
    return arr

def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, use_ignore_index: bool = True):
    """
    Returns CPU tensors:
    X: (B, T) token IDs
    Y: (B, T) next-token labels
    If use_ignore_index=True, sets Y[:, -1] = -1 for compatibility with CE(ignore_index=-1).
    """
    data = _memmap_split(data_dir, split)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if use_ignore_index:
        y[:, -1] = -1
    return x.cpu(), y.cpu()


# Evaluation (uses forward pass only)
@torch.no_grad()
def estimate_loss(pipe: RPCPipeline, cfg: GPTConfig, data_dir: str, batch_size: int, block_size: int, eval_iters: int, use_ignore_index: bool = True):
    out = {}

    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(data_dir, split, batch_size, block_size, use_ignore_index=use_ignore_index)
            logits = pipe.forward(X)  # (B, T, V)
            if use_ignore_index:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    return out

# LR schedule
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / max(1, (warmup_iters + 1))
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    args = parse_args()
    rank, world = args.rank, args.world_size

    if torch.cuda.is_available():
        local_n = torch.cuda.device_count()
        torch.cuda.set_device(rank % max(1, local_n))

    init_rpc(rank, world, args.master_addr, args.master_port)

    if rank == 0:
        # ==== dataset & meta ====
        data_dir = os.path.join("data", args.dataset)
        os.makedirs(args.out_dir, exist_ok=True)
        meta_path = os.path.join(data_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = int(meta["vocab_size"])  # 65

        # ==== model config ====
        cfg = GPTConfig(
            block_size=args.block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bias=True
        )

        # ==== workers/devices ====
        # 8 workers
        workers = [f"worker{i}" for i in range(8)]
        devices = [
            "cuda:0","cuda:1","cuda:2","cuda:3",  # worker0..3 on node A
            "cuda:0","cuda:1","cuda:2","cuda:3",  # worker4..7 on node B
        ]

        # ==== build pipeline ====
        pipe = RPCPipeline(cfg, workers, devices)

        # ==== logging header ====
        tokens_per_iter = args.gradient_accumulation_steps * args.batch_size * args.block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

        # ==== training loop ====
        best_val = float("inf")
        running_tok_per_s = None
        t0 = time.time()

        for iter_num in range(args.max_iters):

            X, Y = get_batch(data_dir, "train", args.batch_size, args.block_size, use_ignore_index=True)

            lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.lr_decay_iters, args.min_lr)

            t_step0 = time.time()
            loss = pipe.train_step(X, Y)
            dt = time.time() - t_step0

            # iter log
            if iter_num % args.log_interval == 0:
                tok_per_s = tokens_per_iter / max(dt, 1e-9)
                running_tok_per_s = tok_per_s if running_tok_per_s is None else 0.9 * running_tok_per_s + 0.1 * tok_per_s
                print(f"iter {iter_num}: loss {loss:.4f}, time {dt*1000:.2f}ms, tok/s {running_tok_per_s:,.0f}")

            # eval loss
            if iter_num % args.eval_interval == 0 and iter_num > 0:
                losses = estimate_loss(
                    pipe, cfg, data_dir,
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                    eval_iters=args.eval_iters,
                    use_ignore_index=True
                )
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if losses["val"] < best_val:
                    best_val = losses["val"]

        total_dt = time.time() - t0
        print(f"[Driver] Done. total time {total_dt:.2f}s")

    rpc.shutdown()

if __name__ == "__main__":
    main()
