import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.autograd import backward as dist_backward
from torch.distributed.autograd import context as dist_autograd_context


# Debugging functions
def _probe(worker_name: str):
    import torch
    return {
        "worker": worker_name,
        "cuda": torch.cuda.is_available(),
        "count": torch.cuda.device_count(),
        "current": torch.cuda.current_device() if torch.cuda.is_available() else -1,
    }

def _validate_device(device_str: str) -> bool:
    import torch
    if device_str == "cpu":
        return True
    if device_str.startswith("cuda:"):
        try:
            idx = int(device_str.split(":")[1])
        except Exception:
            return False
        return torch.cuda.is_available() and (0 <= idx < torch.cuda.device_count())
    return False

# Config & GPT components
@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            mask = torch.tril(torch.ones((T, T), device=att.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Each stage computes on the local GPU; RPC boundaries use CPU tensors.
class EmbeddingStage(nn.Module):
    def __init__(self, cfg: GPTConfig, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.to(self.device)

    @torch.inference_mode(False)
    def forward(self, idx_cpu: torch.Tensor):
        # idx_cpu: (B, T) on CPU
        b, t = idx_cpu.shape
        assert t <= self.cfg.block_size
        idx = idx_cpu.to(self.device, non_blocking=True)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        x = self.wte(idx) + self.wpe(pos)  # (B, T, C)
        x = self.drop(x)
        return x.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

class BlockStage(nn.Module):
    def __init__(self, cfg: GPTConfig, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.block = Block(cfg).to(self.device)

    @torch.inference_mode(False)
    def forward(self, x_cpu: torch.Tensor):
        x = x_cpu.to(self.device, non_blocking=True)
        x = self.block(x)
        return x.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

class OutputStage(nn.Module):
    def __init__(self, cfg: GPTConfig, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias).to(self.device)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False).to(self.device)

    @torch.inference_mode(False)
    def forward(self, x_cpu: torch.Tensor):
        x = x_cpu.to(self.device, non_blocking=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


# RPC Pipeline (rank0 holds RRefs and wraps forward/train_step)

class RPCPipeline:
    def __init__(self, cfg: GPTConfig, workers: List[str], devices: List[str]):
        """
        8 workers
        worker0: EmbeddingStage
        worker1 to 6: BlockStage
        worker7: OutputStage
        """
        assert len(workers) == 8, "Expected 8 workers"
        assert len(devices) == 8

        for w, d in zip(workers, devices):
            info = rpc.rpc_sync(w, _probe, args=(w,))
            print(f"[Probe] {info['worker']}: cuda={info['cuda']}, "
                f"device_count={info['count']}, current={info['current']}, want={d}")
            ok = rpc.rpc_sync(w, _validate_device, args=(d,))
            if not ok:
                raise RuntimeError(
                    f"[DeviceMapError] There is no {d} on worker={w}"
                    f"Please check devices."
                )

        # create stage
        self.emb_rref = rpc.remote(workers[0], EmbeddingStage, args=(cfg, devices[0]))
        self.block_rrefs = [
            rpc.remote(workers[i+1], BlockStage, args=(cfg, devices[i+1]))
            for i in range(6)
        ]
        self.out_rref = rpc.remote(workers[7], OutputStage, args=(cfg, devices[7]))

        # Distributed optimizer
        all_param_rrefs = []
        all_param_rrefs.extend(self.emb_rref.rpc_sync().parameter_rrefs())
        for br in self.block_rrefs:
            all_param_rrefs.extend(br.rpc_sync().parameter_rrefs())
        all_param_rrefs.extend(self.out_rref.rpc_sync().parameter_rrefs())

        from torch.distributed.optim import DistributedOptimizer

        self.opt = DistributedOptimizer(
            torch.optim.AdamW,
            all_param_rrefs,
            lr=4e-3, betas=(0.9, 0.99), weight_decay=0.1,
        )
        self.cfg = cfg

    def forward(self, idx_cpu: torch.Tensor):
        x_cpu = self.emb_rref.rpc_sync().forward(idx_cpu)
        for br in self.block_rrefs:
            x_cpu = br.rpc_sync().forward(x_cpu)
        logits_cpu = self.out_rref.rpc_sync().forward(x_cpu)
        return logits_cpu

    def train_step(self, idx_cpu: torch.Tensor, tgt_cpu: torch.Tensor) -> float:
        with dist_autograd_context() as cid:
            logits = self.forward(idx_cpu)  # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_cpu.view(-1),
                ignore_index=-1
            )
            dist_backward(cid, [loss])
            self.opt.step(cid)
        return float(loss.item())

    def set_train(self, mode: bool = True):
        self.emb_rref.rpc_sync().train(mode)
        for br in self.block_rrefs:
            br.rpc_sync().train(mode)
        self.out_rref.rpc_sync().train(mode)