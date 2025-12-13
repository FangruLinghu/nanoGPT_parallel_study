import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTConfig, GPT

class ModelParallelGPT(GPT):
    def __init__(self, config):
        super().__init__(config)

        n_layers = len(self.transformer.h)
        half = n_layers // 2

        self.transformer.wte.weight = nn.Parameter(self.transformer.wte.weight.detach().clone())
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())

        # split the blocks
        self.seq1 = nn.Sequential(*self.transformer.h[:half]).to('cuda:0')
        self.seq2 = nn.Sequential(*self.transformer.h[half:]).to('cuda:1')

        # other layers
        self.transformer.wte.to('cuda:0')
        self.transformer.wpe.to('cuda:0')
        self.transformer.drop.to('cuda:0')
        self.transformer.ln_f.to('cuda:1')
        self.lm_head.to('cuda:1')

        print(f"[ModelParallelGPT] {half} layers on cuda:0, {n_layers - half} layers on cuda:1")

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')

        pos = torch.arange(0, t, dtype=torch.long, device=device0)
        tok_emb = self.transformer.wte(idx.to(device0))  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)              # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # half blocks on cuda:0
        x = self.seq1(x)

        # send to cuda:1
        x = x.to(device1)

        for name, param in self.seq2.named_parameters():
            if param.device != torch.device('cuda:1'):
                print("[Mismatch]", name, param.device)

        # half blocks on cuda:1
        x = self.seq2(x)
        x = self.transformer.ln_f(x)

        # output and loss on cuda:1
        logits = self.lm_head(x)

        if targets is not None:
            # training
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.to(device1).view(-1),
                ignore_index=-1
            )
        else:
            # inference
            logits = logits[:, [-1], :]
            loss = None

        return logits, loss



class nModelParallelGPT(GPT):
    """
    Split self.transformer.h into mp_size stages and place them to cuda:0..cuda:mp_size-1.
    Embedding (wte/wpe/drop) stays on first device; ln_f/lm_head on last device.
    """
    def __init__(self, config: GPTConfig, mp_size: int = 1):
        super().__init__(config)

        self.transformer.wte.weight = nn.Parameter(self.transformer.wte.weight.detach().clone())
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())

        assert mp_size >= 1, "mp_size must be >= 1"
        self.mp_size = mp_size
        self.devices: List[torch.device] = [torch.device(f"cuda:{i}") for i in range(mp_size)]
        self.first_dev = self.devices[0]
        self.last_dev  = self.devices[-1]

        # Split blocks to stages
        blocks = list(self.transformer.h)  # length = n_layers
        n = len(blocks)
        q, r = divmod(n, mp_size)
        chunks = []
        start = 0
        for i in range(mp_size):
            end = start + q + (1 if i < r else 0)
            chunks.append(blocks[start:end])
            start = end

        # Each stage as a Sequential on its own device
        self.stages = nn.ModuleList()
        for i, chunk in enumerate(chunks):
            stage = nn.Sequential(*chunk) if len(chunk) > 0 else nn.Sequential()
            stage.to(self.devices[i])
            self.stages.append(stage)

        # Other layers
        # Move input-side to first device
        self.transformer.wte.to(self.first_dev)
        self.transformer.wpe.to(self.first_dev)
        self.transformer.drop.to(self.first_dev)

        # Move output-side to last device
        self.transformer.ln_f.to(self.last_dev)
        self.lm_head.to(self.last_dev)

        # Checking
        counts = [len(s) for s in self.stages]
        print(f"[ModelParallelGPT] stages per device: {counts}")
        for i, stage in enumerate(self.stages):
            for bi, b in enumerate(stage):
                print(f"  - Block@device cuda:{i}: {getattr(b, 'ln_1', type('x',(),{})()).__class__.__name__}")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # embeddings on first device
        pos = torch.arange(0, t, dtype=torch.long, device=self.first_dev)
        tok_emb = self.transformer.wte(idx.to(self.first_dev, non_blocking=True))   # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)                                         # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # pass through stages sequentially
        for dev, stage in zip(self.devices, self.stages):
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            if len(stage) > 0:
                x = stage(x)

        # tail on last device
        if x.device != self.last_dev:
            x = x.to(self.last_dev, non_blocking=True)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.to(self.last_dev, non_blocking=True).view(-1),
                ignore_index=-1
            )
        else:
            logits = logits[:, [-1], :]
            loss = None
        return logits, loss