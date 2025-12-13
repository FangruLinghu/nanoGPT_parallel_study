# comm_monitor.py
import time
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

class CommMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.comm_time = 0.0
        self.comm_bytes = 0
        self.collectives = 0
    
    def add_comm(self, tensor, duration):
        self.comm_time += duration
        if tensor is not None and hasattr(tensor, "numel"):
            self.comm_bytes += tensor.numel() * tensor.element_size()
        self.collectives += 1

    @property
    def bandwidth(self):
        if self.comm_time == 0:
            return 0.0
        return (self.comm_bytes / 1e9) / self.comm_time  # GB/s

monitor = CommMonitor()

def wrap(func):
    def wrapper(tensor, *args, **kwargs):
        start = time.time()
        result = func(tensor, *args, **kwargs)
        dist.barrier()  # sync for accurate timing
        duration = time.time() - start
        monitor.add_comm(tensor, duration)
        return result
    return wrapper

# wrap NCCL collective ops
c10d.all_reduce = wrap(c10d.all_reduce)
c10d.all_gather = wrap(c10d.all_gather)
c10d.reduce_scatter = wrap(c10d.reduce_scatter)
c10d.broadcast = wrap(c10d.broadcast)
