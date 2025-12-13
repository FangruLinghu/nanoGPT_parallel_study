import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from model import GPT


def init_2d_mesh(tp_size: int):
    """
    initialize 2D DeviceMesh: ('dp', 'tp')

    world_size = dp_size * tp_size
    - world_size=4 -> dp_size=1, tp_size=4
    - world_size=8 -> dp_size=2, tp_size=4
    """
    assert dist.is_initialized(), "Please call dist.init_process_group() first."

    world_size = dist.get_world_size()
    assert world_size % tp_size == 0, \
        f"world_size={world_size} must be divisible by tp_size={tp_size}"
    dp_size = world_size // tp_size

    # 2D mesh, dim0: dp, dim1: tp
    mesh = init_device_mesh(
        "cuda",
        (dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    # mesh：tp dimension & dp dimension
    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]

    return mesh, dp_mesh, tp_mesh


def apply_tensor_parallel_to_gpt(model: GPT, tp_mesh):
    """
    make sure n_head can be divided by tp_size 
    """
    
    # un-tying
    with torch.no_grad():
        if model.transformer.wte.weight is model.lm_head.weight:
            w = model.transformer.wte.weight.detach().clone()
            model.transformer.wte.weight = nn.Parameter(w)
            model.lm_head.weight = nn.Parameter(w.clone())
    
    tp_size = tp_mesh.size()
    n_head = model.config.n_head
    assert n_head % tp_size == 0, \
        f"n_head ({n_head}) must be divisible by tp_size ({tp_size})"
    
    layer_tp_plan = {
        # "attn.c_attn": ColwiseParallel(use_local_output=True),
        # "attn.c_proj": RowwiseParallel(),
        "mlp.c_fc": ColwiseParallel(use_local_output=True),
        "mlp.c_proj": RowwiseParallel(),
    }
    
    for block in model.transformer.h:
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    
    return model

