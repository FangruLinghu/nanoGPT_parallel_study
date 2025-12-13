# exp_logger.py
import os
import csv
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


def _compute_model_size(model, dtype_str: str):
    """Params count and cuda memory"""
    param_count = sum(p.numel() for p in model.parameters())

    if dtype_str == "float32":
        bytes_per_param = 4
    elif dtype_str in ["float16", "bfloat16"]:
        bytes_per_param = 2
    else:
        print("!dtype is not float32, float16 or bfloat16")
        bytes_per_param = 4

    weights_bytes = param_count * bytes_per_param
    weights_gb = weights_bytes / (1024 ** 3)

    # estimated: weights + gradients + Adam(2 state) ≈ 4 times weights
    train_state_bytes = weights_bytes * 4
    train_state_gb = train_state_bytes / (1024 ** 3)

    return param_count, weights_gb, train_state_gb


@dataclass
class RunSummary:
    run_id: str
    parallel_mode: str              # "single", "dp", "tp", "pp", "zero1", "zero2", "zero3"
    n_gpu: int

    # model structure/size
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dtype: str

    param_count: int
    model_size_GB: float
    train_state_est_GB: float

    # batch / parallel
    global_batch_size: int
    micro_batch_size_per_gpu: int
    grad_accum_steps: int

    # performance
    avg_time_per_iter: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    mfu: Optional[float] = None           # 0~1

    # GPU mem/usage
    peak_mem_rank0_GB: Optional[float] = None
    avg_gpu_util_pct: Optional[float] = None

    # loss
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None

    notes: Optional[str] = None


class ParallelRunLogger:
    """
    Record the logs to csv
    """

    def __init__(
        self,
        *,
        raw_model,
        dtype_str: str,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        parallel_mode: str,
        n_gpu: int,
        micro_batch_size_per_gpu: int,
        grad_accum_steps: int,
        run_id: str,
        results_path: str = "results.csv",
        master_process: bool = True,
        notes: Optional[str] = None,
        dp_size=None,
        param_count_override=None
    ):
        self.master_process = master_process
        self.results_path = results_path

        self.dp_size = dp_size if dp_size is not None else max(n_gpu, 1)

        # compute model size
        # param_count, weights_gb, train_state_gb = _compute_model_size(raw_model, dtype_str)

        param_count_local, weights_gb_local, train_state_gb_local = _compute_model_size(
            raw_model, dtype_str
        )

        if param_count_override is not None:
            param_count = param_count_override

            scale = param_count_override / max(param_count_local, 1)
            weights_gb = weights_gb_local * scale
            train_state_gb = train_state_gb_local * scale
        else:
            param_count = param_count_local
            weights_gb = weights_gb_local
            train_state_gb = train_state_gb_local

        self.param_count = param_count
        self.model_size_gb = weights_gb
        self.train_state_gb = train_state_gb

        if self.master_process:
            print(
                f"[RunLogger] Model params: {param_count:,} "
                f"({param_count/1e6:.2f} M), weights ~ {weights_gb:.3f} GB, "
                f"full train state ~ {train_state_gb:.3f} GB (theoretical)"
            )

        global_batch = micro_batch_size_per_gpu * grad_accum_steps * self.dp_size

        self.summary = RunSummary(
            run_id=run_id,
            parallel_mode=parallel_mode,
            n_gpu=n_gpu,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            dtype=dtype_str,
            param_count=param_count,
            model_size_GB=weights_gb,
            train_state_est_GB=train_state_gb,
            global_batch_size=global_batch,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            grad_accum_steps=grad_accum_steps,
            notes=notes,
        )

    def log_summary(
        self,
        *,
        avg_time_per_iter: Optional[float],
        tokens_per_sec: Optional[float],
        peak_mem_rank0_bytes: Optional[int] = None,
        final_train_loss: Optional[float] = None,
        final_val_loss: Optional[float] = None,
        mfu: Optional[float] = None,  
        avg_gpu_util_pct: Optional[float] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):

        if not self.master_process:
            return  # write only on master process

        self.summary.avg_time_per_iter = avg_time_per_iter
        self.summary.tokens_per_sec = tokens_per_sec
        self.summary.final_train_loss = float(final_train_loss) if final_train_loss is not None else None
        self.summary.final_val_loss = float(final_val_loss) if final_val_loss is not None else None
        self.summary.mfu = float(mfu) if mfu is not None else None
        self.summary.avg_gpu_util_pct = float(avg_gpu_util_pct) if avg_gpu_util_pct is not None else None

        if peak_mem_rank0_bytes is not None:
            self.summary.peak_mem_rank0_GB = peak_mem_rank0_bytes / (1024 ** 3)

        row = asdict(self.summary)

        if extra_fields:
            row.update(extra_fields)

        file_exists = os.path.exists(self.results_path)

        fieldnames = list(row.keys())

        with open(self.results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"[RunLogger] Summary written to {self.results_path}")
