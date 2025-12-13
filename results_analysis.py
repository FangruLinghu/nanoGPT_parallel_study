import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

os.makedirs("results", exist_ok=True)
# ======================
# Load and filter results
# ======================
csv_path = "results.csv"
df = pd.read_csv(csv_path)

# Keep only our target dataset + dtype
df = df[df["dataset"] == "shakespeare_char"]
df = df[df["dtype"] == "float32"]

# ======================
# Label model size (S/M/L/XL)
# ======================
def assign_model_size(row):
    n_layer = row["n_layer"]
    n_embd = row["n_embd"]
    if n_layer == 6 and n_embd == 512:
        return "S (6x512)"
    elif n_layer == 24 and n_embd == 1024:
        return "M (24x1024)"
    elif n_layer == 32 and n_embd == 1536:
        return "L (32x1536)"
    elif n_layer == 48 and n_embd == 2048:
        return "XL (48x2048)"
    else:
        return f"Other({int(n_layer)}x{int(n_embd)})"

df["model_size_tag"] = df.apply(assign_model_size, axis=1)

# model size in million params
df["param_M"] = df["param_count"] / 1e6

# ======================
# Relative metrics vs. single-GPU baseline
# ======================

def add_rel_to_single(group: pd.DataFrame) -> pd.DataFrame:
    """
    For each model_size_tag group, compute speedup and time ratios
    relative to the single-GPU run of the same model size.
    """
    single_rows = group[group["parallel_mode"] == "single"]
    if single_rows.empty:
        # no single-GPU baseline available
        group["speedup_vs_single"] = np.nan
        group["time_ratio_vs_single"] = np.nan
        group["runtime_ratio_vs_single"] = np.nan
        return group

    baseline_tokens = single_rows["tokens_per_sec"].mean()
    baseline_time = single_rows["avg_time_per_iter"].mean()
    baseline_runtime = single_rows["total_runtime_sec"].mean()

    group["speedup_vs_single"] = group["tokens_per_sec"] / baseline_tokens
    group["time_ratio_vs_single"] = group["avg_time_per_iter"] / baseline_time
    group["runtime_ratio_vs_single"] = group["total_runtime_sec"] / baseline_runtime
    return group

df = df.groupby("model_size_tag", group_keys=False).apply(add_rel_to_single)

# ======================
# Summary tables by model size
# ======================

cols_for_summary = [
    "parallel_mode",
    "param_M",
    "avg_time_per_iter",
    "tokens_per_sec",
    "total_runtime_sec",
    "peak_mem_rank0_GB",
    "avg_gpu_util_pct"
]

print("\n================ SUMMARY BY MODEL SIZE ================\n")

latex_tables = {}

for size_tag in sorted(df["model_size_tag"].unique()):
    sub = df[df["model_size_tag"] == size_tag].copy()

    # average over repeated runs for the same (model_size_tag, parallel_mode)
    grouped = (
        sub.groupby("parallel_mode", as_index=True)[cols_for_summary[1:]]
        .mean()
        .reset_index()
        .set_index("parallel_mode")
    )

    print(f"--- Model size: {size_tag} ---")
    print(grouped)
    print()

    latex = grouped.to_latex(
        float_format="%.2f",
        caption=f"Results for model {size_tag}",
        label=f"tab:results_{size_tag.replace(' ', '').replace('(', '').replace(')', '').replace('x', 'x')}",
    )
    latex_tables[size_tag] = latex

# make sure output dir exists
os.makedirs("results", exist_ok=True)

with open("results/tables.tex", "w") as f:
    for size_tag, latex in latex_tables.items():
        f.write("% ==============================\n")
        f.write(f"% {size_tag}\n")
        f.write("% ==============================\n")
        f.write(latex)
        f.write("\n\n")

print("LaTeX tables written to results/tables.tex")



# ======================
# plots
# ======================

# --- Matplotlib global style ---
font_path = "results/TimesNewRoman.ttf"
fm.fontManager.addfont(font_path)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
})

mode_colors = {
    "single": "#929591",  
    "dp":     "#F3C95A",  
    "mp":     "#DDA0DD",  #CFBACB
    "tp":     "#FA8072",  #D2691E
    "zero1":  "#5395F4", 
    "zero2":  "#62BDB6", 
}


# clean group: average over repeated runs
plot_df = (
    df.groupby(["model_size_tag", "parallel_mode"], as_index=False)
      .agg({
          "param_M": "mean",
          "tokens_per_sec": "mean",
          "peak_mem_rank0_GB": "mean",
          "avg_gpu_util_pct": "mean",
          "avg_time_per_iter": "mean",
          "total_runtime_sec": "mean",
      })
)

# set the order of parallel_mode
parallel_order = ["single", "dp", "mp", "tp", "zero1", "zero2", "zero3"]
plot_df["parallel_mode"] = pd.Categorical(
    plot_df["parallel_mode"], categories=parallel_order, ordered=True
)
plot_df = plot_df.sort_values(["parallel_mode", "param_M"])

# ========== compute ratio ==========
baseline = (
    plot_df[plot_df["parallel_mode"] == "single"]
    .loc[:, ["model_size_tag", "avg_time_per_iter", "total_runtime_sec"]]
    .rename(columns={
        "avg_time_per_iter": "base_iter_time",
        "total_runtime_sec": "base_total_time",
    })
)

plot_df = plot_df.merge(baseline, on="model_size_tag", how="left")

plot_df["iter_time_ratio"] = (
    plot_df["avg_time_per_iter"] / plot_df["base_iter_time"]
)
plot_df["total_runtime_ratio"] = (
    plot_df["total_runtime_sec"] / plot_df["base_total_time"]
)

# norm single to 1
plot_df.loc[plot_df["parallel_mode"] == "single", "iter_time_ratio"] = 1.0
plot_df.loc[plot_df["parallel_mode"] == "single", "total_runtime_ratio"] = 1.0

x_label = "Model size (million parameters)"

def _nice_ax(ax):
    # light inner grid
    ax.grid(
        True,
        which="both",        # major + minor grid
        axis="both",
        linestyle="-",
        linewidth=0.4,
        color="#e5e5e5",
        alpha=0.8,
    )

    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#999999")

    # ticks
    ax.tick_params(axis="both", which="both", width=0.6, length=3)

# =======================
# 1) Avg GPU util vs param
# =======================
fig, ax = plt.subplots(figsize=(6.0, 4.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for mode in parallel_order:
    # skip single-GPU here to avoid mixing 1-GPU util with 4-GPU ones
    if mode == "single":
        continue
    sub = plot_df[plot_df["parallel_mode"] == mode]
    if sub.empty:
        continue

    ax.plot(
        sub["param_M"],
        sub["avg_gpu_util_pct"],
        marker="o",
        linewidth=0.9,
        markersize=3.5,
        label=mode,
        color=mode_colors.get(mode, "#000000"),
    )

ax.set_xlabel(x_label)
ax.set_ylabel("Average GPU utilization (%)")
# ax.set_title("Average GPU utilization vs. model size")
_nice_ax(ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=len(labels),
    frameon=False,
)

plt.tight_layout()
plt.savefig("results/gpu_util_vs_param.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =======================
# 2) Peak mem vs param
# =======================
fig, ax = plt.subplots(figsize=(6.0, 4.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for mode in parallel_order:
    sub = plot_df[plot_df["parallel_mode"] == mode]
    if sub.empty:
        continue

    ax.plot(
        sub["param_M"],
        sub["peak_mem_rank0_GB"],
        marker="o",
        linewidth=0.9,
        markersize=3.5,
        label=mode,
        color=mode_colors.get(mode, "#000000"),
    )

ax.set_xlabel(x_label)
ax.set_ylabel("Peak memory on rank 0 (GB)")
# ax.set_title("Peak memory vs. model size")
_nice_ax(ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=len(labels),
    frameon=False,
)

plt.tight_layout()
plt.savefig("results/peak_mem_vs_param.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =======================
# 3) Iteration time ratio vs model size
# =======================
fig, ax = plt.subplots(figsize=(6.0, 4.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for mode in parallel_order:
    sub = plot_df[plot_df["parallel_mode"] == mode]
    if sub.empty:
        continue

    ax.plot(
        sub["param_M"],
        sub["iter_time_ratio"],
        marker="o",
        linewidth=0.9,
        markersize=3.5,
        label=mode,
        color=mode_colors.get(mode, "#000000"),
    )

# single-GPU baseline
ax.axhline(1.0, linestyle="--", linewidth=0.8, color="#666666")
ax.set_xlabel(x_label)
ax.set_ylabel("Iteration time / single-GPU")
# ax.set_title("Iteration time ratio vs. model size")
_nice_ax(ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=len(labels),
    frameon=False,
)

plt.tight_layout()
plt.savefig("results/iter_time_ratio_vs_model_size.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =======================
# 4) Total runtime ratio vs model size
# =======================
fig, ax = plt.subplots(figsize=(6.0, 4.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for mode in parallel_order:
    sub = plot_df[plot_df["parallel_mode"] == mode]
    if sub.empty:
        continue

    ax.plot(
        sub["param_M"],
        sub["total_runtime_ratio"],
        marker="o",
        linewidth=0.9,
        markersize=3.5,
        label=mode,
        color=mode_colors.get(mode, "#000000"),
    )

ax.axhline(1.0, linestyle="--", linewidth=0.8, color="#666666")
ax.set_xlabel(x_label)
ax.set_ylabel("Total runtime / single-GPU")
# ax.set_title("Total runtime ratio vs. model size")
_nice_ax(ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=len(labels),
    frameon=False,
)

plt.tight_layout()
plt.savefig("results/total_runtime_ratio_vs_model_size.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved figures:\n"
      "  - gpu_util_vs_param.png\n"
      "  - peak_mem_vs_param.png\n"
      "  - iter_time_ratio_vs_model_size.png\n"
      "  - total_runtime_ratio_vs_model_size.png")
