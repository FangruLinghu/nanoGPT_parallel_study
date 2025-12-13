import re
import matplotlib.pyplot as plt

log_files = ["output_1g.log", "output_2n.log"]
labels = ["1-GPU", "8-GPUs"]
colors = ["tab:blue", "tab:orange"]

pattern_loss = re.compile(r"step (\d+): train loss ([\d\.]+), val loss ([\d\.]+)")
pattern_time = re.compile(r"iter \d+: loss [\d\.]+, time ([\d\.]+)ms")

plt.figure(figsize=(8,5))

for log_file, label, color in zip(log_files, labels, colors):
    with open(log_file, "r") as f:
        logs = f.readlines()

    steps, train_losses, val_losses = [], [], []
    total_time = 0.0

    for line in logs:
        # extract train/val loss
        match = pattern_loss.search(line)
        if match:
            step = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # extract iter time
        match_time = pattern_time.search(line)
        if match_time:
            total_time += float(match_time.group(1)) / 1000.0

    # print execution time
    print(f"{label} total execution time: {total_time/60:.2f} minutes")

    # plot
    plt.plot(steps, train_losses, label=f"{label} Train", color=color, linestyle="-")
    plt.plot(steps, val_losses, label=f"{label} Val", color=color, linestyle="--")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("loss_curve_compare2.png")
plt.show()
