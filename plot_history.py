"""从真实的 history.csv 绘制训练和验证损失曲线。"""

import argparse
import csv
from pathlib import Path


def plot_history(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("history.csv 没有训练记录")

    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    val = [float(row["val_loss"]) for row in rows]
    best_index = min(range(len(val)), key=val.__getitem__)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    ax.plot(epochs, train, color="#2563eb", marker="o", linewidth=2,
            label="Train loss")
    ax.plot(epochs, val, color="#e76f51", marker="o", linewidth=2,
            label="Validation loss")
    ax.scatter([epochs[best_index]], [val[best_index]], color="#b91c1c",
               s=90, zorder=4, label=f"Best epoch: {epochs[best_index]}")
    ax.set(title="Transformer translation learning curve",
           xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.set_xticks(epochs if len(epochs) <= 20 else epochs[::max(1, len(epochs) // 10)])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(run_dir / "loss_curve.png", dpi=180, facecolor="white")
    fig.savefig(run_dir / "loss_curve.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重绘实验损失曲线")
    parser.add_argument("run_dir", type=Path)
    plot_history(parser.parse_args().run_dir)
