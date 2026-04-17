# -*- coding: utf-8 -*-
"""
stage4_plot_results.py

功能：
1. 读取全区正式结果表
2. 绘制全区正式总体指标图
3. 绘制全区正式分步长曲线图
4. 保存到 results_summary/full_figures
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 中文显示
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_data(summary_root: str):
    summary_root = Path(summary_root)

    overall_path = summary_root / "summary_overall_full_formal.csv"
    per_step_path = summary_root / "summary_per_step_full_formal.csv"

    if not overall_path.exists():
        raise FileNotFoundError(f"未找到文件: {overall_path}")
    if not per_step_path.exists():
        raise FileNotFoundError(f"未找到文件: {per_step_path}")

    df_overall = pd.read_csv(overall_path, encoding="utf-8-sig")
    df_per_step = pd.read_csv(per_step_path, encoding="utf-8-sig")
    return df_overall, df_per_step, summary_root


def shorten_model_name(name: str) -> str:
    mapping = {
        "only_s_direct_cmp6_full_tuned": "Only-S Direct",
        "lag5_gw_direct_cmp6_full": "Lag5-GW Direct",
        "only_s_recursive_cmp6_full": "Only-S Rec",
    }
    return mapping.get(name, name)


def save_fig(fig, save_path: Path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")


def add_bar_labels(ax, bars, fmt="{:.4f}", fontsize=10):
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.02
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def plot_pretty_bar(df: pd.DataFrame, metric: str, title: str, save_path: Path):
    labels = [shorten_model_name(x) for x in df["model"].tolist()]
    values = df[metric].tolist()
    x = np.arange(len(labels))

    # 图稍微放宽一点
    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    # 柱子变窄
    bars = ax.bar(x, values, width=0.38)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", va="top")
    ax.set_ylabel(metric)
    ax.set_title(title, pad=10)

    # 网格更淡一点，减少干扰
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)

    # 只保留左、下边框更干净；如果你想保留全部边框，这两句可以删掉
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = min(values)
    ymax = max(values)
    span = max(ymax - ymin, 1e-6)

    # 上下留白更克制一些
    ax.set_ylim(ymin - span * 0.12, ymax + span * 0.16)

    if metric == "R2":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
        add_bar_labels(ax, bars, fmt="{:.4f}", fontsize=10)
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
        add_bar_labels(ax, bars, fmt="{:.4f}", fontsize=10)

    save_fig(fig, save_path)


def plot_overall_bars(df_overall: pd.DataFrame, fig_dir: Path):
    plot_pretty_bar(
        df_overall,
        metric="MAE",
        title="全区正式结果 MAE 对比",
        save_path=fig_dir / "full_overall_MAE.png",
    )
    plot_pretty_bar(
        df_overall,
        metric="RMSE",
        title="全区正式结果 RMSE 对比",
        save_path=fig_dir / "full_overall_RMSE.png",
    )
    plot_pretty_bar(
        df_overall,
        metric="R2",
        title="全区正式结果 R2 对比",
        save_path=fig_dir / "full_overall_R2.png",
    )


def plot_per_step_curves(df_per_step: pd.DataFrame, fig_dir: Path):
    step_cols = {
        "MAE": [f"step{i}_MAE" for i in range(1, 6)],
        "RMSE": [f"step{i}_RMSE" for i in range(1, 6)],
        "R2": [f"step{i}_R2" for i in range(1, 6)],
    }
    x = [1, 2, 3, 4, 5]

    for metric, cols in step_cols.items():
        fig = plt.figure(figsize=(8.5, 5.2))
        for _, row in df_per_step.iterrows():
            y = [row[c] for c in cols]
            plt.plot(x, y, marker="o", linewidth=2, label=shorten_model_name(row["model"]))

        plt.xticks(x)
        plt.xlabel("预测步长")
        plt.ylabel(metric)
        plt.title(f"全区正式结果分步长 {metric} 对比")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_fig(fig, fig_dir / f"full_per_step_{metric}.png")


def main():
    df_overall, df_per_step, summary_root = load_data(
        r"F:\aaa1\建模预测分析\results_summary"
    )

    fig_dir = summary_root / "full_figures"
    ensure_dir(fig_dir)

    plot_overall_bars(df_overall, fig_dir)
    plot_per_step_curves(df_per_step, fig_dir)

    print("=" * 80)
    print("[INFO] 全区正式图件已生成")
    print(f"[INFO] 图件目录: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()