# -*- coding: utf-8 -*-
"""
stage4_plot_results.py

功能：
1. 读取 results_summary 下的汇总表
2. 绘制总体指标对比图
3. 绘制 lag1~lag6 单滞后变化曲线
4. 绘制 direct vs recursive 对比图
5. 绘制 step1~step5 分步长误差曲线
6. 绘制多滞后对比图
7. 保存图片到 results_summary/figures
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_data(summary_root: str):
    summary_root = Path(summary_root)
    overall_path = summary_root / "summary_overall.csv"
    per_step_path = summary_root / "summary_per_step.csv"

    if not overall_path.exists():
        raise FileNotFoundError(f"未找到文件: {overall_path}")
    if not per_step_path.exists():
        raise FileNotFoundError(f"未找到文件: {per_step_path}")

    df_overall = pd.read_csv(overall_path, encoding="utf-8-sig")
    df_per_step = pd.read_csv(per_step_path, encoding="utf-8-sig")
    return df_overall, df_per_step, summary_root


def save_fig(fig, save_path: Path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")


def shorten_model_name(name: str) -> str:
    mapping = {
        "only_s_direct_cmp6": "Only-S Direct",
        "only_s_recursive_cmp6": "Only-S Rec",
        "sync_gw_direct_cmp6": "Sync-GW Direct",
        "lag1_gw_direct_cmp6": "Lag1-GW Direct",
        "lag2_gw_direct_cmp6": "Lag2-GW Direct",
        "lag3_gw_direct_cmp6": "Lag3-GW Direct",
        "lag4_gw_direct_cmp6": "Lag4-GW Direct",
        "lag5_gw_direct_cmp6": "Lag5-GW Direct",
        "lag6_gw_direct_cmp6": "Lag6-GW Direct",
        "lag5_gw_recursive_cmp6": "Lag5-GW Rec",
        "multi_lag_gw_3_5_direct_cmp6": "Multi(3,5)",
        "multi_lag_gw_5_6_direct_cmp6": "Multi(5,6)",
        "multi_lag_gw_2_3_5_direct_cmp6": "Multi(2,3,5)",
        # 旧结果
        "only_s_direct": "Only-S Direct(old)",
        "sync_gw_direct": "Sync-GW Direct(old)",
        "lag1_gw_direct": "Lag1-GW Direct(old)",
    }
    return mapping.get(name, name)


def decide_decimal_places(values) -> tuple[int, int]:
    vmin = min(values)
    vmax = max(values)
    span = abs(vmax - vmin)

    if span >= 1:
        axis_decimals = 2
    elif span >= 0.1:
        axis_decimals = 3
    elif span >= 0.01:
        axis_decimals = 4
    else:
        axis_decimals = 5

    label_decimals = axis_decimals + 1
    return axis_decimals, label_decimals


def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=9):
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.015
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


def plot_pretty_bar(df: pd.DataFrame, metric: str, title: str, save_path: Path,
                    figsize=(11, 5.8), rotate=0, width=0.62):
    # 数据校验：过滤空值和异常值
    df = df.dropna(subset=[metric])
    df = df[np.isfinite(df[metric])]
    if len(df) == 0:
        print(f"[WARN] 无有效数据绘制 {metric} 图表")
        return

    labels = [shorten_model_name(x) for x in df["model"].tolist()]
    values = df[metric].tolist()
    x = np.arange(len(labels))

    # 使用更轻量的绘图方式
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    bars = ax.bar(x, values, width=width, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="center", fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # 优化y轴范围计算
    ymin = min(values)
    ymax = max(values)
    span = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - span * 0.15, ymax + span * 0.18)

    # ========== 这里是修改的核心部分 ==========
    # 强制纵轴保留3位小数，柱子数值标签保留4位小数
    axis_decimals = 3
    label_decimals = 4
    # ========================================

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.{axis_decimals}f}")
    )

    # 添加标签（可选，注释掉可进一步提升性能）
    add_bar_labels(ax, bars, fmt=f"{{:.{label_decimals}f}}", fontsize=8)
    
    save_fig(fig, save_path)


def plot_overall_bars(df_overall: pd.DataFrame, fig_dir: Path):
    """
    正式可比结果：包含单滞后、多滞后、recursive
    """
    formal_order = [
        "only_s_direct_cmp6",
        "sync_gw_direct_cmp6",
        "lag1_gw_direct_cmp6",
        "lag2_gw_direct_cmp6",
        "lag3_gw_direct_cmp6",
        "lag4_gw_direct_cmp6",
        "lag5_gw_direct_cmp6",
        "lag6_gw_direct_cmp6",
        "multi_lag_gw_3_5_direct_cmp6",
        "multi_lag_gw_5_6_direct_cmp6",
        "multi_lag_gw_2_3_5_direct_cmp6",
        "only_s_recursive_cmp6",
        "lag5_gw_recursive_cmp6",
    ]

    df = df_overall[df_overall["model"].isin(formal_order)].copy()
    order_map = {name: i for i, name in enumerate(formal_order)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    plot_pretty_bar(
        df=df,
        metric="MAE",
        title="Overall MAE Comparison",
        save_path=fig_dir / "overall_MAE.png",
        figsize=(15, 6),
        rotate=0,
        width=0.56,
    )
    plot_pretty_bar(
        df=df,
        metric="RMSE",
        title="Overall RMSE Comparison",
        save_path=fig_dir / "overall_RMSE.png",
        figsize=(15, 6),
        rotate=0,
        width=0.56,
    )
    plot_pretty_bar(
        df=df,
        metric="R2",
        title="Overall R2 Comparison",
        save_path=fig_dir / "overall_R2.png",
        figsize=(15, 6),
        rotate=0,
        width=0.56,
    )


def plot_direct_lag_curve(df_overall: pd.DataFrame, fig_dir: Path):
    lag_rows = []
    pattern = re.compile(r"lag(\d+)_gw_direct_cmp6")

    for _, row in df_overall.iterrows():
        model = row["model"]
        m = pattern.fullmatch(model)
        if m:
            lag = int(m.group(1))
            lag_rows.append({
                "lag": lag,
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
            })

    if not lag_rows:
        print("[WARN] 未找到 lag1~lag6 direct cmp6 结果")
        return

    df_lag = pd.DataFrame(lag_rows).sort_values("lag")

    for metric in ["MAE", "RMSE", "R2"]:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(df_lag["lag"], df_lag[metric], marker="o")
        plt.xticks(df_lag["lag"])
        plt.xlabel("Lag")
        plt.ylabel(metric)
        plt.title(f"Direct Prediction Performance under Different Single Lags ({metric})")
        plt.grid(True, alpha=0.3)
        save_fig(fig, fig_dir / f"lag_curve_{metric}.png")


def plot_core_strategy_compare(df_overall: pd.DataFrame, fig_dir: Path):
    selected_models = [
        "only_s_direct_cmp6",
        "only_s_recursive_cmp6",
        "lag5_gw_direct_cmp6",
        "lag5_gw_recursive_cmp6",
    ]
    df = df_overall[df_overall["model"].isin(selected_models)].copy()

    core_order = [
        "only_s_direct_cmp6",
        "only_s_recursive_cmp6",
        "lag5_gw_direct_cmp6",
        "lag5_gw_recursive_cmp6",
    ]
    order_map = {name: i for i, name in enumerate(core_order)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    plot_pretty_bar(
        df=df,
        metric="MAE",
        title="Core Model Comparison (MAE)",
        save_path=fig_dir / "core_compare_MAE.png",
        figsize=(9.5, 5.6),
        rotate=0,
        width=0.50,
    )
    plot_pretty_bar(
        df=df,
        metric="RMSE",
        title="Core Model Comparison (RMSE)",
        save_path=fig_dir / "core_compare_RMSE.png",
        figsize=(9.5, 5.6),
        rotate=0,
        width=0.50,
    )
    plot_pretty_bar(
        df=df,
        metric="R2",
        title="Core Model Comparison (R2)",
        save_path=fig_dir / "core_compare_R2.png",
        figsize=(9.5, 5.6),
        rotate=0,
        width=0.50,
    )


def plot_step_curves(df_per_step: pd.DataFrame, fig_dir: Path):
    selected_models = [
        "only_s_direct_cmp6",
        "only_s_recursive_cmp6",
        "lag5_gw_direct_cmp6",
        "lag5_gw_recursive_cmp6",
    ]
    df = df_per_step[df_per_step["model"].isin(selected_models)].copy()

    core_order = selected_models
    order_map = {name: i for i, name in enumerate(core_order)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    step_cols = {
        "MAE": [f"step{i}_MAE" for i in range(1, 6)],
        "RMSE": [f"step{i}_RMSE" for i in range(1, 6)],
        "R2": [f"step{i}_R2" for i in range(1, 6)],
    }

    x = [1, 2, 3, 4, 5]

    for metric, cols in step_cols.items():
        fig = plt.figure(figsize=(9, 5))
        for _, row in df.iterrows():
            y = [row[c] for c in cols]
            plt.plot(x, y, marker="o", label=shorten_model_name(row["model"]))

        plt.xticks(x)
        plt.xlabel("Prediction Step")
        plt.ylabel(metric)
        plt.title(f"Per-step {metric} Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_fig(fig, fig_dir / f"per_step_{metric}.png")


def plot_multi_lag_compare(df_overall: pd.DataFrame, fig_dir: Path):
    """
    单滞后最优 vs 多滞后
    """
    selected_models = [
        "lag5_gw_direct_cmp6",
        "multi_lag_gw_3_5_direct_cmp6",
        "multi_lag_gw_5_6_direct_cmp6",
        "multi_lag_gw_2_3_5_direct_cmp6",
    ]
    df = df_overall[df_overall["model"].isin(selected_models)].copy()

    order_map = {name: i for i, name in enumerate(selected_models)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    plot_pretty_bar(
        df=df,
        metric="MAE",
        title="Single-lag Best vs Multi-lag Models (MAE)",
        save_path=fig_dir / "multi_lag_compare_MAE.png",
        figsize=(10, 5.5),
        rotate=0,
        width=0.52,
    )
    plot_pretty_bar(
        df=df,
        metric="RMSE",
        title="Single-lag Best vs Multi-lag Models (RMSE)",
        save_path=fig_dir / "multi_lag_compare_RMSE.png",
        figsize=(10, 5.5),
        rotate=0,
        width=0.52,
    )
    plot_pretty_bar(
        df=df,
        metric="R2",
        title="Single-lag Best vs Multi-lag Models (R2)",
        save_path=fig_dir / "multi_lag_compare_R2.png",
        figsize=(10, 5.5),
        rotate=0,
        width=0.52,
    )


def plot_gw_scheme_compare(df_overall: pd.DataFrame, fig_dir: Path):
    """
    地下水方案对比：同期、最优单滞后、较优多滞后
    """
    selected_models = [
        "sync_gw_direct_cmp6",
        "lag5_gw_direct_cmp6",
        "multi_lag_gw_5_6_direct_cmp6",
        "multi_lag_gw_2_3_5_direct_cmp6",
    ]
    df = df_overall[df_overall["model"].isin(selected_models)].copy()

    order_map = {name: i for i, name in enumerate(selected_models)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    plot_pretty_bar(
        df=df,
        metric="MAE",
        title="Groundwater Feature Scheme Comparison (MAE)",
        save_path=fig_dir / "gw_scheme_compare_MAE.png",
        figsize=(10, 5.5),
        rotate=0,
        width=0.52,
    )
    plot_pretty_bar(
        df=df,
        metric="RMSE",
        title="Groundwater Feature Scheme Comparison (RMSE)",
        save_path=fig_dir / "gw_scheme_compare_RMSE.png",
        figsize=(10, 5.5),
        rotate=0,
        width=0.52,
    )
    plot_pretty_bar(
        df=df,
        metric="R2",
        title="Groundwater Feature Scheme Comparison (R2)",
        save_path=fig_dir / "gw_scheme_compare_R2.png",
        figsize=(10, 5.5),
        rotate=0,
        width=0.52,
    )


def export_formal_tables(df_overall: pd.DataFrame, df_per_step: pd.DataFrame, summary_root: Path):
    formal_models = [
        "only_s_direct_cmp6",
        "sync_gw_direct_cmp6",
        "lag1_gw_direct_cmp6",
        "lag2_gw_direct_cmp6",
        "lag3_gw_direct_cmp6",
        "lag4_gw_direct_cmp6",
        "lag5_gw_direct_cmp6",
        "lag6_gw_direct_cmp6",
        "multi_lag_gw_3_5_direct_cmp6",
        "multi_lag_gw_5_6_direct_cmp6",
        "multi_lag_gw_2_3_5_direct_cmp6",
        "only_s_recursive_cmp6",
        "lag5_gw_recursive_cmp6",
    ]

    df_overall_formal = df_overall[df_overall["model"].isin(formal_models)].copy()
    df_per_step_formal = df_per_step[df_per_step["model"].isin(formal_models)].copy()

    order_map = {name: i for i, name in enumerate(formal_models)}
    df_overall_formal["order"] = df_overall_formal["model"].map(order_map)
    df_per_step_formal["order"] = df_per_step_formal["model"].map(order_map)

    df_overall_formal = df_overall_formal.sort_values("order").drop(columns="order")
    df_per_step_formal = df_per_step_formal.sort_values("order").drop(columns="order")

    overall_path = summary_root / "summary_overall_formal.csv"
    per_step_path = summary_root / "summary_per_step_formal.csv"

    df_overall_formal.to_csv(overall_path, index=False, encoding="utf-8-sig")
    df_per_step_formal.to_csv(per_step_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {overall_path}")
    print(f"[SAVE] {per_step_path}")


def main():
    df_overall, df_per_step, summary_root = load_data(
        r"F:\aaa1\建模预测分析\results_summary"
    )

    fig_dir = summary_root / "figures"
    ensure_dir(fig_dir)

    export_formal_tables(df_overall, df_per_step, summary_root)
    plot_overall_bars(df_overall, fig_dir)
    plot_direct_lag_curve(df_overall, fig_dir)
    plot_core_strategy_compare(df_overall, fig_dir)
    plot_step_curves(df_per_step, fig_dir)
    plot_multi_lag_compare(df_overall, fig_dir)
    plot_gw_scheme_compare(df_overall, fig_dir)

    print("=" * 80)
    print("[INFO] 全部图件已生成")
    print(f"[INFO] 图件目录: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    