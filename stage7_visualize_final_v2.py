# stage7_visualize_final_v2.py
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 中文字体设置
# =========================
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

# =========================
# 路径配置
# =========================
RESULTS_DIR = Path(r"F:\aaa1\lstm建模预测滞后1期\stage7_results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MODES = ["sync_gw", "lag1_gw", "lag2_gw", "lag3_gw", "multi_lag_gw"]
PRED_MODES = ["direct", "rolling"]
METRICS = ["MAE", "RMSE", "R2", "MAPE"]

# 用于显示的中文标签
FEATURE_LABELS = {
    "sync_gw": "同期地下水",
    "lag1_gw": "滞后1期",
    "lag2_gw": "滞后2期",
    "lag3_gw": "滞后3期",
    "multi_lag_gw": "多滞后耦合",
}

PRED_LABELS = {
    "direct": "直接多步预测",
    "rolling": "递归滚动预测",
}


# =========================
# 工具函数
# =========================
def load_summary() -> pd.DataFrame:
    summary_file = RESULTS_DIR / "experiment_summary_all.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"未找到汇总文件：{summary_file}")
    df = pd.read_csv(summary_file)

    required_cols = {"feature_mode", "pred_mode", "MAE", "RMSE", "R2", "MAPE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"汇总表缺少字段：{missing}")

    for col in ["MAE", "RMSE", "R2", "MAPE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_metric_values(summary_df: pd.DataFrame, metric: str, pred_mode: str):
    vals = []
    for fm in FEATURE_MODES:
        row = summary_df[
            (summary_df["feature_mode"] == fm) &
            (summary_df["pred_mode"] == pred_mode)
        ]
        if row.empty:
            vals.append(np.nan)
        else:
            vals.append(float(row[metric].iloc[0]))
    return vals


def save_fig(fig, filename: str):
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 图1：整体指标对比（折线图）
# =========================
def plot_overall_metrics_line(summary_df: pd.DataFrame):
    x_labels = [FEATURE_LABELS[x] for x in FEATURE_MODES]

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))

        for pred_mode in PRED_MODES:
            vals = get_metric_values(summary_df, metric, pred_mode)
            ax.plot(
                x_labels,
                vals,
                marker="o",
                linewidth=2,
                markersize=7,
                label=PRED_LABELS[pred_mode]
            )

            # 标注数值
            for i, v in enumerate(vals):
                if pd.notna(v):
                    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(f"{metric} 对比：不同滞后方案 × 预测模式")
        ax.set_xlabel("输入方案")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

        save_fig(fig, f"overall_{metric}_line.png")


# =========================
# 图2：整体指标对比（分组柱状图）
# =========================
def plot_overall_metrics_bar(summary_df: pd.DataFrame):
    x = np.arange(len(FEATURE_MODES))
    width = 0.36
    x_labels = [FEATURE_LABELS[x] for x in FEATURE_MODES]

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))

        vals_direct = get_metric_values(summary_df, metric, "direct")
        vals_rolling = get_metric_values(summary_df, metric, "rolling")

        bars1 = ax.bar(x - width / 2, vals_direct, width, label=PRED_LABELS["direct"])
        bars2 = ax.bar(x + width / 2, vals_rolling, width, label=PRED_LABELS["rolling"])

        # 柱顶标注
        for bars in [bars1, bars2]:
            for b in bars:
                h = b.get_height()
                if pd.notna(h):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

        ax.set_title(f"{metric} 对比：不同滞后方案 × 预测模式")
        ax.set_xlabel("输入方案")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        save_fig(fig, f"overall_{metric}_bar.png")


# =========================
# 图3：逐步误差对比（每个方案单独一张）
# =========================
def plot_step_metrics_each():
    for fm in FEATURE_MODES:
        for pred_mode in PRED_MODES:
            step_file = RESULTS_DIR / f"{fm}_{pred_mode}" / "step_metrics.csv"
            if not step_file.exists():
                print(f"跳过：未找到 {step_file}")
                continue

            step_df = pd.read_csv(step_file)
            required_cols = {"step", "MAE", "RMSE"}
            if not required_cols.issubset(step_df.columns):
                print(f"跳过：{step_file} 缺少必要列")
                continue

            step_df["step"] = pd.to_numeric(step_df["step"], errors="coerce")
            step_df["MAE"] = pd.to_numeric(step_df["MAE"], errors="coerce")
            step_df["RMSE"] = pd.to_numeric(step_df["RMSE"], errors="coerce")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(step_df["step"], step_df["MAE"], marker="o", linewidth=2, label="MAE")
            ax.plot(step_df["step"], step_df["RMSE"], marker="s", linewidth=2, label="RMSE")

            ax.set_title(f"{FEATURE_LABELS[fm]} + {PRED_LABELS[pred_mode]}：逐步误差变化")
            ax.set_xlabel("预测步长（月）")
            ax.set_ylabel("误差")
            ax.set_xticks(step_df["step"])
            ax.grid(True, alpha=0.3)
            ax.legend()

            save_fig(fig, f"step_{fm}_{pred_mode}.png")


# =========================
# 图4：同一种预测模式下，不同滞后方案的逐步 MAE 对比
# =========================
def plot_step_mae_compare_by_mode():
    for pred_mode in PRED_MODES:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        for fm in FEATURE_MODES:
            step_file = RESULTS_DIR / f"{fm}_{pred_mode}" / "step_metrics.csv"
            if not step_file.exists():
                continue

            step_df = pd.read_csv(step_file)
            if not {"step", "MAE"}.issubset(step_df.columns):
                continue

            step_df["step"] = pd.to_numeric(step_df["step"], errors="coerce")
            step_df["MAE"] = pd.to_numeric(step_df["MAE"], errors="coerce")

            ax.plot(
                step_df["step"],
                step_df["MAE"],
                marker="o",
                linewidth=2,
                label=FEATURE_LABELS[fm]
            )
            plotted = True

        if plotted:
            ax.set_title(f"{PRED_LABELS[pred_mode]}：不同滞后方案下的逐步 MAE 对比")
            ax.set_xlabel("预测步长（月）")
            ax.set_ylabel("MAE")
            ax.grid(True, alpha=0.3)
            ax.legend()
            save_fig(fig, f"step_MAE_compare_{pred_mode}.png")
        else:
            plt.close(fig)


# =========================
# 图5：同一种预测模式下，不同滞后方案的逐步 RMSE 对比
# =========================
def plot_step_rmse_compare_by_mode():
    for pred_mode in PRED_MODES:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        for fm in FEATURE_MODES:
            step_file = RESULTS_DIR / f"{fm}_{pred_mode}" / "step_metrics.csv"
            if not step_file.exists():
                continue

            step_df = pd.read_csv(step_file)
            if not {"step", "RMSE"}.issubset(step_df.columns):
                continue

            step_df["step"] = pd.to_numeric(step_df["step"], errors="coerce")
            step_df["RMSE"] = pd.to_numeric(step_df["RMSE"], errors="coerce")

            ax.plot(
                step_df["step"],
                step_df["RMSE"],
                marker="s",
                linewidth=2,
                label=FEATURE_LABELS[fm]
            )
            plotted = True

        if plotted:
            ax.set_title(f"{PRED_LABELS[pred_mode]}：不同滞后方案下的逐步 RMSE 对比")
            ax.set_xlabel("预测步长（月）")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3)
            ax.legend()
            save_fig(fig, f"step_RMSE_compare_{pred_mode}.png")
        else:
            plt.close(fig)


# =========================
# 主程序
# =========================
def main():
    summary_df = load_summary()
    print("实验汇总读取完成，数据行数：", len(summary_df))
    print(summary_df)

    plot_overall_metrics_line(summary_df)
    print("整体指标折线图完成")

    plot_overall_metrics_bar(summary_df)
    print("整体指标柱状图完成")

    plot_step_metrics_each()
    print("单个方案逐步误差图完成")

    plot_step_mae_compare_by_mode()
    print("逐步 MAE 对比图完成")

    plot_step_rmse_compare_by_mode()
    print("逐步 RMSE 对比图完成")

    print(f"\n所有图件已保存至：{FIG_DIR}")


if __name__ == "__main__":
    main()