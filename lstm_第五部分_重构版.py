import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ===================== 1. 可视化配置 =====================
@dataclass
class VisualConfig:
    stage3_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage3_output"
    stage4_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage4_output"

    dpi: int = 300
    figsize_wide: Tuple[float, float] = (10.0, 6.0)
    figsize_square: Tuple[float, float] = (7.0, 7.0)
    figsize_point: Tuple[float, float] = (10.0, 5.5)

    uplift_threshold: float = 2.0
    subsidence_threshold: float = -2.0
    max_points_per_type: int = 3  # 每类典型点可视化数量
    save_svg: bool = False
    save_pdf: bool = False
    show_plots: bool = False

    def __post_init__(self) -> None:
        self.train_log_path = os.path.join(self.stage3_dir, "train_log.csv")
        self.train_summary_path = os.path.join(self.stage3_dir, "train_summary.json")

        self.overall_metrics_path = os.path.join(self.stage4_dir, "overall_metrics.csv")
        self.step_metrics_path = os.path.join(self.stage4_dir, "step_metrics.csv")
        self.pred_detail_path = os.path.join(self.stage4_dir, "test_predictions_detail.csv")
        self.test_summary_path = os.path.join(self.stage4_dir, "test_summary.json")
        self.test_config_path = os.path.join(self.stage4_dir, "test_config.json")
        self.y_pred_raw_path = os.path.join(self.stage4_dir, "y_pred_test_raw.npy")
        self.y_true_raw_path = os.path.join(self.stage4_dir, "y_true_test_raw.npy")

        for p in [
            self.train_log_path,
            self.overall_metrics_path,
            self.step_metrics_path,
            self.pred_detail_path,
            self.y_pred_raw_path,
            self.y_true_raw_path,
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少必要输入文件：{p}")

        self.stage5_dir = os.path.join(os.path.dirname(self.stage4_dir), "stage5_output")
        os.makedirs(self.stage5_dir, exist_ok=True)

        self.fig_dir = os.path.join(self.stage5_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)


# ===================== 2. 数据读取 =====================
def load_inputs(config: VisualConfig) -> Dict[str, object]:
    data: Dict[str, object] = {}

    data["train_log"] = pd.read_csv(config.train_log_path)
    data["overall_metrics"] = pd.read_csv(config.overall_metrics_path)
    data["step_metrics"] = pd.read_csv(config.step_metrics_path)
    data["pred_detail"] = pd.read_csv(config.pred_detail_path)
    data["y_pred_raw"] = np.load(config.y_pred_raw_path).astype(np.float32)
    data["y_true_raw"] = np.load(config.y_true_raw_path).astype(np.float32)

    if os.path.exists(config.train_summary_path):
        with open(config.train_summary_path, "r", encoding="utf-8") as f:
            data["train_summary"] = json.load(f)
    else:
        data["train_summary"] = {}

    if os.path.exists(config.test_summary_path):
        with open(config.test_summary_path, "r", encoding="utf-8") as f:
            data["test_summary"] = json.load(f)
    else:
        data["test_summary"] = {}

    if os.path.exists(config.test_config_path):
        with open(config.test_config_path, "r", encoding="utf-8") as f:
            data["test_config"] = json.load(f)
    else:
        data["test_config"] = {}

    # 安全获取 future_time_cols
    future_time_cols = data["test_summary"].get("future_time_cols", [])
    if not future_time_cols:
        pred_cols = [c for c in data["pred_detail"].columns if c.startswith("true_")]
        future_time_cols = [c.replace("true_", "") for c in pred_cols if c.startswith("true_")]
    data["future_time_cols"] = future_time_cols

    print("✅ 第五阶段输入读取完成")
    print(f"   train_log shape: {data['train_log'].shape}")
    print(f"   step_metrics shape: {data['step_metrics'].shape}")
    print(f"   pred_detail shape: {data['pred_detail'].shape}")
    print(f"   y_true_raw shape: {data['y_true_raw'].shape}")
    print(f"   y_pred_raw shape: {data['y_pred_raw'].shape}")
    return data


# ===================== 3. 工具函数 =====================
def _save_figure(fig, save_path_no_ext: str, config: VisualConfig) -> None:
    fig.tight_layout()
    fig.savefig(f"{save_path_no_ext}.png", dpi=config.dpi, bbox_inches="tight")
    if config.save_svg:
        fig.savefig(f"{save_path_no_ext}.svg", bbox_inches="tight")
    if config.save_pdf:
        fig.savefig(f"{save_path_no_ext}.pdf", bbox_inches="tight")
    if config.show_plots:
        plt.show()
    plt.close(fig)


# ===================== 4. 图件绘制 =====================
def plot_loss_curve(train_log: pd.DataFrame, config: VisualConfig) -> str:
    fig, ax = plt.subplots(figsize=config.figsize_wide)
    ax.plot(train_log["epoch"], train_log["train_loss"], label="Train Loss")
    ax.plot(train_log["epoch"], train_log["val_loss"], label="Val Loss")
    best_idx = train_log["val_loss"].idxmin()
    best_epoch = int(train_log.loc[best_idx, "epoch"])
    best_val = float(train_log.loc[best_idx, "val_loss"])
    ax.scatter([best_epoch], [best_val], label=f"Best Epoch={best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(config.fig_dir, "fig5_4_loss_curve")
    _save_figure(fig, save_path, config)
    return save_path + ".png"


def plot_step_metrics(step_metrics: pd.DataFrame, config: VisualConfig) -> str:
    x = step_metrics["step"].to_numpy()
    width = 0.38
    fig, ax = plt.subplots(figsize=config.figsize_wide)
    ax.bar(x - width/2, step_metrics["MAE"].to_numpy(), width=width, label="MAE")
    ax.bar(x + width/2, step_metrics["RMSE"].to_numpy(), width=width, label="RMSE")
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("Error")
    ax.set_title("MAE and RMSE by Prediction Step")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_path = os.path.join(config.fig_dir, "fig5_5_step_metrics")
    _save_figure(fig, save_path, config)
    return save_path + ".png"


def _extract_series_from_row(row: pd.Series, prefix: str, future_time_cols: List[str]) -> np.ndarray:
    values = []
    for c in future_time_cols:
        col = f"{prefix}_{c}"
        values.append(float(row[col]) if col in row.index else 0.0)
    return np.asarray(values, dtype=np.float32)


def select_typical_points(pred_detail: pd.DataFrame, future_time_cols: List[str], config: VisualConfig) -> pd.DataFrame:
    df = pred_detail.copy()
    true_cols = [f"true_{c}" for c in future_time_cols]
    df["mean_true_future"] = df[true_cols].mean(axis=1)
    # 分类
    uplift_df = df[df["mean_true_future"] >= config.uplift_threshold].sort_values("mean_true_future", ascending=False).head(config.max_points_per_type)
    subs_df = df[df["mean_true_future"] <= config.subsidence_threshold].sort_values("mean_true_future", ascending=True).head(config.max_points_per_type)
    transition_df = df[(df["mean_true_future"] > config.subsidence_threshold) & (df["mean_true_future"] < config.uplift_threshold)].sort_values("mean_true_future", ascending=True).head(config.max_points_per_type)
    chosen = pd.concat([uplift_df, subs_df, transition_df], ignore_index=True)
    chosen.to_csv(os.path.join(config.stage5_dir, "selected_typical_points.csv"), index=False, encoding="utf-8-sig")
    return chosen


def plot_typical_point(row: pd.Series, point_type: str, future_time_cols: List[str], config: VisualConfig) -> str:
    y_true = _extract_series_from_row(row, "true", future_time_cols)
    y_pred = _extract_series_from_row(row, "pred", future_time_cols)
    x = np.arange(1, len(future_time_cols) + 1)
    fig, ax = plt.subplots(figsize=config.figsize_point)
    ax.plot(x, y_true, marker="o", label="True")
    ax.plot(x, y_pred, marker="s", label="Pred")
    ax.set_xlabel("Future Month Step")
    ax.set_ylabel("Settlement")
    ax.set_title(f"Typical {point_type.capitalize()} Point Prediction")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    meta_parts = []
    for key in ["ID", "Lon", "Lat"]:
        if key in row.index:
            meta_parts.append(f"{key}={row[key]}")
    if meta_parts:
        ax.text(0.01, 0.99, "\n".join(meta_parts), transform=ax.transAxes, va="top", ha="left")
    save_path = os.path.join(config.fig_dir, f"typical_{point_type}_{row.name}")
    _save_figure(fig, save_path, config)
    return save_path + ".png"


def plot_scatter(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, config: VisualConfig) -> str:
    yt, yp = y_true_raw.reshape(-1), y_pred_raw.reshape(-1)
    fig, ax = plt.subplots(figsize=config.figsize_square)
    ax.scatter(yt, yp, alpha=0.45, s=12)
    min_v, max_v = float(min(np.min(yt), np.min(yp))), float(max(np.max(yt), np.max(yp)))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_xlabel("True Settlement")
    ax.set_ylabel("Predicted Settlement")
    ax.set_title("Predicted vs True Settlement")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(config.fig_dir, "fig5_9_scatter_true_vs_pred")
    _save_figure(fig, save_path, config)
    return save_path + ".png"


def plot_error_distribution(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, config: VisualConfig) -> str:
    err = (y_pred_raw - y_true_raw).reshape(-1)
    fig, ax = plt.subplots(figsize=config.figsize_wide)
    ax.hist(err, bins=40, density=True, alpha=0.8)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.grid(True, alpha=0.3)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(err)
        xs = np.linspace(err.min(), err.max(), 300)
        ax.plot(xs, kde(xs), linewidth=2)
    except Exception:
        pass
    save_path = os.path.join(config.fig_dir, "fig5_10_error_distribution")
    _save_figure(fig, save_path, config)
    return save_path + ".png"


def plot_error_boxplot(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, config: VisualConfig) -> str:
    err = (y_pred_raw - y_true_raw).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.boxplot(err, vert=True, showfliers=True)
    ax.set_ylabel("Prediction Error")
    ax.set_title("Error Boxplot")
    ax.grid(True, axis="y", alpha=0.3)
    save_path = os.path.join(config.fig_dir, "fig5_11_error_boxplot")
    _save_figure(fig, save_path, config)
    return save_path + ".png"


# ===================== 5. 汇总输出 =====================
def build_visual_summary(config: VisualConfig, data: Dict[str, object], generated_figs: Dict[str, str], typical_points_df: pd.DataFrame) -> Dict[str, object]:
    overall_metrics = {}
    if isinstance(data["overall_metrics"], pd.DataFrame) and not data["overall_metrics"].empty:
        overall_metrics = data["overall_metrics"].iloc[0].to_dict()
    summary = {
        "stage5_dir": config.stage5_dir,
        "fig_dir": config.fig_dir,
        "pred_mode": data.get("test_summary", {}).get("pred_mode", None),
        "feature_mode": data.get("test_summary", {}).get("feature_mode", None),
        "horizon": int(data["y_true_raw"].shape[1]),
        "num_test_samples": int(data["y_true_raw"].shape[0]),
        "overall_metrics": overall_metrics,
        "generated_figures": generated_figs,
        "num_typical_points": int(len(typical_points_df)),
    }
    return summary


# ===================== 6. 主流程 =====================
def main() -> None:
    config = VisualConfig()
    data = load_inputs(config)

    # 输出整体指标
    if "overall_metrics" in data and not data["overall_metrics"].empty:
        print("\n🔹 测试集整体指标:")
        for k, v in data["overall_metrics"].iloc[0].items():
            print(f"   {k}: {v}")

    generated_figs: Dict[str, str] = {}
    generated_figs["loss_curve"] = plot_loss_curve(data["train_log"], config)
    generated_figs["step_metrics"] = plot_step_metrics(data["step_metrics"], config)

    typical_points_df = select_typical_points(data["pred_detail"], data["future_time_cols"], config)
    if not typical_points_df.empty:
        for _, row in typical_points_df.iterrows():
            generated_figs[f"typical_{row['point_type']}_{row.name}"] = plot_typical_point(row, str(row["point_type"]), data["future_time_cols"], config)

    generated_figs["scatter_true_vs_pred"] = plot_scatter(data["y_true_raw"], data["y_pred_raw"], config)
    generated_figs["error_distribution"] = plot_error_distribution(data["y_true_raw"], data["y_pred_raw"], config)
    generated_figs["error_boxplot"] = plot_error_boxplot(data["y_true_raw"], data["y_pred_raw"], config)

    summary = build_visual_summary(config, data, generated_figs, typical_points_df)
    with open(os.path.join(config.stage5_dir, "visual_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    with open(os.path.join(config.stage5_dir, "visual_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4, ensure_ascii=False)

    print("\n✅ 第五阶段可视化完成")
    print(f"📁 输出目录：{config.stage5_dir}")
    print("生成图件：")
    for k, v in generated_figs.items():
        print(f"- {k}: {v}")
    print("附加文件：")
    print("- selected_typical_points.csv: 典型点信息")
    print("- visual_summary.json: 可视化结果摘要")
    print("- visual_config.json: 可视化配置")


if __name__ == "__main__":
    main()