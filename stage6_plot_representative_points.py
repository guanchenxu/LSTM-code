# -*- coding: utf-8 -*-
"""
stage6_plot_representative_points.py

功能：
1. 读取代表点 selected_points.csv
2. 读取多个模型的 test_predictions.npz
3. 对每个代表点提取对应测试样本
4. 画真实值-预测值对比曲线
5. 保存图片

说明：
- 每个点生成 1 张图
- 每张图包含真实值和多个模型预测值
- 横轴使用测试窗口序号（1~9）
- 纵轴使用该点每个测试窗口未来5步的平均真实/预测值

作者：ChatGPT
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ===== Matplotlib 中文显示设置 =====
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_npz(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def shorten_model_name(name: str) -> str:
    mapping = {
        "only_s_direct_cmp6": "Only-S Direct",
        "lag5_gw_direct_cmp6": "Lag5-GW Direct",
        "only_s_recursive_cmp6": "Only-S Rec",
        "multi_lag_gw_2_3_5_direct_cmp6": "Multi(2,3,5)",
    }
    return mapping.get(name, name)

def normalize_point_id(x):
    """
    统一 point_id 的比较格式：
    - 17668
    - 17668.0
    - "17668"
    最终都转成 "17668"
    """
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf)
    except Exception:
        return str(x).strip()
    
def format_point_id_for_display(x):
    """
    用于标题显示：
    17668.0 -> 17668
    """
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf)
    except Exception:
        return str(x).strip()
    
def build_series_for_model(pred_npz_path: str, point_id_value) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从某个模型的 test_predictions.npz 中提取指定 point_id 的序列

    返回：
    - x: 测试窗口序号
    - y_true_mean: 每个测试窗口未来5步真实值均值
    - y_pred_mean: 每个测试窗口未来5步预测值均值
    """
    data = load_npz(pred_npz_path)

    point_ids = data["point_id"]
    y_true = data["y_true"].astype(np.float32)   # [N, 5]
    y_pred = data["y_pred"].astype(np.float32)   # [N, 5]
    start_idx = data["start_idx"].astype(np.int32)

    # point_id 可能是数字/字符串，统一转字符串比较更稳
    target_id = normalize_point_id(point_id_value)
    mask = np.array([normalize_point_id(x) == target_id for x in point_ids])

    if mask.sum() == 0:
        raise ValueError(f"在 {pred_npz_path} 中未找到 point_id={point_id_value}")

    y_true_pt = y_true[mask]
    y_pred_pt = y_pred[mask]
    start_idx_pt = start_idx[mask]

    # 按时间顺序排序
    order = np.argsort(start_idx_pt)
    y_true_pt = y_true_pt[order]
    y_pred_pt = y_pred_pt[order]

    # 这里将每个测试样本未来5步取均值，得到更清晰的一条曲线
    y_true_mean = y_true_pt.mean(axis=1)
    y_pred_mean = y_pred_pt.mean(axis=1)

    x = np.arange(1, len(y_true_mean) + 1)

    return x, y_true_mean, y_pred_mean


def plot_one_point(selected_row: pd.Series, model_paths: dict[str, str], save_dir: Path):
    point_id_value = selected_row["point_id"]
    point_id_show = format_point_id_for_display(point_id_value)
    category = str(selected_row["category"])
    lon = selected_row["lon"]
    lat = selected_row["lat"]

    fig = plt.figure(figsize=(10, 5.5))

    true_plotted = False

    for model_key, pred_npz_path in model_paths.items():
        x, y_true_mean, y_pred_mean = build_series_for_model(pred_npz_path, point_id_value)

        if not true_plotted:
            plt.plot(x, y_true_mean, marker="o", linewidth=2, label="True")
            true_plotted = True

        plt.plot(
            x,
            y_pred_mean,
            marker="o",
            linewidth=1.8,
            label=shorten_model_name(model_key)
        )

    plt.xlabel("Test Window Index")
    plt.ylabel("Mean Deformation of 5-step Horizon")
    plt.title(
    f"{category}点 (ID={point_id_show}, Lon={lon:.6f}, Lat={lat:.6f})"
)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / f"{category}_point_{point_id_value}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")


def main():
    selected_points_csv = r"F:\aaa1\建模预测分析\results_summary\representative_points\selected_points.csv"
    save_dir = Path(r"F:\aaa1\建模预测分析\results_summary\representative_points\figures")
    ensure_dir(save_dir)

    selected_df = pd.read_csv(selected_points_csv, encoding="utf-8-sig")

    # 需要比较的模型
    model_paths = {
        "only_s_direct_cmp6": r"F:\aaa1\建模预测分析\stage2_outputs\only_s_direct_cmp6\test_predictions.npz",
        "lag5_gw_direct_cmp6": r"F:\aaa1\建模预测分析\stage2_outputs\lag5_gw_direct_cmp6\test_predictions.npz",
        "only_s_recursive_cmp6": r"F:\aaa1\建模预测分析\stage2_outputs\only_s_recursive_cmp6\test_predictions.npz",
        "multi_lag_gw_2_3_5_direct_cmp6": r"F:\aaa1\建模预测分析\stage2_outputs\multi_lag_gw_2_3_5_direct_cmp6\test_predictions.npz",
    }

    print("[INFO] 开始绘制代表点真实值-预测值对比图 ...")

    for _, row in selected_df.iterrows():
        plot_one_point(row, model_paths, save_dir)

    print("=" * 80)
    print("[INFO] 全部代表点图件已生成")
    print(f"[INFO] 图件目录: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()