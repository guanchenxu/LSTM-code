# -*- coding: utf-8 -*-
"""
stage5_select_representative_points.py

功能：
1. 读取 only_s_direct_cmp6 的 test_predictions.npz
2. 按 point_id 聚合测试样本
3. 计算每个点的真实值统计特征
4. 自动筛选三类代表点：
   - 沉降明显
   - 平稳
   - 抬升明显
5. 保存统计表与最终代表点表

筛选逻辑：
- 沉降明显：测试期平均真实值较小（更负）
- 抬升明显：测试期平均真实值较大（更正）
- 平稳：平均值接近0，且波动较小

作者：ChatGPT
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_npz(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def flatten_true_series_per_sample(y_true: np.ndarray) -> np.ndarray:
    """
    y_true: [N, horizon]
    返回每个样本的平均真实值
    """
    return y_true.mean(axis=1)


def build_point_statistics(pred_npz_path: str) -> pd.DataFrame:
    """
    基于 test_predictions.npz 构建每个点的统计特征
    """
    data = load_npz(pred_npz_path)

    y_true = data["y_true"].astype(np.float32)          # [N, horizon]
    point_id = data["point_id"]
    lon = data["lon"].astype(np.float32)
    lat = data["lat"].astype(np.float32)

    sample_true_mean = flatten_true_series_per_sample(y_true)   # [N]

    # 每个样本的真实值波动
    sample_true_std = y_true.std(axis=1)
    sample_true_min = y_true.min(axis=1)
    sample_true_max = y_true.max(axis=1)

    df = pd.DataFrame({
        "point_id": point_id,
        "lon": lon,
        "lat": lat,
        "sample_true_mean": sample_true_mean,
        "sample_true_std": sample_true_std,
        "sample_true_min": sample_true_min,
        "sample_true_max": sample_true_max,
    })

    # 对每个点聚合
    grouped = df.groupby("point_id", dropna=False)

    point_stats = grouped.agg(
        lon=("lon", "first"),
        lat=("lat", "first"),
        n_windows=("sample_true_mean", "size"),
        mean_true=("sample_true_mean", "mean"),
        std_true=("sample_true_mean", "std"),
        mean_inner_std=("sample_true_std", "mean"),
        min_true=("sample_true_min", "min"),
        max_true=("sample_true_max", "max"),
    ).reset_index()

    point_stats["std_true"] = point_stats["std_true"].fillna(0.0)

    # 幅度指标
    point_stats["range_true"] = point_stats["max_true"] - point_stats["min_true"]

    # 接近0程度（平稳性判断辅助）
    point_stats["abs_mean_true"] = point_stats["mean_true"].abs()

    return point_stats


def select_subsidence_point(point_stats: pd.DataFrame) -> pd.Series:
    """
    沉降明显点：
    优先选择 mean_true 最小（更负）的点，
    如有并列，优先 range_true 较大者。
    """
    df = point_stats.sort_values(
        by=["mean_true", "range_true"],
        ascending=[True, False]
    ).reset_index(drop=True)
    row = df.iloc[0].copy()
    row["category"] = "沉降明显"
    return row


def select_uplift_point(point_stats: pd.DataFrame) -> pd.Series:
    """
    抬升明显点：
    优先选择 mean_true 最大（更正）的点，
    如有并列，优先 range_true 较大者。
    """
    df = point_stats.sort_values(
        by=["mean_true", "range_true"],
        ascending=[False, False]
    ).reset_index(drop=True)
    row = df.iloc[0].copy()
    row["category"] = "抬升明显"
    return row


def select_stable_point(point_stats: pd.DataFrame) -> pd.Series:
    """
    平稳点：
    选择 mean_true 接近0 且波动较小的点
    """
    df = point_stats.copy()

    # 先筛掉波动特别大的点，保留更稳定的点
    std_q50 = df["std_true"].quantile(0.50)
    range_q50 = df["range_true"].quantile(0.50)

    df_stable = df[
        (df["std_true"] <= std_q50) &
        (df["range_true"] <= range_q50)
    ].copy()

    # 如果筛得太严导致为空，则退回全量
    if len(df_stable) == 0:
        df_stable = df.copy()

    # 优先 mean_true 接近0，其次 std_true 小，再次 range_true 小
    df_stable = df_stable.sort_values(
        by=["abs_mean_true", "std_true", "range_true"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    row = df_stable.iloc[0].copy()
    row["category"] = "平稳"
    return row


def save_outputs(point_stats: pd.DataFrame, selected_points: pd.DataFrame, save_root: str):
    save_root = Path(save_root)
    ensure_dir(save_root)

    stats_path = save_root / "point_statistics.csv"
    selected_path = save_root / "selected_points.csv"

    point_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    selected_points.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[INFO] 代表点筛选完成")
    print(f"[SAVE] 全部点统计表: {stats_path}")
    print(f"[SAVE] 代表点表: {selected_path}")
    print("=" * 80)


def main():
    pred_npz_path = r"F:\aaa1\建模预测分析\stage2_outputs\only_s_direct_cmp6\test_predictions.npz"
    save_root = r"F:\aaa1\建模预测分析\results_summary\representative_points"

    print("[INFO] 正在读取测试集预测结果并统计点位特征 ...")
    point_stats = build_point_statistics(pred_npz_path)

    print(f"[INFO] 共统计点数: {len(point_stats)}")

    subsidence_point = select_subsidence_point(point_stats)
    stable_point = select_stable_point(point_stats)
    uplift_point = select_uplift_point(point_stats)

    selected_points = pd.DataFrame([
        subsidence_point,
        stable_point,
        uplift_point,
    ])

    # 调整列顺序
    preferred_cols = [
        "category",
        "point_id",
        "lon",
        "lat",
        "n_windows",
        "mean_true",
        "std_true",
        "mean_inner_std",
        "min_true",
        "max_true",
        "range_true",
        "abs_mean_true",
    ]
    selected_points = selected_points[preferred_cols]

    print("\n[INFO] 选中的代表点：")
    print(selected_points)

    save_outputs(point_stats, selected_points, save_root)


if __name__ == "__main__":
    main()