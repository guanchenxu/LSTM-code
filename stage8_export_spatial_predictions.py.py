# -*- coding: utf-8 -*-
"""
stage8_export_spatial_predictions.py

作用：
1. 读取全区最优模型 test_predictions.npz
2. 将重叠窗口中的多步预测整理为“按点-按月”的真实值/预测值/误差表
3. 导出：
   - 长表 long format
   - 宽表 wide format
   - 指定月份单独CSV（适合QGIS）

默认读取：
only_s_direct_cmp6_full_tuned 的预测结果

作者：ChatGPT
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# 需要根据你的实验设置确认的参数
# =========================
NPZ_PATH = r"F:\aaa1\建模预测分析\stage2_outputs_full\only_s_direct_cmp6_full_tuned\test_predictions.npz"
CSV_PATH = r"F:\aaa1\建模预测分析\数据.csv"
OUTPUT_DIR = r"F:\aaa1\建模预测分析\results_summary\spatial_predictions"

LOOKBACK = 12
HORIZON = 5

# 推荐导出的代表月份（适合QGIS展示）
SELECT_MONTHS = ["202203", "202207", "202211"]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_point_id(x):
    """
    统一 point_id 表达，避免 17668 / 17668.0 / '17668' 不一致
    """
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf)
    except Exception:
        return str(x).strip()


def load_npz(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def get_month_labels_from_csv(csv_path: str) -> list[str]:
    """
    从原始CSV中读取 S_YYYYMM 列名，提取月份标签
    """
    df_head = pd.read_csv(csv_path, nrows=1, encoding="utf-8-sig")
    s_cols = [c for c in df_head.columns if c.startswith("S_")]
    month_labels = [c.replace("S_", "") for c in s_cols]
    return month_labels


def build_long_table(npz_path: str, csv_path: str, lookback: int, horizon: int) -> pd.DataFrame:
    data = load_npz(npz_path)
    month_labels = get_month_labels_from_csv(csv_path)

    required_keys = ["point_id", "lon", "lat", "y_true", "y_pred", "start_idx"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"test_predictions.npz 中缺少字段: {k}")

    point_ids = data["point_id"]
    lon = data["lon"].astype(np.float64)
    lat = data["lat"].astype(np.float64)
    y_true = data["y_true"].astype(np.float64)   # [N, H]
    y_pred = data["y_pred"].astype(np.float64)   # [N, H]
    start_idx = data["start_idx"].astype(np.int32)

    rows = []

    n_samples = len(point_ids)
    for i in range(n_samples):
        pid = normalize_point_id(point_ids[i])
        x = lon[i]
        y = lat[i]
        s = int(start_idx[i])

        for h in range(horizon):
            month_idx = s + lookback + h
            if month_idx < 0 or month_idx >= len(month_labels):
                continue

            month = month_labels[month_idx]
            true_val = float(y_true[i, h])
            pred_val = float(y_pred[i, h])
            err_val = pred_val - true_val

            rows.append({
                "point_id": pid,
                "lon": x,
                "lat": y,
                "month": month,
                "true": true_val,
                "pred": pred_val,
                "error": err_val,
                "sample_idx": i,
                "start_idx": s,
                "step": h + 1,
            })

    df_long_raw = pd.DataFrame(rows)

    # 对同一点、同一月份，多个窗口的预测取平均
    df_long = (
        df_long_raw
        .groupby(["point_id", "lon", "lat", "month"], as_index=False)
        .agg(
            true=("true", "mean"),
            pred=("pred", "mean"),
            error=("error", "mean"),
            n_contrib=("sample_idx", "size"),
        )
        .sort_values(["month", "point_id"])
        .reset_index(drop=True)
    )

    return df_long


def build_wide_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    将长表转换为宽表：
    point_id / lon / lat / true_202203 / pred_202203 / error_202203 ...
    """
    id_cols = ["point_id", "lon", "lat"]
    months = sorted(df_long["month"].unique())

    base = df_long[id_cols].drop_duplicates().copy()

    for m in months:
        df_m = df_long[df_long["month"] == m][["point_id", "lon", "lat", "true", "pred", "error"]].copy()
        df_m = df_m.rename(columns={
            "true": f"true_{m}",
            "pred": f"pred_{m}",
            "error": f"error_{m}",
        })
        base = base.merge(df_m, on=id_cols, how="left")

    return base


def export_selected_months(df_long: pd.DataFrame, output_dir: Path, select_months: list[str]):
    for m in select_months:
        df_m = df_long[df_long["month"] == m].copy()
        if df_m.empty:
            print(f"[WARN] 月份 {m} 未找到，跳过导出。")
            continue

        out_path = output_dir / f"pred_map_{m}.csv"
        df_m.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 单月QGIS制图表: {out_path}")


def main():
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    print("[INFO] 正在读取预测结果并重建全区按月真实值-预测值表 ...")
    df_long = build_long_table(
        npz_path=NPZ_PATH,
        csv_path=CSV_PATH,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    df_wide = build_wide_table(df_long)

    long_path = output_dir / "all_points_predictions_long.csv"
    wide_path = output_dir / "all_points_predictions_wide.csv"

    df_long.to_csv(long_path, index=False, encoding="utf-8-sig")
    df_wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] 长表: {long_path}")
    print(f"[SAVE] 宽表: {wide_path}")

    export_selected_months(df_long, output_dir, SELECT_MONTHS)

    print("=" * 80)
    print("[INFO] 导出完成")
    print(f"[INFO] 输出目录: {output_dir}")
    print("=" * 80)

    print("\n[INFO] 长表预览:")
    print(df_long.head())

    print("\n[INFO] 月份范围:")
    print(sorted(df_long['month'].unique()))


if __name__ == "__main__":
    main()