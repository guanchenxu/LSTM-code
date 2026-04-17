# -*- coding: utf-8 -*-
"""
stage1_build_dataset.py

功能：
1. 读取原始 CSV
2. 自动识别 S_ 和 CY_ 时间序列列
3. 检查月份列是否连续且一一对应
4. 构造 LSTM 监督学习样本
5. 按时间顺序划分 train / val / test
6. 保存为后续训练可直接读取的 npz 文件

支持的 feature_mode:
- only_s   : 仅使用地面变形序列
- sync_gw  : 使用同期地下水序列
- lag_gw   : 使用单一滞后地下水序列

默认参数：
- lookback = 12
- horizon = 5
- train/val/test = 0.7 / 0.1 / 0.2
- sample_points = 10000

作者：ChatGPT
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# =========================
# 配置数据类
# =========================
@dataclass
class BuildConfig:
    csv_path: str
    output_root: str
    feature_mode: str = "only_s"   # only_s / sync_gw / lag_gw
    lag: int = 1
    compare_max_lag: int | None = 6   # 新增：用于统一比较窗口
    lookback: int = 12
    horizon: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    sample_points: int | None = 10000
    random_seed: int = 42
    dtype: str = "float32"


# =========================
# 工具函数
# =========================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_safe(csv_path: str, usecols=None, nrows=None) -> pd.DataFrame:
    """
    尝试多种编码读取 CSV，避免中文路径或编码报错。
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, usecols=usecols, nrows=nrows, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV 文件：{csv_path}\n最后一次报错：{last_err}")


def parse_month_from_col(col_name: str, prefix: str) -> str:
    """
    例如：
    S_201801 -> 201801
    CY_201801 -> 201801
    """
    return col_name.replace(prefix, "", 1)


def generate_expected_months(start: str, end: str) -> List[str]:
    """
    根据起止月份生成连续月份序列，格式 YYYYMM
    """
    start_dt = pd.to_datetime(start, format="%Y%m")
    end_dt = pd.to_datetime(end, format="%Y%m")
    months = pd.date_range(start_dt, end_dt, freq="MS")
    return [d.strftime("%Y%m") for d in months]


def detect_time_series_columns(columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    自动识别 S_ 与 CY_ 开头的列，并按月份排序
    """
    s_cols = [c for c in columns if c.startswith("S_")]
    cy_cols = [c for c in columns if c.startswith("CY_")]

    if len(s_cols) == 0:
        raise ValueError("未找到任何 S_ 开头的地面变形列。")
    if len(cy_cols) == 0:
        raise ValueError("未找到任何 CY_ 开头的地下水列。")

    s_cols = sorted(s_cols, key=lambda x: parse_month_from_col(x, "S_"))
    cy_cols = sorted(cy_cols, key=lambda x: parse_month_from_col(x, "CY_"))

    return s_cols, cy_cols


def validate_month_columns(s_cols: List[str], cy_cols: List[str]) -> List[str]:
    """
    检查：
    1. S 与 CY 的月份数量是否一致
    2. S 与 CY 的月份是否逐一对应
    3. 月份是否连续
    """
    s_months = [parse_month_from_col(c, "S_") for c in s_cols]
    cy_months = [parse_month_from_col(c, "CY_") for c in cy_cols]

    if len(s_months) != len(cy_months):
        raise ValueError(
            f"S_ 列数量 ({len(s_months)}) 与 CY_ 列数量 ({len(cy_months)}) 不一致。"
        )

    if s_months != cy_months:
        raise ValueError("S_ 与 CY_ 的月份不一一对应，请检查列名。")

    expected = generate_expected_months(s_months[0], s_months[-1])
    if s_months != expected:
        raise ValueError(
            f"月份序列不连续。\n实际月份：{s_months[:5]} ... {s_months[-5:]}\n"
            f"期望月份：{expected[:5]} ... {expected[-5:]}"
        )

    return s_months


def select_points(df: pd.DataFrame, sample_points: int | None, random_seed: int) -> pd.DataFrame:
    """
    随机抽样部分点用于调试；若为 None，则使用全量数据。
    """
    n_total = len(df)
    if sample_points is None or sample_points >= n_total:
        print(f"[INFO] 使用全量样本点：{n_total}")
        return df.reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    idx = rng.choice(n_total, size=sample_points, replace=False)
    idx = np.sort(idx)
    sampled = df.iloc[idx].reset_index(drop=True)
    print(f"[INFO] 随机抽样样本点：{sample_points} / {n_total}")
    return sampled


def get_valid_start_indices(
    n_time: int,
    lookback: int,
    horizon: int,
    feature_mode: str,
    lag: int = 0,
    compare_max_lag: int | None = None,
) -> np.ndarray:
    """
    返回有效窗口起始索引 start 的数组。

    为了保证不同 feature_mode / lag 之间可公平比较，
    可以通过 compare_max_lag 统一所有模型的最小起点。

    例如 compare_max_lag=6 时：
    - only_s / sync_gw / lag1 / lag2 / ... / lag6
      都统一使用 start >= 6
    """

    max_start = n_time - lookback - horizon
    if max_start < 0:
        raise ValueError(
            f"时间序列长度不足：n_time={n_time}, lookback={lookback}, horizon={horizon}"
        )

    if compare_max_lag is not None:
        min_start = compare_max_lag
    else:
        if feature_mode == "lag_gw":
            min_start = lag
        else:
            min_start = 0

    if min_start > max_start:
        raise ValueError(
            f"没有可用窗口：feature_mode={feature_mode}, lag={lag}, "
            f"compare_max_lag={compare_max_lag}, "
            f"min_start={min_start}, max_start={max_start}"
        )

    return np.arange(min_start, max_start + 1, dtype=np.int32)


def split_start_indices(
    valid_starts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, np.ndarray]:
    """
    按时间顺序划分窗口起点。
    """
    total = len(valid_starts)
    if total < 3:
        raise ValueError(f"有效窗口数过少：{total}")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = total - train_n - val_n

    # 防止出现空 split
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError(
            f"划分后存在空集合：train={train_n}, val={val_n}, test={test_n}, total={total}"
        )

    train_starts = valid_starts[:train_n]
    val_starts = valid_starts[train_n:train_n + val_n]
    test_starts = valid_starts[train_n + val_n:]

    return {
        "train": train_starts,
        "val": val_starts,
        "test": test_starts,
    }


def build_arrays_for_split(
    s_data: np.ndarray,         # [n_points, n_time]
    gw_data: np.ndarray,        # [n_points, n_time]
    point_ids: np.ndarray,      # [n_points]
    lons: np.ndarray,           # [n_points]
    lats: np.ndarray,           # [n_points]
    months: List[str],
    split_starts: np.ndarray,   # [n_windows]
    cfg: BuildConfig,
) -> Dict[str, np.ndarray]:
    """
    构造某个 split 的 X / y / metadata
    输出：
    - X: [N, lookback, n_features]
    - y: [N, horizon]
    """
    n_points, n_time = s_data.shape
    lookback = cfg.lookback
    horizon = cfg.horizon
    feature_mode = cfg.feature_mode
    lag = cfg.lag

    # settlement 历史窗口视图: [n_points, n_time-lookback+1, lookback]
    s_hist_view = sliding_window_view(s_data, window_shape=lookback, axis=1)

    # target 窗口视图:
    # 对 start=s, target = S[s+lookback : s+lookback+horizon]
    # 先从 lookback 位置截断，再做 horizon 滑窗
    y_view = sliding_window_view(s_data[:, lookback:], window_shape=horizon, axis=1)

    # 对 groundwater 也构造历史窗口视图
    gw_hist_view = sliding_window_view(gw_data, window_shape=lookback, axis=1)

    # settlement history
    x_s = s_hist_view[:, split_starts, :]   # [n_points, n_windows, lookback]
    y = y_view[:, split_starts, :]          # [n_points, n_windows, horizon]

    if feature_mode == "only_s":
        # [n_points, n_windows, lookback, 1]
        X = x_s[..., np.newaxis]

    elif feature_mode == "sync_gw":
        x_gw = gw_hist_view[:, split_starts, :]
        X = np.stack([x_s, x_gw], axis=-1)  # [..., 2]

    elif feature_mode == "lag_gw":
        gw_indices = split_starts - lag
        x_gw = gw_hist_view[:, gw_indices, :]
        X = np.stack([x_s, x_gw], axis=-1)  # [..., 2]

    else:
        raise ValueError(f"不支持的 feature_mode: {feature_mode}")

    # reshape 成样本维度
    # 当前 X: [n_points, n_windows, lookback, n_features]
    # 目标变成: [N, lookback, n_features]
    n_windows = len(split_starts)
    n_features = X.shape[-1]

    X = X.reshape(n_points * n_windows, lookback, n_features)
    y = y.reshape(n_points * n_windows, horizon)

    # 元数据
    start_idx = np.tile(split_starts, n_points)
    point_id_rep = np.repeat(point_ids, n_windows)
    lon_rep = np.repeat(lons, n_windows)
    lat_rep = np.repeat(lats, n_windows)

    input_start_idx = start_idx
    input_end_idx = start_idx + lookback - 1
    target_start_idx = start_idx + lookback
    target_end_idx = start_idx + lookback + horizon - 1

    input_start_month = np.array([months[i] for i in input_start_idx], dtype=object)
    input_end_month = np.array([months[i] for i in input_end_idx], dtype=object)
    target_start_month = np.array([months[i] for i in target_start_idx], dtype=object)
    target_end_month = np.array([months[i] for i in target_end_idx], dtype=object)

    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.float32),
        "point_id": point_id_rep,
        "lon": lon_rep.astype(np.float32),
        "lat": lat_rep.astype(np.float32),
        "start_idx": start_idx.astype(np.int32),
        "input_start_idx": input_start_idx.astype(np.int32),
        "input_end_idx": input_end_idx.astype(np.int32),
        "target_start_idx": target_start_idx.astype(np.int32),
        "target_end_idx": target_end_idx.astype(np.int32),
        "input_start_month": input_start_month,
        "input_end_month": input_end_month,
        "target_start_month": target_start_month,
        "target_end_month": target_end_month,
    }


def save_split_npz(split_name: str, data_dict: Dict[str, np.ndarray], out_dir: Path) -> None:
    save_path = out_dir / f"{split_name}.npz"
    np.savez_compressed(save_path, **data_dict)
    print(f"[SAVE] {split_name}: {save_path}")


def write_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize_split(data_dict: Dict[str, np.ndarray], split_name: str) -> dict:
    X = data_dict["X"]
    y = data_dict["y"]
    return {
        "split": split_name,
        "n_samples": int(X.shape[0]),
        "lookback": int(X.shape[1]),
        "n_features": int(X.shape[2]),
        "horizon": int(y.shape[1]),
    }


# =========================
# 主流程
# =========================
def run_build(cfg: BuildConfig) -> None:
    print("=" * 80)
    print("[INFO] 开始构造数据集")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("=" * 80)

    # ---------- 1. 读取表头 ----------
    header_df = read_csv_safe(cfg.csv_path, nrows=0)
    all_cols = list(header_df.columns)

    required_basic_cols = ["ID", "Lon", "Lat"]
    for c in required_basic_cols:
        if c not in all_cols:
            raise ValueError(f"缺少必要列：{c}")

    s_cols, cy_cols = detect_time_series_columns(all_cols)
    months = validate_month_columns(s_cols, cy_cols)

    print(f"[INFO] 检测到 S_ 列数：{len(s_cols)}")
    print(f"[INFO] 检测到 CY_ 列数：{len(cy_cols)}")
    print(f"[INFO] 时间范围：{months[0]} ~ {months[-1]}")

    usecols = required_basic_cols + s_cols + cy_cols

    # ---------- 2. 读取完整数据 ----------
    df = read_csv_safe(cfg.csv_path, usecols=usecols)
    print(f"[INFO] 原始样本点数量：{len(df)}")

    # ---------- 3. 抽样 ----------
    df = select_points(df, cfg.sample_points, cfg.random_seed)

    # ---------- 4. 转为 numpy ----------
    # ID 可能是字符串，也可能是数字，这里统一保留原值
    point_ids = df["ID"].to_numpy()
    lons = df["Lon"].to_numpy(dtype=np.float32)
    lats = df["Lat"].to_numpy(dtype=np.float32)

    s_data = df[s_cols].to_numpy(dtype=np.float32)    # [n_points, 60]
    gw_data = df[cy_cols].to_numpy(dtype=np.float32)  # [n_points, 60]

    # ---------- 5. 基本检查 ----------
    n_points, n_time = s_data.shape
    if gw_data.shape != (n_points, n_time):
        raise ValueError("地面变形与地下水数据形状不一致。")

    print(f"[INFO] 数据矩阵形状：s_data={s_data.shape}, gw_data={gw_data.shape}")

    # ---------- 6. 有效窗口 ----------
    valid_starts = get_valid_start_indices(
    n_time=n_time,
    lookback=cfg.lookback,
    horizon=cfg.horizon,
    feature_mode=cfg.feature_mode,
    lag=cfg.lag,
    compare_max_lag=cfg.compare_max_lag,
    )
    
    split_starts = split_start_indices(
        valid_starts=valid_starts,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )

    print("[INFO] 有效窗口起点：", valid_starts.tolist())
    print(
        f"[INFO] split窗口数：train={len(split_starts['train'])}, "
        f"val={len(split_starts['val'])}, test={len(split_starts['test'])}"
    )

    # ---------- 7. 输出目录 ----------
    mode_name = cfg.feature_mode
    if cfg.feature_mode == "lag_gw":
        mode_name = f"{cfg.feature_mode}_lag{cfg.lag}"

    out_dir = Path(cfg.output_root) / mode_name
    ensure_dir(out_dir)

    # ---------- 8. 各 split 构造 ----------
    summaries = []

    for split_name in ["train", "val", "test"]:
        data_dict = build_arrays_for_split(
            s_data=s_data,
            gw_data=gw_data,
            point_ids=point_ids,
            lons=lons,
            lats=lats,
            months=months,
            split_starts=split_starts[split_name],
            cfg=cfg,
        )

        save_split_npz(split_name, data_dict, out_dir)
        summaries.append(summarize_split(data_dict, split_name))

    # ---------- 9. 保存配置 ----------
    config_dict = asdict(cfg)
    config_dict.update(
        {
            "n_points_used": int(n_points),
            "n_time": int(n_time),
            "months": months,
            "s_cols": s_cols,
            "cy_cols": cy_cols,
            "valid_starts": valid_starts.tolist(),
            "split_starts": {
                k: v.tolist() for k, v in split_starts.items()
            },
            "effective_history_len": int(n_time - cfg.horizon - (cfg.lag if cfg.feature_mode == "lag_gw" else 0)),
            "summary": summaries,
        }
    )

    write_json(config_dict, out_dir / "config.json")

    # ---------- 10. 保存简要说明 ----------
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[INFO] 数据集构造完成")
    print(f"[INFO] 输出目录：{out_dir}")
    print(summary_df)
    print("=" * 80)


# =========================
# 命令行入口
# =========================
def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="构造 LSTM 地面沉降预测数据集")

    parser.add_argument("--csv_path", type=str, required=True, help="原始 CSV 路径")
    parser.add_argument("--output_root", type=str, default="./stage1_outputs", help="输出根目录")

    parser.add_argument(
        "--feature_mode",
        type=str,
        default="only_s",
        choices=["only_s", "sync_gw", "lag_gw"],
        help="特征模式"
    )
    parser.add_argument("--lag", type=int, default=1, help="单滞后期，仅 lag_gw 生效")

    parser.add_argument(
    "--compare_max_lag",
    type=int,
    default=6,
    help="为公平比较不同lag，统一所有模型的最小窗口起点；不需要时传 -1"
    )
    parser.add_argument("--lookback", type=int, default=12, help="历史窗口长度")
    parser.add_argument("--horizon", type=int, default=5, help="预测步长")

    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")

    parser.add_argument(
        "--sample_points",
        type=int,
        default=10000,
        help="调试阶段抽样点数；若想使用全量，可传 0"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    sample_points = None if args.sample_points == 0 else args.sample_points

    return BuildConfig(
        csv_path=args.csv_path,
        output_root=args.output_root,
        feature_mode=args.feature_mode,
        lag=args.lag,
        compare_max_lag=None if args.compare_max_lag == -1 else args.compare_max_lag,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sample_points=sample_points,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_build(cfg)