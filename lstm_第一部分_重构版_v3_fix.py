import os
import re
import json
import math
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ===================== 1. 配置区 =====================
@dataclass
class Config:
    """
    第一阶段：数据处理与样本构造配置

    设计目标：
    1. 严格支持同步 / 单滞后 / 多滞后特征构造
    2. 先划分数据集，再拟合标准化器，避免信息泄漏
    3. 同时兼容直接多步预测（direct）和滚动预测（rolling）
    4. 为第二~第六阶段保留兼容文件名与扩展元数据
    """

    # ---------- 路径 ----------
    input_csv: str = r"F:\aaa1\lstm建模预测滞后1期\承压水监测井8.csv"
    output_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage1_output"

    # ---------- 基本字段 ----------
    id_col: str = "ID"
    lon_col: str = "Lon"
    lat_col: str = "Lat"
    gw_prefix: str = "CY_"  # 地下水列名前缀，如 CY_201801

    # ---------- 样本长度 ----------
    total_seq_len: int = 60     # 201801~202212 共 60 期
    history_len: int = 48       # 基础历史窗口（无滞后时最大可用长度）
    horizon: int = 12           # 未来窗口（direct）
    compare_max_lag: Optional[int] = None  # 统一按最大lag截断历史长度；None时默认取multi_lags最大值

    # ---------- 预测方式 ----------
    pred_mode: str = "direct"  # direct / rolling

    # ---------- 输入特征方式 ----------
    # settlement_only / sync_gw / lag1_gw / lag2_gw / lag3_gw / multi_lag_gw
    feature_mode: str = "lag1_gw"
    multi_lags: Tuple[int, ...] = (1, 2, 3, 6)

    # ---------- 数据集划分 ----------
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    random_seed: int = 42
    stratify_by_quantile: bool = False  # 可选：按沉降强度分层

    # ---------- 标准化 ----------
    scaler_type: str = "standard"  # standard / minmax

    # ---------- DataLoader（为第三阶段或调试保留） ----------
    batch_size: int = 64
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 数据清洗 ----------
    allow_drop_na_rows: bool = True
    max_allowed_na_ratio: float = 0.0

    # ---------- 文件保存 ----------
    save_dataloaders_preview: bool = True
    save_raw_targets: bool = True
    save_all_samples: bool = True

    def __post_init__(self) -> None:
        s = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"train_ratio + val_ratio + test_ratio 必须等于1，当前为 {s}")
        if self.pred_mode not in {"direct", "rolling"}:
            raise ValueError("pred_mode 必须为 'direct' 或 'rolling'")
        valid_modes = {
            "settlement_only",
            "sync_gw",
            "lag1_gw",
            "lag2_gw",
            "lag3_gw",
            "multi_lag_gw",
        }
        if self.feature_mode not in valid_modes:
            raise ValueError(f"feature_mode 不合法：{self.feature_mode}")
        if self.scaler_type not in {"standard", "minmax"}:
            raise ValueError("scaler_type 必须为 'standard' 或 'minmax'")
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def L(self) -> int:
        """
        实际送入模型的历史长度。
        由于严格滞后构造下需要统一截断，因此这里必须返回 effective_history_len，
        以保证 stage2 读取 config.json 中的 L 时与 X.shape[1] 完全一致。
        """
        return self.effective_history_len

    @property
    def H(self) -> int:
        return self.horizon if self.pred_mode == "direct" else 1

    @property
    def lag_list(self) -> List[int]:
        if self.feature_mode == "settlement_only":
            return []
        if self.feature_mode == "sync_gw":
            return [0]
        if self.feature_mode == "lag1_gw":
            return [1]
        if self.feature_mode == "lag2_gw":
            return [2]
        if self.feature_mode == "lag3_gw":
            return [3]
        if self.feature_mode == "multi_lag_gw":
            return list(self.multi_lags)
        raise ValueError(f"未知 feature_mode: {self.feature_mode}")

    @property
    def max_lag(self) -> int:
        return max(self.lag_list) if self.lag_list else 0

    @property
    def effective_compare_max_lag(self) -> int:
        if self.compare_max_lag is not None:
            return int(self.compare_max_lag)
        if len(self.multi_lags) > 0:
            return max(int(x) for x in self.multi_lags)
        return self.max_lag

    @property
    def effective_history_len(self) -> int:
        val = self.history_len - self.effective_compare_max_lag
        if val <= 0:
            raise ValueError(
                f"有效历史长度必须大于0：history_len={self.history_len}, compare_max_lag={self.effective_compare_max_lag}"
            )
        return val

    @property
    def input_feature_names(self) -> List[str]:
        if self.feature_mode == "settlement_only":
            return ["settlement"]
        if self.feature_mode == "sync_gw":
            return ["settlement", "gw_lag0"]
        if self.feature_mode in {"lag1_gw", "lag2_gw", "lag3_gw"}:
            lag = self.lag_list[0]
            return ["settlement", f"gw_lag{lag}"]
        if self.feature_mode == "multi_lag_gw":
            return ["settlement"] + [f"gw_lag{lag}" for lag in self.lag_list]
        raise ValueError(f"未知 feature_mode: {self.feature_mode}")

    @property
    def input_dim(self) -> int:
        return len(self.input_feature_names)


# ===================== 2. 工具函数 =====================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"未知 scaler_type: {kind}")


def is_yyyymm(text: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(text)))


def sort_month_cols(cols: List[str]) -> List[str]:
    return sorted(cols, key=lambda x: int(re.sub(r"\D", "", x)))


def extract_month_from_gw_col(col: str, prefix: str) -> str:
    if not str(col).startswith(prefix):
        raise ValueError(f"地下水列名 {col} 不以指定前缀 {prefix} 开头")
    return str(col)[len(prefix):]


# ===================== 3. 数据读取与检查 =====================
def load_and_validate_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    if not os.path.exists(config.input_csv):
        raise FileNotFoundError(f"输入CSV不存在：{config.input_csv}")

    df = pd.read_csv(config.input_csv)
    required = [config.id_col, config.lon_col, config.lat_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少基础字段：{missing}")

    settle_cols = [c for c in df.columns if is_yyyymm(c)]
    gw_cols = [c for c in df.columns if str(c).startswith(config.gw_prefix) and is_yyyymm(str(c)[len(config.gw_prefix):])]

    settle_cols = sort_month_cols(settle_cols)
    gw_cols = sorted(gw_cols, key=lambda x: int(extract_month_from_gw_col(x, config.gw_prefix)))

    if len(settle_cols) != config.total_seq_len:
        raise ValueError(f"沉降列数量={len(settle_cols)}，与 total_seq_len={config.total_seq_len} 不一致")
    if len(gw_cols) != config.total_seq_len:
        raise ValueError(f"地下水列数量={len(gw_cols)}，与 total_seq_len={config.total_seq_len} 不一致")

    settle_months = settle_cols
    gw_months = [extract_month_from_gw_col(c, config.gw_prefix) for c in gw_cols]
    if settle_months != gw_months:
        raise ValueError("沉降列月份与地下水列月份不一致，请检查列名")

    geo_df = df[[config.id_col, config.lon_col, config.lat_col]].copy()
    geo_df.columns = ["ID", "Lon", "Lat"]

    settle_raw = df[settle_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    gw_raw = df[gw_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    na_mask = np.isnan(settle_raw).any(axis=1) | np.isnan(gw_raw).any(axis=1)
    valid_mask = ~na_mask
    if na_mask.any():
        na_count = int(na_mask.sum())
        na_ratio = na_count / len(df)
        print(f"⚠️ 检测到含缺失值样本：{na_count} 行，占比 {na_ratio:.4%}")
        allow_ratio = config.max_allowed_na_ratio
        ratio_ok = (allow_ratio <= 0) or (na_ratio <= allow_ratio + 1e-12)
        if config.allow_drop_na_rows and ratio_ok:
            geo_df = geo_df.loc[valid_mask].reset_index(drop=True)
            settle_raw = settle_raw[valid_mask]
            gw_raw = gw_raw[valid_mask]
            df_clean = df.loc[valid_mask].reset_index(drop=True)
            print(f"✅ 已删除缺失样本，剩余 {len(geo_df)} 个测点")
        else:
            raise ValueError(
                "数据中存在缺失值，且缺失占比超过 max_allowed_na_ratio，"
                f"当前缺失占比={na_ratio:.4%}, 允许上限={config.max_allowed_na_ratio:.4%}"
            )
    else:
        df_clean = df.copy().reset_index(drop=True)

    if len(geo_df) == 0:
        raise ValueError("清洗后无有效样本")

    print("✅ 数据读取完成")
    print(f"   测点数：{len(geo_df)}")
    print(f"   沉降数据形状：{settle_raw.shape}")
    print(f"   地下水数据形状：{gw_raw.shape}")
    print(f"   时间范围：{settle_cols[0]} ~ {settle_cols[-1]}")

    return geo_df, df_clean, settle_raw, gw_raw, settle_cols, gw_cols


# ===================== 4. 原始样本构造 =====================
def _history_end_index(config: Config, total_seq_len: int) -> int:
    """
    将历史窗口末端固定为预测窗口开始前一个时间步，
    再统一按 compare_max_lag 截断历史长度，以便不同 lag 方案公平比较。
    """
    if config.pred_mode == "direct":
        return total_seq_len - config.horizon - 1
    return total_seq_len - 2


def validate_window_feasibility(config: Config, total_seq_len: int) -> None:
    end_hist = _history_end_index(config, total_seq_len)
    start_hist = end_hist - config.effective_history_len + 1

    if start_hist - config.max_lag < 0:
        raise ValueError(
            "当前配置无法构造严格滞后样本："
            f"history_start={start_hist}, max_lag={config.max_lag}，会访问负索引。"
            "请减小 compare_max_lag / max_lag，或减小 horizon。"
        )

    future_end = end_hist + (config.horizon if config.pred_mode == "direct" else 1)
    if future_end > total_seq_len - 1:
        raise ValueError(
            f"当前配置无法构造样本：需要访问到索引 {future_end}，但总长度仅到 {total_seq_len - 1}。"
            f"请减小 horizon，或检查 total_seq_len 设置。"
        )


def build_raw_samples(
    settle_raw: np.ndarray,
    gw_raw: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    构造未标准化原始样本。

    输出：
        X_raw: (n, L, d)
        y_raw: (n, H) 或 (n, 1)
    """
    num_samples, total_len = settle_raw.shape
    validate_window_feasibility(config, total_len)

    end_hist = _history_end_index(config, total_len)
    hist_times = list(range(end_hist - config.effective_history_len + 1, end_hist + 1))

    if config.pred_mode == "direct":
        future_times = list(range(end_hist + 1, end_hist + 1 + config.horizon))
    else:
        future_times = [end_hist + 1]

    X_raw = np.zeros((num_samples, config.effective_history_len, config.input_dim), dtype=np.float32)
    y_raw = np.zeros((num_samples, len(future_times)), dtype=np.float32)

    for i in range(num_samples):
        features = []
        # 特征1：沉降历史
        settle_hist = settle_raw[i, hist_times].reshape(-1, 1)
        features.append(settle_hist)

        # 地下水特征
        for lag in config.lag_list:
            gw_hist = np.array([gw_raw[i, t - lag] for t in hist_times], dtype=np.float32).reshape(-1, 1)
            features.append(gw_hist)

        x_hist = np.concatenate(features, axis=1)  # (L, d)
        y_future = settle_raw[i, future_times].reshape(-1)

        X_raw[i] = x_hist
        y_raw[i] = y_future

    meta = {
        "hist_time_idx": hist_times,
        "future_time_idx": future_times,
        "history_start_idx": hist_times[0],
        "history_end_idx": hist_times[-1],
        "future_start_idx": future_times[0],
        "future_end_idx": future_times[-1],
        "input_feature_names": config.input_feature_names,
        "input_dim": config.input_dim,
        "output_dim": len(future_times),
    }

    print("✅ 原始样本构造完成")
    print(f"   X_raw形状：{X_raw.shape} (样本数, L, 特征数)")
    print(f"   y_raw形状：{y_raw.shape} (样本数, 输出步长)")
    print(f"   历史时间索引：{hist_times[0]} ~ {hist_times[-1]}")
    print(f"   未来标签索引：{future_times[0]} ~ {future_times[-1]}")
    return X_raw, y_raw, meta


# ===================== 5. 数据集划分 =====================
def split_dataset(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    geo_df: pd.DataFrame,
    config: Config,
) -> Dict[str, np.ndarray]:
    n = len(X_raw)
    all_idx = np.arange(n)

    stratify = None
    if config.stratify_by_quantile:
        # 基于未来标签均值分层，可选
        target_mean = y_raw.mean(axis=1)
        try:
            stratify = pd.qcut(target_mean, q=5, labels=False, duplicates="drop")
        except Exception:
            stratify = None

    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(1.0 - config.train_ratio),
        random_state=config.random_seed,
        shuffle=True,
        stratify=stratify,
    )

    temp_ratio = config.val_ratio + config.test_ratio
    val_size_in_temp = config.val_ratio / temp_ratio

    temp_stratify = None
    if stratify is not None:
        temp_stratify = np.asarray(stratify)[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size_in_temp),
        random_state=config.random_seed,
        shuffle=True,
        stratify=temp_stratify,
    )

    split = {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "test_idx": np.sort(test_idx),
    }

    print("✅ 数据集划分完成")
    print(f"   训练集：{len(split['train_idx'])}")
    print(f"   验证集：{len(split['val_idx'])}")
    print(f"   测试集：{len(split['test_idx'])}")

    for name, idx in split.items():
        sub = geo_df.iloc[idx]
        print(
            f"   {name}: 经度范围 {sub['Lon'].min():.4f} ~ {sub['Lon'].max():.4f}, "
            f"纬度范围 {sub['Lat'].min():.4f} ~ {sub['Lat'].max():.4f}"
        )

    return split


# ===================== 6. 标准化 =====================
def fit_feature_scalers(X_train_raw: np.ndarray, config: Config) -> Dict[str, object]:
    """
    仅使用训练集拟合标准化器。
    规则：
    - settlement 单独一个 scaler
    - 所有 gw 相关特征共享一个 scaler（同一物理变量）
    """
    settlement_scaler = get_scaler(config.scaler_type)
    settlement_scaler.fit(X_train_raw[:, :, 0].reshape(-1, 1))

    gw_scaler = None
    if config.input_dim >= 2:
        gw_scaler = get_scaler(config.scaler_type)
        gw_train_concat = X_train_raw[:, :, 1:].reshape(-1, 1)
        gw_scaler.fit(gw_train_concat)

    target_scaler = get_scaler(config.scaler_type)
    # 标签本质是沉降，也单独按沉降标签拟合，避免强行复用输入沉降尺度
    # 注意：为保证预测反标准化严谨，target_scaler 应只在训练集标签上拟合
    return {
        "settlement_scaler": settlement_scaler,
        "gw_scaler": gw_scaler,
        "target_scaler": target_scaler,
    }


def fit_target_scaler(y_train_raw: np.ndarray, scaler_type: str):
    target_scaler = get_scaler(scaler_type)
    target_scaler.fit(y_train_raw.reshape(-1, 1))
    return target_scaler


def transform_X(X_raw: np.ndarray, scalers: Dict[str, object]) -> np.ndarray:
    X = X_raw.copy().astype(np.float32)
    settlement_scaler = scalers["settlement_scaler"]
    gw_scaler = scalers["gw_scaler"]

    X[:, :, 0] = settlement_scaler.transform(X[:, :, 0].reshape(-1, 1)).reshape(X.shape[0], X.shape[1])
    if X.shape[2] >= 2 and gw_scaler is not None:
        gw_trans = gw_scaler.transform(X[:, :, 1:].reshape(-1, 1)).reshape(X.shape[0], X.shape[1], X.shape[2] - 1)
        X[:, :, 1:] = gw_trans
    return X.astype(np.float32)



def transform_y(y_raw: np.ndarray, target_scaler) -> np.ndarray:
    y = target_scaler.transform(y_raw.reshape(-1, 1)).reshape(y_raw.shape)
    return y.astype(np.float32)


# ===================== 7. Dataset / DataLoader（兼容后续与调试） =====================
class SettleForecastDataset(Dataset):
    def __init__(self, X_hist: np.ndarray, y_future: np.ndarray):
        self.X_hist = X_hist.astype(np.float32)
        self.y_future = y_future.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X_hist)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X_hist[idx]), torch.from_numpy(self.y_future[idx])



def build_dataloaders(config: Config, data: Dict[str, np.ndarray]):
    train_loader = DataLoader(
        SettleForecastDataset(data["X_train"], data["y_train"]),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        SettleForecastDataset(data["X_val"], data["y_val"]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        SettleForecastDataset(data["X_test"], data["y_test"]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader


# ===================== 8. 保存函数 =====================
def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)



def summarize_dataset(geo_df: pd.DataFrame, split: Dict[str, np.ndarray], config: Config, meta: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "num_points": int(len(geo_df)),
        "feature_mode": config.feature_mode,
        "pred_mode": config.pred_mode,
        "lag_list": config.lag_list,
        "max_lag": config.max_lag,
        "input_feature_names": meta["input_feature_names"],
        "input_dim": meta["input_dim"],
        "history_len_base": config.history_len,
        "effective_history_len": config.effective_history_len,
        "compare_max_lag": config.effective_compare_max_lag,
        "horizon": config.H,
        "history_time_idx": meta["hist_time_idx"],
        "future_time_idx": meta["future_time_idx"],
        "split": {
            "train": int(len(split["train_idx"])),
            "val": int(len(split["val_idx"])),
            "test": int(len(split["test_idx"])),
        },
        "geo_range": {
            name: {
                "lon_min": float(geo_df.iloc[idx]["Lon"].min()),
                "lon_max": float(geo_df.iloc[idx]["Lon"].max()),
                "lat_min": float(geo_df.iloc[idx]["Lat"].min()),
                "lat_max": float(geo_df.iloc[idx]["Lat"].max()),
            }
            for name, idx in split.items()
        },
    }
    return summary



def save_stage1_outputs(
    config: Config,
    geo_df: pd.DataFrame,
    settle_cols: List[str],
    split: Dict[str, np.ndarray],
    raw_data: Dict[str, np.ndarray],
    proc_data: Dict[str, np.ndarray],
    scalers: Dict[str, object],
    meta: Dict[str, object],
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    # 1) 保存核心数组（兼容旧流程命名）
    np.save(os.path.join(config.output_dir, "X_train.npy"), proc_data["X_train"])
    np.save(os.path.join(config.output_dir, "X_val.npy"), proc_data["X_val"])
    np.save(os.path.join(config.output_dir, "X_test.npy"), proc_data["X_test"])
    np.save(os.path.join(config.output_dir, "y_train.npy"), proc_data["y_train"])
    np.save(os.path.join(config.output_dir, "y_val.npy"), proc_data["y_val"])
    np.save(os.path.join(config.output_dir, "y_test.npy"), proc_data["y_test"])

    # 兼容旧第1/第6阶段：保存全部历史特征
    if config.save_all_samples:
        np.save(os.path.join(config.output_dir, "X_hist.npy"), proc_data["X_all"])
        np.save(os.path.join(config.output_dir, "y_all.npy"), proc_data["y_all"])

    # 2) 保存原始标签（便于核查与后续分析）
    if config.save_raw_targets:
        np.save(os.path.join(config.output_dir, "y_train_raw.npy"), raw_data["y_train_raw"])
        np.save(os.path.join(config.output_dir, "y_val_raw.npy"), raw_data["y_val_raw"])
        np.save(os.path.join(config.output_dir, "y_test_raw.npy"), raw_data["y_test_raw"])
        np.save(os.path.join(config.output_dir, "y_all_raw.npy"), raw_data["y_all_raw"])

    # 3) 保存索引
    np.save(os.path.join(config.output_dir, "train_idx.npy"), split["train_idx"])
    np.save(os.path.join(config.output_dir, "val_idx.npy"), split["val_idx"])
    np.save(os.path.join(config.output_dir, "test_idx.npy"), split["test_idx"])

    # 4) 保存地理信息（兼容旧代码保留 geo_data.csv）
    geo_df.to_csv(os.path.join(config.output_dir, "geo_data.csv"), index=False, encoding="utf-8-sig")

    # 5) 保存标准化器
    joblib.dump(scalers["settlement_scaler"], os.path.join(config.output_dir, "settle_scaler.pkl"))
    if scalers["gw_scaler"] is not None:
        joblib.dump(scalers["gw_scaler"], os.path.join(config.output_dir, "cy_scaler.pkl"))
    joblib.dump(scalers["target_scaler"], os.path.join(config.output_dir, "target_scaler.pkl"))

    # 6) 保存配置（兼容 stage2 当前读取方式，保留 L/H/device/scaler_type）
    config_payload = asdict(config)
    config_payload.update(
        {
            "L": config.L,
            "H": config.H,
            "input_dim": config.input_dim,
            "input_feature_names": config.input_feature_names,
            "lag_list": config.lag_list,
            "max_lag": config.max_lag,
            "effective_compare_max_lag": config.effective_compare_max_lag,
            "effective_history_len": config.effective_history_len,
            "history_len_base": config.history_len,
            "history_time_idx": meta["hist_time_idx"],
            "future_time_idx": meta["future_time_idx"],
            "history_time_cols": [settle_cols[i] for i in meta["hist_time_idx"]],
            "future_time_cols": [settle_cols[i] for i in meta["future_time_idx"]],
        }
    )
    save_json(config_payload, os.path.join(config.output_dir, "config.json"))

    # 7) 保存时间与摘要信息
    time_payload = {
        "all_time_cols": settle_cols,
        "history_time_idx": meta["hist_time_idx"],
        "future_time_idx": meta["future_time_idx"],
        "history_time_cols": [settle_cols[i] for i in meta["hist_time_idx"]],
        "future_time_cols": [settle_cols[i] for i in meta["future_time_idx"]],
    }
    save_json(time_payload, os.path.join(config.output_dir, "time_labels.json"))

    dataset_summary = summarize_dataset(geo_df, split, config, meta)
    save_json(dataset_summary, os.path.join(config.output_dir, "dataset_summary.json"))

    print(f"\n📁 第一阶段结果已保存至：{config.output_dir}")
    print("保存文件说明：")
    print("- X_train/X_val/X_test.npy: 标准化后的输入特征")
    print("- y_train/y_val/y_test.npy: 标准化后的标签")
    print("- X_hist.npy / y_all.npy: 全部样本（兼容后续阶段）")
    print("- y_*_raw.npy: 原始尺度标签")
    print("- train_idx/val_idx/test_idx.npy: 数据集索引")
    print("- geo_data.csv: 全部测点地理信息")
    print("- settle_scaler.pkl / cy_scaler.pkl / target_scaler.pkl: 标准化器")
    print("- config.json / time_labels.json / dataset_summary.json: 配置与元数据")


# ===================== 9. 主流程 =====================
def main() -> None:
    config = Config()
    set_random_seed(config.random_seed)

    # Step1: 数据读取
    geo_df, _, settle_raw, gw_raw, settle_cols, _ = load_and_validate_data(config)

    # Step2: 构造未标准化原始样本
    X_raw, y_raw, meta = build_raw_samples(settle_raw, gw_raw, config)

    # Step3: 划分数据集（在标准化之前）
    split = split_dataset(X_raw, y_raw, geo_df, config)

    # Step4: 用训练集拟合 scaler
    X_train_raw = X_raw[split["train_idx"]]
    y_train_raw = y_raw[split["train_idx"]]
    scalers = fit_feature_scalers(X_train_raw, config)
    scalers["target_scaler"] = fit_target_scaler(y_train_raw, config.scaler_type)

    # Step5: 变换各子集
    X_all = transform_X(X_raw, scalers)
    y_all = transform_y(y_raw, scalers["target_scaler"])

    proc_data = {
        "X_all": X_all,
        "y_all": y_all,
        "X_train": X_all[split["train_idx"]],
        "X_val": X_all[split["val_idx"]],
        "X_test": X_all[split["test_idx"]],
        "y_train": y_all[split["train_idx"]],
        "y_val": y_all[split["val_idx"]],
        "y_test": y_all[split["test_idx"]],
    }

    raw_data = {
        "y_all_raw": y_raw,
        "y_train_raw": y_raw[split["train_idx"]],
        "y_val_raw": y_raw[split["val_idx"]],
        "y_test_raw": y_raw[split["test_idx"]],
    }

    # 基本检查
    for name, arr in proc_data.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 中存在 NaN 或 Inf")

    # Step6: DataLoader预览（可选，便于后续阶段检查）
    if config.save_dataloaders_preview:
        train_loader, val_loader, test_loader = build_dataloaders(config, proc_data)
        xb, yb = next(iter(train_loader))
        print("\n📦 DataLoader预览：")
        print(f"   train 第一个batch X形状：{tuple(xb.shape)}")
        print(f"   train 第一个batch y形状：{tuple(yb.shape)}")
        print(f"   val batches: {len(val_loader)}, test batches: {len(test_loader)}")

    # Step7: 保存结果
    save_stage1_outputs(
        config=config,
        geo_df=geo_df,
        settle_cols=settle_cols,
        split=split,
        raw_data=raw_data,
        proc_data=proc_data,
        scalers=scalers,
        meta=meta,
    )

    print("\n🎉 第一阶段重构版运行完成！")
    print(f"   feature_mode          = {config.feature_mode}")
    print(f"   pred_mode             = {config.pred_mode}")
    print(f"   input_dim             = {config.input_dim}")
    print(f"   output_dim            = {config.H}")
    print(f"   effective_history_len = {config.effective_history_len}")
    print(f"   compare_max_lag       = {config.effective_compare_max_lag}")


if __name__ == "__main__":
    main()
