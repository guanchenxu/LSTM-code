# -*- coding: utf-8 -*-
"""
stage2_train_lstm_recursive.py

功能：
1. 读取 stage1 输出的 train/val/test.npz
2. 训练单步 LSTM（one-step ahead）
3. 在测试集上做 recursive rolling 预测（滚动预测 horizon 步）
4. 支持：
   - only_s
   - sync_gw
   - lag_gw
5. 保存模型、指标、预测结果

说明：
- 这版与 direct 版分开保存，便于后续比较
- 训练目标仅为下一步（y[:, 0]）
- 测试评估时采用递归方式滚动预测未来 5 期

作者：ChatGPT
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader


# =========================
# 配置
# =========================
@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str

    batch_size: int = 256
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0

    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 30
    patience: int = 5

    num_workers: int = 0
    device: str = "cpu"
    random_seed: int = 42

    save_predictions: bool = True


# =========================
# 工具函数
# =========================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz_data(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_safe(csv_path: str, usecols=None, nrows=None) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, usecols=usecols, nrows=nrows, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV 文件：{csv_path}\n最后一次报错：{last_err}")


# =========================
# 标准化器
# =========================
class StandardScalerND:
    """
    X: [N, T, F]
    y: [N, 1]
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, arr: np.ndarray):
        if arr.ndim == 3:
            self.mean_ = arr.mean(axis=(0, 1), keepdims=True)
            self.std_ = arr.std(axis=(0, 1), keepdims=True)
        elif arr.ndim == 2:
            self.mean_ = arr.mean(axis=0, keepdims=True)
            self.std_ = arr.std(axis=0, keepdims=True)
        else:
            raise ValueError(f"不支持的维度: {arr.ndim}")

        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std_ + self.mean_

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }


# =========================
# Dataset
# =========================
class OneStepDataset(Dataset):
    def __init__(self, X: np.ndarray, y_next: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y_next).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# 模型
# =========================
class LSTMOneStepModel(nn.Module):
    """
    输入: [B, T, F]
    输出: [B, 1]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        y = self.head(last_out)
        return y


# =========================
# 指标
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def compute_step_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    horizon = y_true.shape[1]
    results = {}
    for i in range(horizon):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        results[f"step_{i+1}"] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }
    return results


# =========================
# 训练 / 验证（单步）
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        pred = model(Xb)
        loss = criterion(pred, yb)

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


# =========================
# 递归预测辅助
# =========================
def load_stage1_config(data_dir: str) -> dict:
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_id_to_gw_matrix(csv_path: str, cy_cols: list[str]) -> Tuple[dict, np.ndarray]:
    """
    读取原始 CSV 中的 ID + CY_* 列
    返回：
    - id_to_row: {ID: row_index}
    - gw_matrix: [n_points, n_time]
    """
    usecols = ["ID"] + cy_cols
    df = read_csv_safe(csv_path, usecols=usecols)

    ids = df["ID"].to_numpy()
    gw_matrix = df[cy_cols].to_numpy(dtype=np.float32)

    id_to_row = {ids[i]: i for i in range(len(ids))}
    return id_to_row, gw_matrix


def get_gw_sequences_for_samples(
    point_ids: np.ndarray,
    id_to_row: dict,
    gw_matrix: np.ndarray,
) -> np.ndarray:
    """
    根据样本 point_id 取出对应完整地下水序列
    返回 shape: [N, n_time]
    """
    row_idx = np.array([id_to_row[x] for x in point_ids], dtype=np.int32)
    return gw_matrix[row_idx]


@torch.no_grad()
def recursive_predict(
    model,
    X_raw: np.ndarray,                 # [N, lookback, F]
    y_true: np.ndarray,                # [N, horizon]
    point_ids: np.ndarray,             # [N]
    start_idx: np.ndarray,             # [N]
    x_scaler: StandardScalerND,
    y_scaler: StandardScalerND,
    feature_mode: str,
    lag: int,
    gw_sequences: np.ndarray | None,   # [N, n_time]
    device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    对全体样本做 horizon 步递归预测
    """
    model.eval()

    N, lookback, n_features = X_raw.shape
    horizon = y_true.shape[1]

    y_pred_all = np.zeros((N, horizon), dtype=np.float32)

    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)

        Xb_raw = X_raw[st:ed].copy()
        start_idx_b = start_idx[st:ed]

        # settlement 历史窗口
        if n_features == 1:
            s_hist = Xb_raw[:, :, 0].copy()  # [B, lookback]
            gw_hist = None
        else:
            s_hist = Xb_raw[:, :, 0].copy()
            gw_hist = Xb_raw[:, :, 1].copy()

        gw_seq_b = None if gw_sequences is None else gw_sequences[st:ed]

        for step in range(horizon):
            # 组装当前输入
            if n_features == 1:
                X_step_raw = s_hist[:, :, None]  # [B, T, 1]
            else:
                X_step_raw = np.stack([s_hist, gw_hist], axis=-1)  # [B, T, 2]

            X_step_std = x_scaler.transform(X_step_raw).astype(np.float32)
            X_step_tensor = torch.from_numpy(X_step_std).float().to(device)

            pred_std = model(X_step_tensor).cpu().numpy()   # [B, 1]
            pred_raw = y_scaler.inverse_transform(pred_std).reshape(-1)  # [B]

            y_pred_all[st:ed, step] = pred_raw

            # 更新 settlement 滚动窗口
            s_hist = np.concatenate([s_hist[:, 1:], pred_raw[:, None]], axis=1)

            # 更新 groundwater 滚动窗口（若存在）
            if n_features > 1:
                # 当前 stage1 的逻辑下，下一步窗口需要追加的 groundwater 索引为：
                # append_idx = target_start_idx - lag + step
                # 其中 target_start_idx = start_idx + lookback
                append_idx = start_idx_b + lookback - lag + step

                append_gw = gw_seq_b[np.arange(ed - st), append_idx]
                gw_hist = np.concatenate([gw_hist[:, 1:], append_gw[:, None]], axis=1)

    return y_pred_all


# =========================
# 主流程
# =========================
def run_train(cfg: TrainConfig):
    print("=" * 80)
    print("[INFO] 开始训练 LSTM recursive 模型")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("=" * 80)

    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)
    ensure_dir(cfg.output_dir)

    # ---------- 1. 读取 stage1 配置 ----------
    stage1_cfg = load_stage1_config(cfg.data_dir)
    feature_mode = stage1_cfg["feature_mode"]
    lag = int(stage1_cfg.get("lag", 0))
    csv_path = stage1_cfg["csv_path"]
    cy_cols = stage1_cfg["cy_cols"]

    print(f"[INFO] feature_mode = {feature_mode}")
    print(f"[INFO] lag = {lag}")

    # ---------- 2. 读取数据 ----------
    train_data = load_npz_data(os.path.join(cfg.data_dir, "train.npz"))
    val_data = load_npz_data(os.path.join(cfg.data_dir, "val.npz"))
    test_data = load_npz_data(os.path.join(cfg.data_dir, "test.npz"))

    X_train_raw = train_data["X"].astype(np.float32)
    X_val_raw = val_data["X"].astype(np.float32)
    X_test_raw = test_data["X"].astype(np.float32)

    y_train_full = train_data["y"].astype(np.float32)   # [N, horizon]
    y_val_full = val_data["y"].astype(np.float32)
    y_test_full = test_data["y"].astype(np.float32)

    # 训练只用下一步
    y_train_next = y_train_full[:, [0]]
    y_val_next = y_val_full[:, [0]]

    print(f"[INFO] X_train: {X_train_raw.shape}, y_train_full: {y_train_full.shape}")
    print(f"[INFO] X_val  : {X_val_raw.shape}, y_val_full  : {y_val_full.shape}")
    print(f"[INFO] X_test : {X_test_raw.shape}, y_test_full : {y_test_full.shape}")

    input_size = X_train_raw.shape[-1]

    # ---------- 3. 标准化 ----------
    x_scaler = StandardScalerND()
    y_scaler = StandardScalerND()

    x_scaler.fit(X_train_raw)
    y_scaler.fit(y_train_next)

    X_train_std = x_scaler.transform(X_train_raw).astype(np.float32)
    X_val_std = x_scaler.transform(X_val_raw).astype(np.float32)

    y_train_next_std = y_scaler.transform(y_train_next).astype(np.float32)
    y_val_next_std = y_scaler.transform(y_val_next).astype(np.float32)

    save_json(
        {
            "x_scaler": x_scaler.to_dict(),
            "y_scaler": y_scaler.to_dict(),
        },
        os.path.join(cfg.output_dir, "scalers.json")
    )

    # ---------- 4. DataLoader ----------
    train_ds = OneStepDataset(X_train_std, y_train_next_std)
    val_ds = OneStepDataset(X_val_std, y_val_next_std)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # ---------- 5. 模型 ----------
    model = LSTMOneStepModel(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ---------- 6. 训练 ----------
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    history = []

    ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss_one_step": float(val_loss),
        })

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | val_loss_one_step={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] 保存最优模型: epoch={epoch}, val_loss_one_step={val_loss:.6f}")
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[INFO] Early stopping at epoch={epoch}")
                break

    save_json(
        {
            "train_config": asdict(cfg),
            "stage1_feature_mode": feature_mode,
            "stage1_lag": lag,
            "best_val_loss_one_step": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "history": history,
        },
        os.path.join(cfg.output_dir, "train_history.json")
    )

    # ---------- 7. 读取原始 groundwater 序列（only_s 时不需要） ----------
    if feature_mode == "only_s":
        id_to_row = None
        gw_matrix = None
        gw_seq_test = None
    else:
        print("[INFO] 正在读取原始地下水序列用于递归滚动预测 ...")
        id_to_row, gw_matrix = build_id_to_gw_matrix(csv_path, cy_cols)
        gw_seq_test = get_gw_sequences_for_samples(
            point_ids=test_data["point_id"],
            id_to_row=id_to_row,
            gw_matrix=gw_matrix,
        ).astype(np.float32)

    # ---------- 8. 测试集递归预测 ----------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_pred_test = recursive_predict(
        model=model,
        X_raw=X_test_raw,
        y_true=y_test_full,
        point_ids=test_data["point_id"],
        start_idx=test_data["start_idx"].astype(np.int32),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_mode=feature_mode,
        lag=lag if feature_mode == "lag_gw" else 0,
        gw_sequences=gw_seq_test,
        device=device,
        batch_size=1024,
    )

    metrics_overall = compute_metrics(y_test_full, y_pred_test)
    metrics_steps = compute_step_metrics(y_test_full, y_pred_test)

    metrics = {
        "overall": metrics_overall,
        "per_step": metrics_steps,
    }

    save_json(metrics, os.path.join(cfg.output_dir, "metrics.json"))

    print("=" * 80)
    print("[INFO] 测试集总体指标：")
    print(json.dumps(metrics_overall, ensure_ascii=False, indent=2))
    print("[INFO] 分步长指标：")
    print(json.dumps(metrics_steps, ensure_ascii=False, indent=2))
    print("=" * 80)

    # ---------- 9. 保存预测 ----------
    if cfg.save_predictions:
        pred_path = os.path.join(cfg.output_dir, "test_predictions.npz")
        np.savez_compressed(
            pred_path,
            y_true=y_test_full.astype(np.float32),
            y_pred=y_pred_test.astype(np.float32),
            point_id=test_data["point_id"],
            lon=test_data["lon"],
            lat=test_data["lat"],
            start_idx=test_data["start_idx"],
            input_start_month=test_data["input_start_month"],
            input_end_month=test_data["input_end_month"],
            target_start_month=test_data["target_start_month"],
            target_end_month=test_data["target_end_month"],
        )
        print(f"[SAVE] 测试集预测结果已保存: {pred_path}")

    print(f"[INFO] 训练完成，输出目录: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = TrainConfig(
        data_dir=r"F:\aaa1\建模预测分析\stage1_outputs_full\only_s",
        output_dir=r"F:\aaa1\建模预测分析\stage2_outputs_full\only_s_recursive_cmp6_full",
        batch_size=256,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=30,
        patience=5,
        num_workers=0,
        device="cpu",
        random_seed=42,
        save_predictions=True,
    )
    run_train(cfg)