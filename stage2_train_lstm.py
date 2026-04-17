# -*- coding: utf-8 -*-
"""
stage2_train_lstm.py

功能：
1. 读取 stage1_build_dataset.py 生成的 train/val/test.npz
2. 对 X / y 做标准化（基于训练集）
3. 构建 PyTorch Dataset / DataLoader
4. 定义 LSTM 直接多步预测模型（direct multi-step）
5. 训练并保存最优模型
6. 在测试集上评估并保存预测结果

当前版本：
- 仅支持 direct 多步预测
- 支持 feature_mode 对应的任意输入维度
- CPU 训练

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


# =========================
# 标准化器
# =========================
class StandardScalerND:
    """
    针对 X:[N, T, F] 或 y:[N, H] 的标准化器
    使用训练集统计量
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, arr: np.ndarray):
        if arr.ndim == 3:
            # X: [N, T, F] -> 对 (N,T) 维求均值方差，保留 F
            self.mean_ = arr.mean(axis=(0, 1), keepdims=True)
            self.std_ = arr.std(axis=(0, 1), keepdims=True)
        elif arr.ndim == 2:
            # y: [N, H] -> 对 N 维求均值方差，保留 H
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
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# 模型
# =========================
class LSTMDirectModel(nn.Module):
    """
    直接多步预测：
    输入 [B, T, F]
    输出 [B, H]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
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
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, (h_n, c_n) = self.lstm(x)
        # 取最后时刻输出
        last_out = out[:, -1, :]   # [B, hidden]
        y = self.head(last_out)    # [B, horizon]
        return y


# =========================
# 评估函数
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
    """
    对 horizon 的每一步分别计算指标
    """
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
# 训练 / 验证
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


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for Xb, _ in loader:
        Xb = Xb.to(device)
        pred = model(Xb)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


# =========================
# 主流程
# =========================
def run_train(cfg: TrainConfig):
    print("=" * 80)
    print("[INFO] 开始训练 LSTM direct 模型")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("=" * 80)

    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)

    ensure_dir(cfg.output_dir)

    # ---------- 1. 读取数据 ----------
    train_data = load_npz_data(os.path.join(cfg.data_dir, "train.npz"))
    val_data = load_npz_data(os.path.join(cfg.data_dir, "val.npz"))
    test_data = load_npz_data(os.path.join(cfg.data_dir, "test.npz"))

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.float32)

    X_val = val_data["X"].astype(np.float32)
    y_val = val_data["y"].astype(np.float32)

    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.float32)

    print(f"[INFO] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[INFO] X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"[INFO] X_test : {X_test.shape}, y_test : {y_test.shape}")

    input_size = X_train.shape[-1]
    lookback = X_train.shape[1]
    horizon = y_train.shape[1]

    # ---------- 2. 标准化 ----------
    x_scaler = StandardScalerND()
    y_scaler = StandardScalerND()

    x_scaler.fit(X_train)
    y_scaler.fit(y_train)

    X_train_std = x_scaler.transform(X_train).astype(np.float32)
    X_val_std = x_scaler.transform(X_val).astype(np.float32)
    X_test_std = x_scaler.transform(X_test).astype(np.float32)

    y_train_std = y_scaler.transform(y_train).astype(np.float32)
    y_val_std = y_scaler.transform(y_val).astype(np.float32)
    y_test_std = y_scaler.transform(y_test).astype(np.float32)

    save_json(
        {
            "x_scaler": x_scaler.to_dict(),
            "y_scaler": y_scaler.to_dict(),
        },
        os.path.join(cfg.output_dir, "scalers.json")
    )

    # ---------- 3. DataLoader ----------
    train_ds = TimeSeriesDataset(X_train_std, y_train_std)
    val_ds = TimeSeriesDataset(X_val_std, y_val_std)
    test_ds = TimeSeriesDataset(X_test_std, y_test_std)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # ---------- 4. 模型 ----------
    model = LSTMDirectModel(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        horizon=horizon,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ---------- 5. 训练 ----------
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
            "val_loss": float(val_loss),
        })

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] 保存最优模型: epoch={epoch}, val_loss={val_loss:.6f}")
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[INFO] Early stopping at epoch={epoch}")
                break

    save_json(
        {
            "train_config": asdict(cfg),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "history": history,
        },
        os.path.join(cfg.output_dir, "train_history.json")
    )

    # ---------- 6. 测试集评估 ----------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_pred_test_std = predict(model, test_loader, device)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_std)

    metrics_overall = compute_metrics(y_test, y_pred_test)
    metrics_steps = compute_step_metrics(y_test, y_pred_test)

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

    # ---------- 7. 保存测试预测 ----------
    if cfg.save_predictions:
        pred_path = os.path.join(cfg.output_dir, "test_predictions.npz")
        np.savez_compressed(
            pred_path,
            y_true=y_test.astype(np.float32),
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
        output_dir=r"F:\aaa1\建模预测分析\stage2_outputs_full\only_s_direct_cmp6_full_tuned",
        batch_size=256,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
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