#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第六部分（重构版）：全区预测结果还原与 QGIS 表格输出
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# =========================
# 可修改配置
# =========================
BASE_DIR = Path(r"F:\aaa1\lstm建模预测滞后1期")
STAGE1_DIR = BASE_DIR / "stage1_output"
STAGE2_DIR = BASE_DIR / "stage2_output"
STAGE3_DIR = BASE_DIR / "stage3_output"
OUTPUT_DIR = BASE_DIR / "stage6_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024


# =========================
# 工具函数
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_pickle(path: Path):
    # 已弃用 pickle，统一用 joblib
    return joblib.load(path)

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# =========================
# 模型定义（与第二部分兼容）
# =========================
class BaseSettleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2, bidirectional: bool = False,
                 pooling: str = "last"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, output_dim)
        self.pooling = pooling

    def _pool(self, lstm_out: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return lstm_out[:, -1, :]
        elif self.pooling == "mean":
            return lstm_out.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        feat = self._pool(out)
        feat = self.drop(feat)
        return self.fc(feat)

class SettleLSTMDirect(BaseSettleLSTM):
    pass

class SettleLSTMRolling(BaseSettleLSTM):
    pass


# =========================
# 数据读取
# =========================
def load_stage_configs() -> dict:
    cfg1 = load_json(STAGE1_DIR / "config.json")
    cfg2 = load_json(STAGE2_DIR / "model_config.json")
    cfg3_path = STAGE3_DIR / "train_config.json"
    cfg3 = load_json(cfg3_path) if cfg3_path.exists() else None
    return {"stage1": cfg1, "stage2": cfg2, "stage3": cfg3}

def load_time_labels() -> dict:
    path = STAGE1_DIR / "time_labels.json"
    return load_json(path) if path.exists() else {}

def load_geo_info() -> pd.DataFrame:
    p = STAGE1_DIR / "geo_data.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return pd.read_csv(p)

def load_full_history_features() -> np.ndarray:
    path = STAGE1_DIR / "X_hist.npy"
    if path.exists():
        return np.load(path)
    # 兜底兼容 X_raw 或 X_all
    for alt in ["X_all.npy", "X_raw.npy"]:
        p = STAGE1_DIR / alt
        if p.exists():
            return np.load(p)
    raise FileNotFoundError("X_hist.npy not found")

def load_true_future_all(stage1_cfg: dict) -> Tuple[Optional[np.ndarray], Optional[list]]:
    path = STAGE1_DIR / "y_all_raw.npy"
    if path.exists():
        y_all = np.load(path)
        labels = load_time_labels().get("future_labels")
        return y_all, labels
    return None, None

def load_target_scaler():
    for p in [STAGE1_DIR / "target_scaler.pkl", STAGE1_DIR / "settle_scaler.pkl"]:
        if p.exists():
            return joblib.load(p)
    return None


# =========================
# 模型构建
# =========================
def build_model(cfg: dict) -> nn.Module:
    params = dict(
        input_dim=cfg["stage2"]["input_dim"],
        hidden_dim=cfg["stage2"]["hidden_dim"],
        num_layers=cfg["stage2"]["num_layers"],
        output_dim=cfg["stage2"]["output_dim"],
        dropout=cfg["stage2"].get("dropout", 0.2),
        bidirectional=cfg["stage2"].get("bidirectional", False),
        pooling=cfg["stage2"].get("pooling", "last"),
    )
    model_cls = SettleLSTMDirect if cfg["stage1"].get("pred_mode","direct")=="direct" else SettleLSTMRolling
    model = model_cls(**params)

    weight_path = STAGE3_DIR / "best_model_weights.pth"
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])  # ✅ 这里改成取 model_state_dict
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# 预测逻辑
# =========================
def predict_direct(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start+batch_size], dtype=torch.float32).to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)

def update_window_for_rolling(window: np.ndarray, next_pred: np.ndarray) -> np.ndarray:
    new_window = np.empty_like(window)
    new_window[:-1] = window[1:]
    new_window[-1] = window[-1]
    new_window[-1,0] = float(next_pred)
    return new_window

def predict_rolling(model: nn.Module, X: np.ndarray, horizon: int, batch_size: int = 1024) -> np.ndarray:
    n = X.shape[0]
    work = X.copy()
    preds_all = np.zeros((n,horizon), dtype=np.float32)
    for step in range(horizon):
        step_preds = []
        with torch.no_grad():
            for start in range(0,n,batch_size):
                batch = torch.tensor(work[start:start+batch_size], dtype=torch.float32).to(DEVICE)
                pred = model(batch).cpu().numpy().reshape(-1)
                step_preds.append(pred)
        step_pred = np.concatenate(step_preds)
        preds_all[:,step] = step_pred
        for i in range(n):
            work[i] = update_window_for_rolling(work[i], step_pred[i])
    return preds_all


# =========================
# 反标准化 & 输出
# =========================
def inverse_transform_target(y_scaled: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return y_scaled
    orig_shape = y_scaled.shape
    return scaler.inverse_transform(y_scaled.reshape(-1,1)).reshape(orig_shape)


def build_month_columns(prefix: str, labels: list, count: int) -> list:
    if labels and len(labels)==count:
        return [f"{prefix}_{lab}" for lab in labels]
    return [f"{prefix}_step{i+1:02d}" for i in range(count)]


def prepare_full_output_dataframe(geo_df: pd.DataFrame, y_pred_raw: np.ndarray,
                                  y_true_raw: Optional[np.ndarray],
                                  future_labels: Optional[list]) -> pd.DataFrame:
    n,h = y_pred_raw.shape
    out_df = geo_df.copy().reset_index(drop=True)

    pred_cols = build_month_columns("pred", future_labels or [], h)
    pred_df = pd.DataFrame(y_pred_raw, columns=pred_cols)
    out_df = pd.concat([out_df, pred_df], axis=1)

    if y_true_raw is not None and y_true_raw.shape==y_pred_raw.shape:
        true_cols = build_month_columns("true", future_labels or [], h)
        err_cols = build_month_columns("err", future_labels or [], h)
        out_df = pd.concat([out_df,
                            pd.DataFrame(y_true_raw, columns=true_cols),
                            pd.DataFrame(y_pred_raw - y_true_raw, columns=err_cols)],
                           axis=1)
        out_df["err_mean"] = np.mean(y_pred_raw - y_true_raw, axis=1)
        out_df["err_abs_mean"] = np.mean(np.abs(y_pred_raw - y_true_raw), axis=1)
        out_df["rmse_point"] = np.sqrt(np.mean((y_pred_raw - y_true_raw)**2, axis=1))
    else:
        out_df["err_mean"] = np.nan
        out_df["err_abs_mean"] = np.nan
        out_df["rmse_point"] = np.nan

    out_df["pred_mean"] = np.mean(y_pred_raw, axis=1)
    out_df["pred_min"] = np.min(y_pred_raw, axis=1)
    out_df["pred_max"] = np.max(y_pred_raw, axis=1)
    if y_true_raw is not None and y_true_raw.shape==y_pred_raw.shape:
        out_df["true_mean"] = np.mean(y_true_raw, axis=1)
        out_df["true_min"] = np.min(y_true_raw, axis=1)
        out_df["true_max"] = np.max(y_true_raw, axis=1)
    return out_df


# =========================
# 主程序
# =========================
def main() -> None:
    ensure_dir(OUTPUT_DIR)
    cfg = load_stage_configs()
    geo_df = load_geo_info()
    X_hist = load_full_history_features().astype(np.float32)
    scaler = load_target_scaler()
    model = build_model(cfg)

    # 全区预测
    if cfg["stage1"].get("pred_mode","direct")=="direct":
        y_pred_scaled = predict_direct(model, X_hist, batch_size=BATCH_SIZE)
    else:
        y_pred_scaled = predict_rolling(model, X_hist, horizon=cfg["stage1"].get("horizon",12), batch_size=BATCH_SIZE)

    # 反标准化
    y_pred_raw = inverse_transform_target(y_pred_scaled, scaler)
    y_true_raw, future_labels = load_true_future_all(cfg["stage1"])

    # 输出表格
    full_df = prepare_full_output_dataframe(geo_df, y_pred_raw, y_true_raw, future_labels)
    csv_path = OUTPUT_DIR / "full_area_prediction_qgis.csv"
    full_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存 numpy
    np.save(OUTPUT_DIR / "y_pred_all_raw.npy", y_pred_raw)
    np.save(OUTPUT_DIR / "y_pred_all_scaled.npy", y_pred_scaled)
    if y_true_raw is not None:
        np.save(OUTPUT_DIR / "y_true_all_raw.npy", y_true_raw)

    # 指标
    overall = None
    step_df = None
    if y_true_raw is not None:
        yt = y_true_raw.reshape(-1)
        yp = y_pred_raw.reshape(-1)
        overall = {
            "MAE": mae_np(yt, yp),
            "RMSE": rmse_np(yt, yp),
            "R2": r2_score_np(yt, yp),
            "MAPE": mape_np(yt, yp),
        }
        rows = []
        for i in range(y_pred_raw.shape[1]):
            yt_step = y_true_raw[:,i]
            yp_step = y_pred_raw[:,i]
            rows.append({
                "step": i+1,
                "MAE": mae_np(yt_step, yp_step),
                "RMSE": rmse_np(yt_step, yp_step),
                "R2": r2_score_np(yt_step, yp_step),
                "MAPE": mape_np(yt_step, yp_step),
            })
        step_df = pd.DataFrame(rows)
        step_df.to_csv(OUTPUT_DIR / "full_area_step_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([overall]).to_csv(OUTPUT_DIR / "full_area_overall_metrics.csv", index=False, encoding="utf-8-sig")

    # 保存概要
    summary = {
        "pred_mode": cfg["stage1"].get("pred_mode","direct"),
        "history_len": cfg["stage1"].get("history_len",48),
        "horizon": cfg["stage1"].get("horizon",12),
        "input_dim": cfg["stage2"]["input_dim"],
        "n_points": len(geo_df),
        "device": DEVICE,
        "has_true_future": y_true_raw is not None,
        "future_labels": future_labels,
        "output_csv": str(csv_path),
    }
    if overall is not None:
        summary["overall_metrics"] = overall
    save_json(summary, OUTPUT_DIR / "stage6_summary.json")
    run_cfg = {
        "stage1_dir": str(STAGE1_DIR),
        "stage2_dir": str(STAGE2_DIR),
        "stage3_dir": str(STAGE3_DIR),
        "output_dir": str(OUTPUT_DIR),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
    }
    save_json(run_cfg, OUTPUT_DIR / "stage6_config.json")

    print("="*72)
    print("第六部分（重构版）执行完成")
    print(f"预测模式: {cfg['stage1'].get('pred_mode','direct')}")
    print(f"全区监测点数量: {len(geo_df)}")
    print(f"输出表格: {csv_path}")
    if overall is not None:
        print("全区整体指标（有真实未来值时）:")
        for k,v in overall.items():
            print(f"  {k}: {v:.6f}")
    else:
        print("未恢复到全区真实未来值，仅输出预测结果表。")
    print("="*72)


if __name__=="__main__":
    main()