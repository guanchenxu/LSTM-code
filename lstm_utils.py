# lstm_utils.py
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import joblib
import json

# =========================
# 文件与目录操作
# =========================
def ensure_dir(path: Path) -> None:
    """创建目录，如果不存在"""
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj: dict, path):
    """保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

# =========================
# 指标计算
# =========================
def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res/ss_tot) if ss_tot>0 else float("nan")

def mape_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred)/denom))*100.0)

# =========================
# 数据加载函数
# =========================
def load_stage_configs(stage1_dir, stage2_dir, stage3_dir):
    cfg1 = json.load(open(Path(stage1_dir)/"config.json", "r", encoding="utf-8"))
    cfg2 = json.load(open(Path(stage2_dir)/"model_config.json", "r", encoding="utf-8"))
    cfg3_path = Path(stage3_dir)/"train_config.json"
    cfg3 = json.load(open(cfg3_path, "r", encoding="utf-8")) if cfg3_path.exists() else None
    return {"stage1": cfg1, "stage2": cfg2, "stage3": cfg3}

def load_full_history_features(stage1_dir):
    path = Path(stage1_dir)/"X_hist.npy"
    if path.exists():
        return np.load(path)
    # 兜底兼容
    for alt in ["X_all.npy", "X_raw.npy"]:
        p = Path(stage1_dir)/alt
        if p.exists(): return np.load(p)
    raise FileNotFoundError("X_hist.npy not found")

def load_geo_info(stage1_dir):
    path = Path(stage1_dir)/"geo_data.csv"
    if not path.exists(): raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)

def load_target_scaler(stage1_dir):
    for p in [Path(stage1_dir)/"target_scaler.pkl", Path(stage1_dir)/"settle_scaler.pkl"]:
        if p.exists(): return joblib.load(p)
    return None

# =========================
# 模型构建与预测
# =========================
class BaseSettleLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def build_model(cfg, stage3_dir, device="cpu"):
    input_dim = cfg["stage2"]["input_dim"]
    hidden_dim = cfg["stage2"]["hidden_dim"]
    num_layers = cfg["stage2"]["num_layers"]
    output_dim = cfg["stage2"]["output_dim"]
    model = BaseSettleLSTM(input_dim, hidden_dim, num_layers, output_dim)
    weight_path = Path(stage3_dir)/"best_model_weights.pth"
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def predict_direct(model, X, batch_size=1024, device="cpu"):
    preds=[]
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start:start+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)

def update_window_for_rolling(window: np.ndarray, next_pred: np.ndarray) -> np.ndarray:
    new_window = np.empty_like(window)
    new_window[:-1] = window[1:]
    new_window[-1] = window[-1]
    new_window[-1, 0] = float(next_pred)
    return new_window

def predict_rolling(model, X: np.ndarray, horizon: int, batch_size=1024, device="cpu") -> np.ndarray:
    """
    滚动预测（递归滚动），支持两种情况：
    1. 模型 output_dim = 1：标准单步滚动
    2. 模型 output_dim > 1：直接多步模型，取第0步进行滚动
    """
    work = X.copy()
    n = X.shape[0]
    preds_all = np.zeros((n, horizon), dtype=np.float32)

    output_dim = model.fc.out_features
    single_step = (output_dim == 1)

    for step in range(horizon):
        step_preds = []
        for start in range(0, n, batch_size):
            batch = torch.tensor(work[start:start+batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model(batch).cpu().numpy()  # shape: (batch_size, output_dim)
                if not single_step:
                    # direct 多步模型：只取第0步进行滚动
                    pred = pred[:, 0]
                else:
                    pred = pred.reshape(-1)
                step_preds.append(pred)
        step_pred = np.concatenate(step_preds)
        if step_pred.shape[0] != n:
            raise ValueError(f"滚动预测长度不匹配：{step_pred.shape[0]} vs {n}")
        preds_all[:, step] = step_pred

        # 更新滚动窗口：沉降特征更新为预测值，其他特征保持最后一行
        for i in range(n):
            work[i] = update_window_for_rolling(work[i], step_pred[i])

    return preds_all