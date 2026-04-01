import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')


# ===================== 1. 测试配置 =====================
@dataclass
class TestConfig:
    stage1_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage1_output"
    stage2_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage2_output"
    stage3_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage3_output"

    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    device: Optional[str] = None

    save_predictions_csv: bool = True
    save_step_metrics_csv: bool = True
    save_test_arrays: bool = True
    save_predictions_npz: bool = True

    def __post_init__(self):
        # 关键文件路径
        self.stage1_config_path = os.path.join(self.stage1_dir, "config.json")
        self.stage2_config_path = os.path.join(self.stage2_dir, "model_config.json")
        self.stage3_best_weights_path = os.path.join(self.stage3_dir, "best_model_weights.pth")
        self.stage4_dir = os.path.join(os.path.dirname(self.stage3_dir), "stage4_output")
        os.makedirs(self.stage4_dir, exist_ok=True)

        # 检查文件存在
        for p in [self.stage1_config_path, self.stage2_config_path, self.stage3_best_weights_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少必要输入文件：{p}")

        # 读取配置
        with open(self.stage1_config_path, "r", encoding="utf-8") as f:
            s1 = json.load(f)
        with open(self.stage2_config_path, "r", encoding="utf-8") as f:
            s2 = json.load(f)
        s3 = {}
        stage3_cfg_path = os.path.join(self.stage3_dir, "train_config.json")
        if os.path.exists(stage3_cfg_path):
            with open(stage3_cfg_path, "r", encoding="utf-8") as f:
                s3 = json.load(f)

        # ---------- 核心参数优先读取 effective_history_len ----------
        self.history_len = int(
            s2.get("L", s2.get("effective_history_len", s1.get("effective_history_len", s1["history_len"])))
        )
        self.horizon = int(s2.get("horizon", s1["horizon"]))
        self.pred_mode = str(s2.get("pred_mode", s1["pred_mode"]))
        self.feature_mode = str(s2.get("feature_mode", s1["feature_mode"]))
        self.input_dim = int(s2["input_dim"])
        self.output_dim = int(s2["output_dim"])
        self.hidden_dim = int(s2["hidden_dim"])
        self.num_layers = int(s2["num_layers"])
        self.dropout_rate = float(s2["dropout_rate"])
        self.bidirectional = bool(s2.get("bidirectional", False))
        self.pooling = str(s2.get("pooling", "last"))
        self.model_name = str(s2.get("model_name", "SettleLSTMDirect" if self.pred_mode == "direct" else "SettleLSTMRolling"))
        self.input_feature_names = list(s2.get("input_feature_names", s1.get("input_feature_names", [])))
        self.lag_list = list(s2.get("lag_list", s1.get("lag_list", [])))
        self.max_lag = int(s2.get("max_lag", s1.get("max_lag", 0)))
        self.scaler_type = str(s2.get("scaler_type", s1.get("scaler_type", "standard")))
        self.future_time_cols = list(s1.get("future_time_cols", []))

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(self.device)

        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling 必须为 'last' 或 'mean'")
        if self.pred_mode not in {"direct", "rolling"}:
            raise ValueError("pred_mode 必须为 'direct' 或 'rolling'")


# ===================== 2. 模型定义 =====================
class BaseSettleLSTM(nn.Module):
    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(out_dim, config.output_dim)

    def _pool_sequence(self, lstm_out: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "last":
            return lstm_out[:, -1, :]
        if self.config.pooling == "mean":
            return lstm_out.mean(dim=1)
        raise ValueError(f"未知 pooling: {self.config.pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        features = self._pool_sequence(lstm_out)
        features = self.dropout(features)
        out = self.fc(features)
        return out


class SettleLSTMDirect(BaseSettleLSTM):
    pass


class SettleLSTMRolling(BaseSettleLSTM):
    pass


def build_model(config: TestConfig) -> nn.Module:
    if config.pred_mode == "direct":
        return SettleLSTMDirect(config).to(config.device)
    else:
        return SettleLSTMRolling(config).to(config.device)


# ===================== 3. 数据与标准化器 =====================
def load_stage1_test_arrays(config: TestConfig) -> Dict[str, np.ndarray]:
    arrays = {
        "X_test": np.load(os.path.join(config.stage1_dir, "X_test.npy")).astype(np.float32),
        "y_test": np.load(os.path.join(config.stage1_dir, "y_test.npy")).astype(np.float32),
        "test_idx": np.load(os.path.join(config.stage1_dir, "test_idx.npy")),
    }
    raw_path = os.path.join(config.stage1_dir, "y_test_raw.npy")
    arrays["y_test_raw"] = np.load(raw_path).astype(np.float32) if os.path.exists(raw_path) else None
    geo_df = pd.read_csv(os.path.join(config.stage1_dir, "geo_data.csv"))
    arrays["geo_test"] = geo_df.iloc[arrays["test_idx"]].reset_index(drop=True)
    with open(os.path.join(config.stage1_dir, "time_labels.json"), "r", encoding="utf-8") as f:
        arrays["time_labels"] = json.load(f)
    return arrays


def load_scalers(config: TestConfig):
    target_scaler = joblib.load(os.path.join(config.stage1_dir, "target_scaler.pkl"))
    settle_scaler = joblib.load(os.path.join(config.stage1_dir, "settle_scaler.pkl")) if os.path.exists(os.path.join(config.stage1_dir, "settle_scaler.pkl")) else None
    gw_scaler = joblib.load(os.path.join(config.stage1_dir, "cy_scaler.pkl")) if os.path.exists(os.path.join(config.stage1_dir, "cy_scaler.pkl")) else None
    return {"target_scaler": target_scaler, "settlement_scaler": settle_scaler, "gw_scaler": gw_scaler}


# ===================== 4. 推理 =====================
def load_best_model_weights(config: TestConfig, model: nn.Module) -> None:
    ckpt = torch.load(config.stage3_best_weights_path, map_location=config.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)


def build_test_loader(config: TestConfig, arrays: Dict[str, np.ndarray]) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(arrays["X_test"]).float(), torch.from_numpy(arrays["y_test"]).float())
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)


@torch.no_grad()
def predict_direct(model: nn.Module, loader: DataLoader, config: TestConfig) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(config.device)
        yb = yb.to(config.device)
        preds.append(model(xb).cpu().numpy())
        trues.append(yb.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def predict_rolling(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, config: TestConfig) -> Tuple[np.ndarray, np.ndarray]:
    n, horizon = X_test.shape[0], config.horizon
    y_pred_roll = np.zeros((n, horizon), dtype=np.float32)
    X_curr = X_test.copy()
    for step in range(horizon):
        batch_preds = []
        for b in range(0, n, config.batch_size):
            xb = torch.from_numpy(X_curr[b:b + config.batch_size]).float().to(config.device)
            batch_preds.append(model(xb).detach().cpu().numpy().reshape(-1))
        pred_step_all = np.concatenate(batch_preds)
        y_pred_roll[:, step] = pred_step_all
        # 滚动更新输入
        X_next = np.zeros_like(X_curr)
        X_next[:, :-1, :] = X_curr[:, 1:, :]
        X_next[:, -1, :] = X_curr[:, -1, :]
        X_next[:, -1, 0] = pred_step_all
        X_curr = X_next
    y_true = y_test if y_test.shape[1] == horizon else np.repeat(y_test, horizon, axis=1)
    return y_pred_roll.astype(np.float32), y_true.astype(np.float32)


# ===================== 5. 反标准化与指标 =====================
def inverse_transform_target(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape).astype(np.float32)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_flat, y_pred_flat = y_true.reshape(-1), y_pred.reshape(-1)
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    try:
        r2 = float(r2_score(y_true_flat, y_pred_flat))
    except Exception:
        r2 = float("nan")
    mask = np.abs(y_true_flat) > 1e-8
    mape = float(np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100.0) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def calc_step_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    steps = y_true.shape[1]
    records = []
    for i in range(steps):
        yt, yp = y_true[:, i], y_pred[:, i]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = float("nan")
        mask = np.abs(yt) > 1e-8
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0) if mask.any() else float("nan")
        records.append({"step": i + 1, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape})
    return pd.DataFrame(records)


# ===================== 6. 保存结果 =====================
def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def build_predictions_dataframe(geo_test: pd.DataFrame, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, future_time_cols: List[str]) -> pd.DataFrame:
    df = geo_test.copy().reset_index(drop=True)
    steps = y_true_raw.shape[1]
    cols = future_time_cols[:steps] if len(future_time_cols) >= steps else [f"step_{i+1}" for i in range(steps)]
    for i, c in enumerate(cols):
        df[f"true_{c}"] = y_true_raw[:, i]
        df[f"pred_{c}"] = y_pred_raw[:, i]
        df[f"err_{c}"] = y_pred_raw[:, i] - y_true_raw[:, i]
    df["mean_abs_error"] = np.mean(np.abs(y_pred_raw - y_true_raw), axis=1)
    df["mean_error"] = np.mean(y_pred_raw - y_true_raw, axis=1)
    return df


def save_stage4_outputs(config: TestConfig, arrays: Dict[str, np.ndarray], y_pred_scaled: np.ndarray, y_true_scaled: np.ndarray, y_pred_raw: np.ndarray, y_true_raw: np.ndarray, overall_metrics: Dict[str, float], step_metrics_df: pd.DataFrame) -> None:
    os.makedirs(config.stage4_dir, exist_ok=True)

    if config.save_test_arrays:
        np.save(os.path.join(config.stage4_dir, "y_pred_test_scaled.npy"), y_pred_scaled)
        np.save(os.path.join(config.stage4_dir, "y_true_test_scaled.npy"), y_true_scaled)
        np.save(os.path.join(config.stage4_dir, "y_pred_test_raw.npy"), y_pred_raw)
        np.save(os.path.join(config.stage4_dir, "y_true_test_raw.npy"), y_true_raw)

    if config.save_step_metrics_csv:
        step_metrics_df.to_csv(os.path.join(config.stage4_dir, "step_metrics.csv"), index=False, encoding="utf-8-sig")

    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(os.path.join(config.stage4_dir, "overall_metrics.csv"), index=False, encoding="utf-8-sig")

    pred_df = build_predictions_dataframe(arrays["geo_test"], y_true_raw, y_pred_raw, config.future_time_cols)
    if config.save_predictions_csv:
        pred_df.to_csv(os.path.join(config.stage4_dir, "test_predictions_detail.csv"), index=False, encoding="utf-8-sig")

    if config.save_predictions_npz:
        np.savez(os.path.join(config.stage4_dir, "test_predictions.npz"),
                 X_test=arrays["X_test"], y_true=y_true_raw, y_pred=y_pred_raw)

    summary = {
        "pred_mode": config.pred_mode,
        "feature_mode": config.feature_mode,
        "input_dim": config.input_dim,
        "output_dim": config.output_dim,
        "horizon": config.horizon,
        "lag_list": config.lag_list,
        "overall_metrics": overall_metrics,
        "num_test_samples": int(arrays["X_test"].shape[0]),
        "future_time_cols": config.future_time_cols,
        "stage4_dir": config.stage4_dir,
    }
    save_json(summary, os.path.join(config.stage4_dir, "test_summary.json"))
    save_json(asdict(config), os.path.join(config.stage4_dir, "test_config.json"))

    print(f"\n📁 第四阶段结果已保存至：{config.stage4_dir}")
    print("- overall_metrics.csv")
    print("- step_metrics.csv")
    print("- test_predictions_detail.csv")
    print("- y_pred_test_raw.npy / y_true_test_raw.npy")
    print("- y_pred_test_scaled.npy / y_true_test_scaled.npy")
    print("- test_predictions.npz")
    print("- test_summary.json / test_config.json")


# ===================== 7. 主流程 =====================
def main():
    config = TestConfig()
    arrays = load_stage1_test_arrays(config)
    scalers = load_scalers(config)
    model = build_model(config)
    load_best_model_weights(config, model)

    if config.pred_mode == "direct":
        test_loader = build_test_loader(config, arrays)
        y_pred_scaled, y_true_scaled = predict_direct(model, test_loader, config)
    else:
        y_pred_scaled, y_true_scaled = predict_rolling(model, arrays["X_test"], arrays["y_test"], config)

    y_pred_raw = inverse_transform_target(y_pred_scaled, scalers["target_scaler"])
    if arrays.get("y_test_raw") is not None and arrays["y_test_raw"] is not None and arrays["y_test_raw"].shape == y_pred_raw.shape:
        y_true_raw = arrays["y_test_raw"].astype(np.float32)
    else:
        y_true_raw = inverse_transform_target(y_true_scaled, scalers["target_scaler"])

    overall_metrics = calc_metrics(y_true_raw, y_pred_raw)
    step_metrics_df = calc_step_metrics(y_true_raw, y_pred_raw)

    print("\n✅ 第四阶段测试完成")
    for k, v in overall_metrics.items():
        print(f"   {k}: {v:.6f}" if isinstance(v, float) and np.isfinite(v) else f"   {k}: {v}")

    save_stage4_outputs(config, arrays, y_pred_scaled, y_true_scaled, y_pred_raw, y_true_raw, overall_metrics, step_metrics_df)


if __name__ == "__main__":
    main()