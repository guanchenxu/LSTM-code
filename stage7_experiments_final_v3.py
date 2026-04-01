# stage7_experiments_final_v3.py
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import joblib
from lstm_utils import *

# =========================
# 路径配置
# =========================
RESULTS_BASE_DIR = Path(r"F:\aaa1\lstm建模预测滞后1期")
STAGE1_DIR = RESULTS_BASE_DIR / "stage1_output"
STAGE2_DIR = RESULTS_BASE_DIR / "stage2_output"
STAGE3_DIR = RESULTS_BASE_DIR / "stage3_output"
STAGE7_DIR = RESULTS_BASE_DIR / "stage7_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024

FEATURE_MODES = ["sync_gw","lag1_gw","lag2_gw","lag3_gw","multi_lag_gw"]
PRED_MODES = ["direct","rolling"]

# =========================
# 测试集真实值 + 索引
# =========================
y_test_path = STAGE1_DIR / "y_test.npy"
y_true = np.load(y_test_path)

scaler_path = STAGE1_DIR / "target_scaler.pkl"
scaler = joblib.load(scaler_path)
y_true_raw = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)

test_idx_path = STAGE1_DIR / "test_idx.npy"
test_idx = np.load(test_idx_path)

# =========================
# 指标函数
# =========================
def compute_metrics(y_true, y_pred):
    return {
        "MAE": mae_np(y_true, y_pred),
        "RMSE": rmse_np(y_true, y_pred),
        "R2": r2_score_np(y_true, y_pred),
        "MAPE": mape_np(y_true, y_pred)
    }

def compute_step_metrics(y_true, y_pred):
    rows=[]
    for i in range(y_pred.shape[1]):
        rows.append({
            "step": i+1,
            "MAE": mae_np(y_true[:,i],y_pred[:,i]),
            "RMSE": rmse_np(y_true[:,i],y_pred[:,i]),
            "R2": r2_score_np(y_true[:,i],y_pred[:,i]),
            "MAPE": mape_np(y_true[:,i],y_pred[:,i])
        })
    return pd.DataFrame(rows)

# =========================
# 实验循环
# =========================
def run_experiments(feature_modes=FEATURE_MODES, pred_modes=PRED_MODES):
    cfg_all = load_stage_configs(STAGE1_DIR, STAGE2_DIR, STAGE3_DIR)
    X_hist = load_full_history_features(STAGE1_DIR).astype(np.float32)
    ensure_dir(STAGE7_DIR)

    summary_all = []

    for feature_mode in feature_modes:
        for pred_mode in pred_modes:
            print(f"\n--- Running experiment: {feature_mode} + {pred_mode} ---")
            cfg = deepcopy(cfg_all)
            cfg["stage1"]["feature_mode"] = feature_mode
            cfg["stage1"]["pred_mode"] = pred_mode

            model = build_model(cfg, STAGE3_DIR, DEVICE)

            # 全区预测
            if pred_mode=="direct":
                y_pred = predict_direct(model, X_hist, batch_size=BATCH_SIZE, device=DEVICE)
            else:
                y_pred = predict_rolling(model, X_hist,
                                         horizon=cfg["stage1"].get("horizon",12),
                                         batch_size=BATCH_SIZE, device=DEVICE)

            # 保存目录
            exp_dir = STAGE7_DIR / f"{feature_mode}_{pred_mode}"
            ensure_dir(exp_dir)

            # 全区预测保存
            np.save(exp_dir / "y_pred.npy", y_pred)

            # =========================
            # 测试集指标计算
            # =========================
            y_pred_test = y_pred[test_idx, :]
            # 如果预测值是标准化，需要反标准化
            y_pred_test_raw = scaler.inverse_transform(y_pred_test.reshape(-1,1)).reshape(y_pred_test.shape)

            metrics_overall = compute_metrics(y_true_raw, y_pred_test_raw)
            metrics_step = compute_step_metrics(y_true_raw, y_pred_test_raw)

            # 保存指标
            pd.DataFrame([metrics_overall]).to_csv(exp_dir / "overall_metrics.csv", index=False)
            metrics_step.to_csv(exp_dir / "step_metrics.csv", index=False)

            # 保存实验摘要
            save_json({
                "feature_mode": feature_mode,
                "pred_mode": pred_mode,
                "summary_overall": metrics_overall
            }, exp_dir / "experiment_summary.json")

            summary_all.append({
                "feature_mode": feature_mode,
                "pred_mode": pred_mode,
                **metrics_overall
            })

    # 保存汇总表
    pd.DataFrame(summary_all).to_csv(STAGE7_DIR / "experiment_summary_all.csv", index=False)
    print("\n✅ All experiments completed. Summary saved in stage7_results.")

# =========================
# 主程序
# =========================
if __name__=="__main__":
    run_experiments(FEATURE_MODES, PRED_MODES)