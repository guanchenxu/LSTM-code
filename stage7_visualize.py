import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 中文字体设置
# =========================
rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False    # 负号正常显示

# =========================
# 路径配置
# =========================
RESULTS_DIR = Path(r"F:\aaa1\lstm建模预测滞后1期\stage7_results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FEATURE_MODES = ["sync_gw", "lag1_gw", "lag2_gw", "lag3_gw", "multi_lag_gw"]
PRED_MODES = ["direct", "rolling"]

# =========================
# 读取实验汇总表
# =========================
summary_file = RESULTS_DIR / "experiment_summary_all.csv"
summary_df = pd.read_csv(summary_file)
print("实验汇总读取完成，数据行数:", len(summary_df))

metrics = ["MAE", "RMSE", "R2", "MAPE"]

# =========================
# 图 1: 滞后方案 × 预测模式指标对比
# =========================
for metric in metrics:
    plt.figure(figsize=(10,6))
    for pred_mode in PRED_MODES:
        vals = []
        for fm in FEATURE_MODES:
            row = summary_df[(summary_df["feature_mode"]==fm) & (summary_df["pred_mode"]==pred_mode)]
            if row.empty:
                vals.append(np.nan)
            else:
                vals.append(float(row[metric].values[0]))
        plt.plot(FEATURE_MODES, vals, marker='o', label=pred_mode)
    plt.title(f"{metric} 对比：滞后方案 × 预测模式")
    plt.xlabel("滞后方案")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"metric_{metric}.png", dpi=300)
    plt.close()

print("图 1: 滞后方案 × direct/rolling 指标对比完成")

# =========================
# 图 2: 逐步指标折线图（每步预测误差）
# =========================
for fm in FEATURE_MODES:
    for pm in PRED_MODES:
        step_file = RESULTS_DIR / f"{fm}_{pm}/step_metrics.csv"
        if step_file.exists():
            step_df = pd.read_csv(step_file)
            plt.figure(figsize=(10,6))
            for m in ["MAE", "RMSE"]:
                plt.plot(step_df["step"], step_df[m], marker='o', label=m)
            plt.title(f"{fm} + {pm} 逐步预测误差")
            plt.xlabel("预测步长（月）")
            plt.ylabel("误差")
            plt.xticks(step_df["step"])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"step_{fm}_{pm}.png", dpi=300)
            plt.close()

print("图 2: 逐步预测误差折线图完成")