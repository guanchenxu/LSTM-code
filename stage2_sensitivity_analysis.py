# -*- coding: utf-8 -*-
"""
stage2_sensitivity_analysis.py

功能：
1. 基于 stage2_train_lstm.py 做参数敏感性分析
2. 控制变量法（一次只改变一个参数）
3. 自动保存每组实验结果
4. 汇总生成 sensitivity_results.csv

作者：ChatGPT
"""

import os
import json
import copy
import pandas as pd

from stage2_train_lstm import TrainConfig, run_train


# =========================
# 基准配置（固定不变）
# =========================
BASE_CFG = TrainConfig(
    data_dir=r"F:\aaa1\建模预测分析\stage1_outputs\only_s",

    # ⭐ 新总输出目录（敏感性分析专用）
    output_dir=r"F:\aaa1\建模预测分析\stage2_outputs_sensitivity",

    batch_size=256,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    lr=1e-3,

    max_epochs=30,
    patience=5,
    device="cpu",
    random_seed=42,

    save_predictions=False,  # ⭐敏感性分析不保存预测（节省空间）
)


# =========================
# 参数敏感性设置（控制变量）
# =========================
SENSITIVITY_PARAMS = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.2, 0.5],
    "lr": [1e-4, 1e-3, 5e-3],
}


# =========================
# 主函数
# =========================
def run_sensitivity():
    results = []

    base_output = BASE_CFG.output_dir
    os.makedirs(base_output, exist_ok=True)

    print("=" * 80)
    print("[INFO] 开始参数敏感性分析")
    print("=" * 80)

    for param_name, values in SENSITIVITY_PARAMS.items():
        print(f"\n[INFO] 分析参数: {param_name}")
        print("-" * 60)

        for val in values:
            print(f"[RUN] {param_name} = {val}")

            # 深拷贝配置（避免污染）
            cfg = copy.deepcopy(BASE_CFG)

            # 设置参数
            setattr(cfg, param_name, val)

            # 输出路径：按参数分类
            cfg.output_dir = os.path.join(
                base_output,
                f"{param_name}",
                f"{param_name}_{val}"
            )

            # 运行训练
            run_train(cfg)

            # 读取结果
            metrics_path = os.path.join(cfg.output_dir, "metrics.json")

            if not os.path.exists(metrics_path):
                print(f"[WARN] 未找到 metrics.json: {metrics_path}")
                continue

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            row = {
                "param": param_name,
                "value": val,
                "MAE": metrics["overall"]["MAE"],
                "RMSE": metrics["overall"]["RMSE"],
                "R2": metrics["overall"]["R2"],
            }

            # ⭐ 加入分步指标（可选）
            for step, vals in metrics["per_step"].items():
                row[f"{step}_RMSE"] = vals["RMSE"]

            results.append(row)

    # =========================
    # 保存总结果
    # =========================
    df = pd.DataFrame(results)

    csv_path = os.path.join(base_output, "sensitivity_results.csv")
    json_path = os.path.join(base_output, "sensitivity_results.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print("\n" + "=" * 80)
    print("[INFO] 敏感性分析完成！")
    print(f"[SAVE] CSV: {csv_path}")
    print(f"[SAVE] JSON: {json_path}")
    print("=" * 80)


# =========================
# 入口
# =========================
if __name__ == "__main__":
    run_sensitivity()