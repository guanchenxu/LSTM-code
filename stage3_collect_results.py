# -*- coding: utf-8 -*-
"""
stage3_collect_results.py

功能：
1. 扫描 stage2_outputs 与 stage2_outputs_full 下各实验目录
2. 自动读取 metrics.json
3. 汇总总体指标与分步长指标
4. 保存为 CSV 文件
5. 单独导出全区正式结果表（_full）

作者：ChatGPT
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_from_root(exp_root: Path, rows_overall: list, rows_per_step: list):
    if not exp_root.exists():
        print(f"[WARN] 目录不存在，跳过: {exp_root}")
        return

    exp_dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    exp_dirs = sorted(exp_dirs, key=lambda x: x.name)

    print(f"[INFO] 在 {exp_root} 下发现实验目录: {len(exp_dirs)}")

    for exp_dir in exp_dirs:
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"[WARN] 跳过，无 metrics.json: {exp_dir}")
            continue

        metrics = load_json(metrics_path)
        exp_name = exp_dir.name

        overall = metrics.get("overall", {})
        per_step = metrics.get("per_step", {})

        rows_overall.append({
            "model": exp_name,
            "MAE": overall.get("MAE"),
            "RMSE": overall.get("RMSE"),
            "R2": overall.get("R2"),
        })

        row = {"model": exp_name}
        for step_name, step_metrics in per_step.items():
            step_idx = step_name.replace("step_", "step")
            row[f"{step_idx}_MAE"] = step_metrics.get("MAE")
            row[f"{step_idx}_RMSE"] = step_metrics.get("RMSE")
            row[f"{step_idx}_R2"] = step_metrics.get("R2")
        rows_per_step.append(row)

        print(f"[OK] 已读取: {exp_name}")


def export_full_formal_tables(df_overall: pd.DataFrame, df_per_step: pd.DataFrame, save_root: Path):
    """
    导出全区正式结果表
    """
    formal_full_models = [
    "only_s_direct_cmp6_full_tuned",
    "lag5_gw_direct_cmp6_full",
    "only_s_recursive_cmp6_full",
]

    df_overall_full = df_overall[df_overall["model"].isin(formal_full_models)].copy()
    df_per_step_full = df_per_step[df_per_step["model"].isin(formal_full_models)].copy()

    order_map = {name: i for i, name in enumerate(formal_full_models)}
    df_overall_full["order"] = df_overall_full["model"].map(order_map)
    df_per_step_full["order"] = df_per_step_full["model"].map(order_map)

    df_overall_full = df_overall_full.sort_values("order").drop(columns="order")
    df_per_step_full = df_per_step_full.sort_values("order").drop(columns="order")

    overall_path = save_root / "summary_overall_full_formal.csv"
    per_step_path = save_root / "summary_per_step_full_formal.csv"

    df_overall_full.to_csv(overall_path, index=False, encoding="utf-8-sig")
    df_per_step_full.to_csv(per_step_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] 全区正式总体表: {overall_path}")
    print(f"[SAVE] 全区正式分步表: {per_step_path}")


def collect_results(stage2_roots: list[str], save_root: str):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    per_step_rows = []

    for root in stage2_roots:
        collect_from_root(Path(root), overall_rows, per_step_rows)

    df_overall = pd.DataFrame(overall_rows)
    df_per_step = pd.DataFrame(per_step_rows)

    if not df_overall.empty:
        df_overall = df_overall.sort_values("model").reset_index(drop=True)
    if not df_per_step.empty:
        df_per_step = df_per_step.sort_values("model").reset_index(drop=True)

    overall_csv = save_root / "summary_overall.csv"
    per_step_csv = save_root / "summary_per_step.csv"

    df_overall.to_csv(overall_csv, index=False, encoding="utf-8-sig")
    df_per_step.to_csv(per_step_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[INFO] 汇总完成")
    print(f"[SAVE] 总体指标表: {overall_csv}")
    print(f"[SAVE] 分步长指标表: {per_step_csv}")
    print("=" * 80)

    export_full_formal_tables(df_overall, df_per_step, save_root)

    print("\n[INFO] 总体指标预览:")
    print(df_overall.tail(10))

    print("\n[INFO] 分步长指标预览:")
    print(df_per_step.tail(5))


if __name__ == "__main__":
    collect_results(
        stage2_roots=[
            r"F:\aaa1\建模预测分析\stage2_outputs",
            r"F:\aaa1\建模预测分析\stage2_outputs_full",
        ],
        save_root=r"F:\aaa1\建模预测分析\results_summary",
    )