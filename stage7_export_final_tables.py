# -*- coding: utf-8 -*-
"""
stage7_export_final_tables.py

功能：
1. 读取 results_summary 中已有汇总表
2. 导出论文使用的最终表格：
   - 全区正式总体结果表
   - 全区正式分步长结果表
   - 抽样阶段方案筛选表
   - 代表点信息表

作者：ChatGPT
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def shorten_model_name(name: str) -> str:
    mapping = {
        "only_s_direct_cmp6_full_tuned": "Only-S Direct (Full)",
        "lag5_gw_direct_cmp6_full": "Lag5-GW Direct (Full)",
        "only_s_recursive_cmp6_full": "Only-S Recursive (Full)",

        "only_s_direct_cmp6": "Only-S Direct",
        "sync_gw_direct_cmp6": "Sync-GW Direct",
        "lag1_gw_direct_cmp6": "Lag1-GW Direct",
        "lag2_gw_direct_cmp6": "Lag2-GW Direct",
        "lag3_gw_direct_cmp6": "Lag3-GW Direct",
        "lag4_gw_direct_cmp6": "Lag4-GW Direct",
        "lag5_gw_direct_cmp6": "Lag5-GW Direct",
        "lag6_gw_direct_cmp6": "Lag6-GW Direct",
        "multi_lag_gw_3_5_direct_cmp6": "Multi-Lag (3,5) Direct",
        "multi_lag_gw_5_6_direct_cmp6": "Multi-Lag (5,6) Direct",
        "multi_lag_gw_2_3_5_direct_cmp6": "Multi-Lag (2,3,5) Direct",
        "only_s_recursive_cmp6": "Only-S Recursive",
        "lag5_gw_recursive_cmp6": "Lag5-GW Recursive",
    }
    return mapping.get(name, name)


def export_full_overall_table(df_overall: pd.DataFrame, save_dir: Path):
    models = [
        "only_s_direct_cmp6_full_tuned",
        "lag5_gw_direct_cmp6_full",
        "only_s_recursive_cmp6_full",
    ]
    df = df_overall[df_overall["model"].isin(models)].copy()
    order_map = {m: i for i, m in enumerate(models)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    df["模型"] = df["model"].map(shorten_model_name)
    df = df[["模型", "MAE", "RMSE", "R2"]]
    df = df.rename(columns={"R2": "R²"})

    out_path = save_dir / "表1_全区正式总体结果表.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")


def export_full_per_step_table(df_per_step: pd.DataFrame, save_dir: Path):
    models = [
        "only_s_direct_cmp6_full_tuned",
        "lag5_gw_direct_cmp6_full",
        "only_s_recursive_cmp6_full",
    ]
    df = df_per_step[df_per_step["model"].isin(models)].copy()
    order_map = {m: i for i, m in enumerate(models)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    df["模型"] = df["model"].map(shorten_model_name)

    keep_cols = [
        "模型",
        "step1_MAE", "step2_MAE", "step3_MAE", "step4_MAE", "step5_MAE",
        "step1_RMSE", "step2_RMSE", "step3_RMSE", "step4_RMSE", "step5_RMSE",
        "step1_R2", "step2_R2", "step3_R2", "step4_R2", "step5_R2",
    ]
    df = df[keep_cols]

    rename_map = {
        "step1_MAE": "Step1_MAE",
        "step2_MAE": "Step2_MAE",
        "step3_MAE": "Step3_MAE",
        "step4_MAE": "Step4_MAE",
        "step5_MAE": "Step5_MAE",
        "step1_RMSE": "Step1_RMSE",
        "step2_RMSE": "Step2_RMSE",
        "step3_RMSE": "Step3_RMSE",
        "step4_RMSE": "Step4_RMSE",
        "step5_RMSE": "Step5_RMSE",
        "step1_R2": "Step1_R²",
        "step2_R2": "Step2_R²",
        "step3_R2": "Step3_R²",
        "step4_R2": "Step4_R²",
        "step5_R2": "Step5_R²",
    }
    df = df.rename(columns=rename_map)

    out_path = save_dir / "表2_全区正式分步长结果表.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")


def export_sampling_selection_table(df_overall: pd.DataFrame, save_dir: Path):
    models = [
        "only_s_direct_cmp6",
        "sync_gw_direct_cmp6",
        "lag1_gw_direct_cmp6",
        "lag2_gw_direct_cmp6",
        "lag3_gw_direct_cmp6",
        "lag4_gw_direct_cmp6",
        "lag5_gw_direct_cmp6",
        "lag6_gw_direct_cmp6",
        "multi_lag_gw_3_5_direct_cmp6",
        "multi_lag_gw_5_6_direct_cmp6",
        "multi_lag_gw_2_3_5_direct_cmp6",
        "only_s_recursive_cmp6",
        "lag5_gw_recursive_cmp6",
    ]
    df = df_overall[df_overall["model"].isin(models)].copy()
    order_map = {m: i for i, m in enumerate(models)}
    df["order"] = df["model"].map(order_map)
    df = df.sort_values("order").drop(columns="order")

    df["模型"] = df["model"].map(shorten_model_name)
    df = df[["模型", "MAE", "RMSE", "R2"]]
    df = df.rename(columns={"R2": "R²"})

    out_path = save_dir / "表3_抽样阶段方案筛选结果表.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")


def export_representative_points_table(rep_points_df: pd.DataFrame, save_dir: Path):
    df = rep_points_df.copy()

    keep_cols = [
        "category", "point_id", "lon", "lat",
        "mean_true", "std_true", "range_true"
    ]
    df = df[keep_cols]

    df = df.rename(columns={
        "category": "类别",
        "point_id": "点ID",
        "lon": "经度",
        "lat": "纬度",
        "mean_true": "测试期平均真实值",
        "std_true": "测试期真实值标准差",
        "range_true": "测试期真实值范围",
    })

    out_path = save_dir / "表4_代表点信息表.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")


def main():
    summary_root = Path(r"F:\aaa1\建模预测分析\results_summary")
    save_dir = summary_root / "final_tables"
    ensure_dir(save_dir)

    df_overall = load_csv(summary_root / "summary_overall.csv")
    df_per_step = load_csv(summary_root / "summary_per_step.csv")
    rep_points_df = load_csv(summary_root / "representative_points" / "selected_points.csv")

    export_full_overall_table(df_overall, save_dir)
    export_full_per_step_table(df_per_step, save_dir)
    export_sampling_selection_table(df_overall, save_dir)
    export_representative_points_table(rep_points_df, save_dir)

    print("=" * 80)
    print("[INFO] 最终表格导出完成")
    print(f"[INFO] 输出目录: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()