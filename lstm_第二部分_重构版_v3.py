import json
import os
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')


# ===================== 1. 工具函数 =====================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ===================== 2. 模型配置 =====================
@dataclass
class ModelConfig:
    """
    第二阶段：LSTM模型定义与初始化配置

    设计目标：
    1. 自动读取第一阶段输出的 config.json，避免手工同步参数
    2. 同时支持 direct（直接多步预测）和 rolling（单步滚动预测）
    3. 支持不同输入维度（含多lag地下水特征）
    4. 为第三阶段训练、第四阶段测试保留统一配置接口
    """

    # ---------- 第一阶段目录 ----------
    # 可直接修改这里；也支持系统环境变量 STAGE1_DIR 自动覆盖
    stage1_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage1_output"

    # ---------- 模型超参数（可调） ----------
    hidden_dim: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    bidirectional: bool = False

    # ---------- 输出头策略 ----------
    # last: 取最后一个时间步隐藏状态
    # mean: 对全部时间步做平均池化
    pooling: str = "last"

    # ---------- 训练准备参数（供后续阶段共享） ----------
    random_seed: int = 42
    device: Optional[str] = None

    # ---------- 初始化方式 ----------
    init_method: str = "xavier"  # xavier / kaiming / default

    def __post_init__(self) -> None:
        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling 必须为 'last' 或 'mean'")
        if self.init_method not in {"xavier", "kaiming", "default"}:
            raise ValueError("init_method 必须为 'xavier'、'kaiming' 或 'default'")

        # 优先支持环境变量覆盖，避免换目录时频繁改代码
        env_stage1_dir = os.environ.get("STAGE1_DIR", "").strip()
        if env_stage1_dir:
            self.stage1_dir = env_stage1_dir

        self.stage1_config_path = os.path.join(self.stage1_dir, "config.json")
        if not os.path.exists(self.stage1_config_path):
            raise FileNotFoundError(f"找不到第一阶段配置文件：{self.stage1_config_path}")

        with open(self.stage1_config_path, "r", encoding="utf-8") as f:
            stage1_cfg = json.load(f)

        # ---------- 第一阶段核心参数 ----------
        # 兼容新旧字段：优先使用第一阶段实际建模长度 L / effective_history_len
        stage1_L = stage1_cfg.get("L", None)
        stage1_effective_history_len = stage1_cfg.get("effective_history_len", None)
        stage1_history_len_base = stage1_cfg.get("history_len", None)

        if stage1_L is not None:
            self.L = int(stage1_L)
        elif stage1_effective_history_len is not None:
            self.L = int(stage1_effective_history_len)
        elif stage1_history_len_base is not None:
            self.L = int(stage1_history_len_base)
        else:
            raise KeyError("第一阶段 config.json 中缺少 L / effective_history_len / history_len，无法确定输入序列长度。")

        self.history_len_base = int(stage1_history_len_base) if stage1_history_len_base is not None else self.L
        self.effective_history_len = int(stage1_effective_history_len) if stage1_effective_history_len is not None else self.L

        # 一致性检查：若第一阶段同时保存了 L 和 effective_history_len，则两者应一致
        if stage1_L is not None and stage1_effective_history_len is not None:
            if int(stage1_L) != int(stage1_effective_history_len):
                raise ValueError(
                    f"第一阶段配置不一致：L={stage1_L}，effective_history_len={stage1_effective_history_len}。"
                    f"请检查第一阶段输出的 config.json。"
                )

        self.horizon = int(stage1_cfg["horizon"])
        self.pred_mode = str(stage1_cfg["pred_mode"])
        self.feature_mode = str(stage1_cfg["feature_mode"])
        self.input_dim = int(stage1_cfg["input_dim"])
        self.input_feature_names = list(stage1_cfg.get("input_feature_names", []))
        self.scaler_type = str(stage1_cfg.get("scaler_type", "standard"))
        self.lag_list = list(stage1_cfg.get("lag_list", []))
        self.max_lag = int(stage1_cfg.get("max_lag", 0))

        # compare_max_lag 兼容读取
        self.compare_max_lag = int(stage1_cfg.get("effective_compare_max_lag", stage1_cfg.get("compare_max_lag", self.max_lag)))

        # 标签输出长度
        self.H = self.horizon if self.pred_mode == "direct" else 1
        self.output_dim = self.H

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(self.device)

        # 第二阶段输出目录
        self.stage2_dir = os.path.join(os.path.dirname(self.stage1_dir), "stage2_output")
        os.makedirs(self.stage2_dir, exist_ok=True)

    @property
    def lstm_hidden_out_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    @property
    def model_name(self) -> str:
        return "SettleLSTMDirect" if self.pred_mode == "direct" else "SettleLSTMRolling"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


# ===================== 3. 模型定义 =====================
class BaseSettleLSTM(nn.Module):
    """
    通用LSTM骨架：
    - 输入:  (batch, L, input_dim)
    - 输出:  (batch, output_dim)

    说明：
    1. direct 模式下，output_dim = H（例如未来12个月）
    2. rolling 模式下，output_dim = 1（下一月），后续由测试阶段递归滚动调用
    """

    def __init__(self, config: ModelConfig):
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

        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.lstm_hidden_out_dim, config.output_dim)

        self._initialize_weights(config.init_method)

    def _initialize_weights(self, init_method: str) -> None:
        if init_method == "default":
            return

        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                if init_method == "xavier":
                    nn.init.xavier_uniform_(param)
                elif init_method == "kaiming":
                    nn.init.kaiming_uniform_(param, nonlinearity="sigmoid")
            elif "bias" in name:
                nn.init.zeros_(param)
                # LSTM遗忘门偏置置为1，有助于训练稳定
                n = param.size(0)
                start = n // 4
                end = n // 2
                param.data[start:end].fill_(1.0)
            elif name == "fc.weight":
                nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                nn.init.zeros_(param)

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
    """直接多步预测模型：输出未来 H 步。"""
    pass


class SettleLSTMRolling(BaseSettleLSTM):
    """单步预测模型：输出未来 1 步，供后续递归滚动预测。"""
    pass


def build_model(config: ModelConfig) -> nn.Module:
    if config.pred_mode == "direct":
        model = SettleLSTMDirect(config)
    elif config.pred_mode == "rolling":
        model = SettleLSTMRolling(config)
    else:
        raise ValueError(f"未知 pred_mode: {config.pred_mode}")
    return model.to(config.device)


# ===================== 4. 结构验证 =====================
def validate_model_structure(config: ModelConfig, model: nn.Module) -> Dict[str, Any]:
    """验证模型输入输出维度是否与第一阶段数据一致。"""
    model.eval()

    batch_size = 8
    x = torch.randn(batch_size, config.L, config.input_dim, device=config.device)

    with torch.no_grad():
        y = model(x)

    expected_in = (batch_size, config.L, config.input_dim)
    expected_out = (batch_size, config.output_dim)

    assert tuple(x.shape) == expected_in, f"输入维度错误：预期 {expected_in}，实际 {tuple(x.shape)}"
    assert tuple(y.shape) == expected_out, f"输出维度错误：预期 {expected_out}，实际 {tuple(y.shape)}"

    # 补充第一阶段 config 一致性检查
    with open(config.stage1_config_path, "r", encoding="utf-8") as f:
        stage1_cfg = json.load(f)

    stage1_L = stage1_cfg.get("L", None)
    if stage1_L is not None and int(stage1_L) != int(config.L):
        raise ValueError(
            f"第二阶段读取到的 L={config.L} 与第一阶段 config.json 中 L={stage1_L} 不一致，请检查配置。"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "model_name": config.model_name,
        "pred_mode": config.pred_mode,
        "feature_mode": config.feature_mode,
        "input_dim": config.input_dim,
        "output_dim": config.output_dim,
        "history_len": config.L,
        "history_len_base": config.history_len_base,
        "effective_history_len": config.effective_history_len,
        "compare_max_lag": config.compare_max_lag,
        "pooling": config.pooling,
        "bidirectional": config.bidirectional,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "device": config.device,
    }

    print("✅ 第二阶段模型结构验证通过")
    print(f"   模型名称：{summary['model_name']}")
    print(f"   输入维度：{tuple(x.shape)}")
    print(f"   输出维度：{tuple(y.shape)}")
    print(f"   特征方案：{config.feature_mode}")
    print(f"   预测方式：{config.pred_mode}")
    print(f"   总参数量：{total_params:,}")
    print(f"   设备：{config.device}")
    if config.history_len_base != config.L:
        print(f"   基础历史长度：{config.history_len_base}")
        print(f"   实际有效历史长度：{config.L}")
        print(f"   对齐比较最大滞后：{config.compare_max_lag}")

    return summary


# ===================== 5. 保存第二阶段结果 =====================
def save_model_stage_results(config: ModelConfig, model: nn.Module, summary: Dict[str, Any]) -> None:
    os.makedirs(config.stage2_dir, exist_ok=True)

    model_config_dict = {
        # 与第一阶段相关的关键参数
        "stage1_dir": config.stage1_dir,
        "history_len_base": config.history_len_base,
        "effective_history_len": config.effective_history_len,
        "history_len": config.L,
        "horizon": config.horizon,
        "L": config.L,
        "H": config.H,
        "pred_mode": config.pred_mode,
        "feature_mode": config.feature_mode,
        "input_dim": config.input_dim,
        "input_feature_names": config.input_feature_names,
        "lag_list": config.lag_list,
        "max_lag": config.max_lag,
        "compare_max_lag": config.compare_max_lag,
        "scaler_type": config.scaler_type,
        # 模型超参数
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout_rate": config.dropout_rate,
        "bidirectional": config.bidirectional,
        "pooling": config.pooling,
        "output_dim": config.output_dim,
        "random_seed": config.random_seed,
        "device": config.device,
        "init_method": config.init_method,
        # 验证摘要
        "model_name": config.model_name,
        "total_params": summary["total_params"],
        "trainable_params": summary["trainable_params"],
    }

    # 1) 保存模型配置
    with open(os.path.join(config.stage2_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config_dict, f, indent=4, ensure_ascii=False)

    # 2) 保存初始权重
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config_dict,
            "summary": summary,
        },
        os.path.join(config.stage2_dir, "initial_model_weights.pth"),
    )

    # 3) 保存模型结构文本
    with open(os.path.join(config.stage2_dir, "model_structure.txt"), "w", encoding="utf-8") as f:
        f.write(str(model))

    # 4) 保存维度与参数摘要
    with open(os.path.join(config.stage2_dir, "model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n📁 第二阶段结果已保存至：{config.stage2_dir}")
    print("保存文件说明：")
    print("- model_config.json: 统一模型配置（第三阶段训练直接读取）")
    print("- initial_model_weights.pth: 模型初始权重")
    print("- model_structure.txt: 模型结构文本")
    print("- model_summary.json: 维度与参数统计摘要")


# ===================== 6. 主流程 =====================
def main() -> None:
    config = ModelConfig()
    set_random_seed(config.random_seed)

    print("🔧 加载第一阶段配置并构建第二阶段模型...")
    model = build_model(config)

    print("\n🔍 验证模型输入输出维度...")
    summary = validate_model_structure(config, model)

    print("\n💾 保存第二阶段结果...")
    save_model_stage_results(config, model, summary)

    print("\n📋 模型结构：")
    print(model)
    print("\n🎉 第二阶段重构版运行完成！")
    print(f"   pred_mode             = {config.pred_mode}")
    print(f"   feature_mode          = {config.feature_mode}")
    print(f"   input_dim             = {config.input_dim}")
    print(f"   output_dim            = {config.output_dim}")
    print(f"   effective_history_len = {config.L}")
    print(f"   compare_max_lag       = {config.compare_max_lag}")


if __name__ == "__main__":
    main()