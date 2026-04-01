import json
import os
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ===================== 1. 工具函数 =====================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ===================== 2. 配置 =====================
@dataclass
class TrainConfig:
    """
    第三阶段：模型训练与验证配置

    设计目标：
    1. 自动读取第一阶段与第二阶段配置，避免参数手工同步
    2. 同时兼容 direct 与 rolling 两种预测模式
    3. 统一训练日志、最优模型保存与实验摘要输出
    4. 为第四阶段测试和第五阶段可视化提供标准文件接口
    """

    # ---------- 输入路径 ----------
    stage1_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage1_output"
    stage2_dir: str = r"F:\aaa1\lstm建模预测滞后1期\stage2_output"

    # ---------- 训练超参数 ----------
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # ---------- 早停 ----------
    patience: int = 10
    min_delta: float = 1e-6

    # ---------- 学习率调度（可选） ----------
    use_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    scheduler_min_lr: float = 1e-6

    # ---------- 训练设备 ----------
    num_workers: int = 0
    pin_memory: bool = False
    device: Optional[str] = None

    # ---------- 损失函数 ----------
    loss_name: str = "mse"

    # ---------- 可复现 ----------
    random_seed: int = 42

    def __post_init__(self) -> None:
        env_stage1_dir = os.environ.get("STAGE1_DIR", "").strip()
        env_stage2_dir = os.environ.get("STAGE2_DIR", "").strip()
        if env_stage1_dir:
            self.stage1_dir = env_stage1_dir
        if env_stage2_dir:
            self.stage2_dir = env_stage2_dir

        self.stage1_config_path = os.path.join(self.stage1_dir, "config.json")
        self.stage2_config_path = os.path.join(self.stage2_dir, "model_config.json")
        self.stage2_init_weights_path = os.path.join(self.stage2_dir, "initial_model_weights.pth")

        if not os.path.exists(self.stage1_config_path):
            raise FileNotFoundError(f"找不到第一阶段配置：{self.stage1_config_path}")
        if not os.path.exists(self.stage2_config_path):
            raise FileNotFoundError(f"找不到第二阶段配置：{self.stage2_config_path}")
        if not os.path.exists(self.stage2_init_weights_path):
            raise FileNotFoundError(f"找不到第二阶段初始权重：{self.stage2_init_weights_path}")

        with open(self.stage1_config_path, "r", encoding="utf-8") as f:
            s1 = json.load(f)
        with open(self.stage2_config_path, "r", encoding="utf-8") as f:
            s2 = json.load(f)

        # ---------- 继承第一/二阶段关键参数 ----------
        # 优先使用实际建模长度 L / effective_history_len，避免误读成基础历史长度
        self.history_len = int(
            s2.get(
                "L",
                s2.get(
                    "effective_history_len",
                    s2.get(
                        "history_len",
                        s1.get("L", s1.get("effective_history_len", s1["history_len"]))
                    )
                )
            )
        )
        self.history_len_base = int(s2.get("history_len_base", s1.get("history_len", self.history_len)))
        self.effective_history_len = int(s2.get("effective_history_len", self.history_len))

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
        self.init_method = str(s2.get("init_method", "xavier"))
        self.model_name = str(
            s2.get("model_name", "SettleLSTMDirect" if self.pred_mode == "direct" else "SettleLSTMRolling")
        )
        self.input_feature_names = list(s2.get("input_feature_names", s1.get("input_feature_names", [])))
        self.lag_list = list(s2.get("lag_list", s1.get("lag_list", [])))
        self.max_lag = int(s2.get("max_lag", s1.get("max_lag", 0)))
        self.compare_max_lag = int(s2.get("compare_max_lag", s1.get("effective_compare_max_lag", s1.get("compare_max_lag", self.max_lag))))
        self.scaler_type = str(s2.get("scaler_type", s1.get("scaler_type", "standard")))

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(self.device)

        self.stage3_dir = os.path.join(os.path.dirname(self.stage2_dir), "stage3_output")
        os.makedirs(self.stage3_dir, exist_ok=True)

        if self.loss_name.lower() != "mse":
            raise ValueError("当前重构版第三阶段仅支持 MSELoss，请设置 loss_name='mse'")
        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling 必须为 'last' 或 'mean'")

    @property
    def L(self) -> int:
        return self.history_len


# ===================== 3. 模型定义（与第二阶段保持一致） =====================
class BaseSettleLSTM(nn.Module):
    def __init__(self, config: TrainConfig):
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


def build_model(config: TrainConfig) -> nn.Module:
    if config.pred_mode == "direct":
        model = SettleLSTMDirect(config)
    elif config.pred_mode == "rolling":
        model = SettleLSTMRolling(config)
    else:
        raise ValueError(f"未知 pred_mode: {config.pred_mode}")
    return model.to(config.device)


# ===================== 4. 数据加载 =====================
def load_stage1_arrays(config: TrainConfig) -> Dict[str, np.ndarray]:
    required_files = [
        "X_train.npy", "X_val.npy", "X_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "train_idx.npy", "val_idx.npy", "test_idx.npy",
    ]
    for f in required_files:
        path = os.path.join(config.stage1_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少第一阶段输出文件：{path}")

    data = {
        "X_train": np.load(os.path.join(config.stage1_dir, "X_train.npy")),
        "X_val": np.load(os.path.join(config.stage1_dir, "X_val.npy")),
        "X_test": np.load(os.path.join(config.stage1_dir, "X_test.npy")),
        "y_train": np.load(os.path.join(config.stage1_dir, "y_train.npy")),
        "y_val": np.load(os.path.join(config.stage1_dir, "y_val.npy")),
        "y_test": np.load(os.path.join(config.stage1_dir, "y_test.npy")),
        "train_idx": np.load(os.path.join(config.stage1_dir, "train_idx.npy")),
        "val_idx": np.load(os.path.join(config.stage1_dir, "val_idx.npy")),
        "test_idx": np.load(os.path.join(config.stage1_dir, "test_idx.npy")),
    }

    for split_name in ["train", "val", "test"]:
        X = data[f"X_{split_name}"]
        y = data[f"y_{split_name}"]
        if X.ndim != 3:
            raise ValueError(f"X_{split_name} 应为三维张量，实际形状：{X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y_{split_name} 应为二维张量，实际形状：{y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_{split_name} 与 y_{split_name} 样本数不一致：{X.shape[0]} vs {y.shape[0]}")

    if data["X_train"].shape[1] != config.history_len:
        raise ValueError(
            f"history_len 不一致：stage1={data['X_train'].shape[1]} vs config={config.history_len}"
        )
    if data["X_train"].shape[2] != config.input_dim:
        raise ValueError(
            f"input_dim 不一致：stage1={data['X_train'].shape[2]} vs config={config.input_dim}"
        )
    if data["y_train"].shape[1] != config.output_dim:
        raise ValueError(
            f"output_dim 不一致：stage1={data['y_train'].shape[1]} vs config={config.output_dim}"
        )

    print("✅ 第一阶段训练数据读取完成")
    print(f"   X_train: {data['X_train'].shape}")
    print(f"   y_train: {data['y_train'].shape}")
    print(f"   X_val  : {data['X_val'].shape}")
    print(f"   y_val  : {data['y_val'].shape}")
    print(f"   X_test : {data['X_test'].shape}")
    print(f"   y_test : {data['y_test'].shape}")
    return data


def build_dataloaders(config: TrainConfig, arrays: Dict[str, np.ndarray]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(arrays["X_train"]).float(),
        torch.from_numpy(arrays["y_train"]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(arrays["X_val"]).float(),
        torch.from_numpy(arrays["y_val"]).float(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(arrays["X_test"]).float(),
        torch.from_numpy(arrays["y_test"]).float(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, val_loader, test_loader


# ===================== 5. 训练逻辑 =====================
def load_stage2_init_weights(config: TrainConfig, model: nn.Module) -> None:
    ckpt = torch.load(config.stage2_init_weights_path, map_location=config.device)
    if "model_state_dict" not in ckpt:
        raise KeyError("第二阶段初始权重文件缺少 model_state_dict")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("✅ 已加载第二阶段初始权重")


def build_loss(config: TrainConfig) -> nn.Module:
    return nn.MSELoss()


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(config: TrainConfig, optimizer: torch.optim.Optimizer):
    if not config.use_scheduler:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    grad_clip_norm: float,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(xb)
                loss = criterion(pred, yb)

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(total_samples, 1)

    current_lr = 0.0
    if optimizer is not None:
        current_lr = optimizer.param_groups[0]["lr"]
    return avg_loss, current_lr


def train_model(
    config: TrainConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, Any], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    criterion = build_loss(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    no_improve_count = 0
    history: List[Dict[str, Any]] = []

    print("\n🚀 开始训练模型...")
    print(f"   model_name   = {config.model_name}")
    print(f"   pred_mode    = {config.pred_mode}")
    print(f"   feature_mode = {config.feature_mode}")
    print(f"   input_dim    = {config.input_dim}")
    print(f"   output_dim   = {config.output_dim}")
    print(f"   epochs       = {config.max_epochs}")
    print(f"   batch_size   = {config.batch_size}")
    print(f"   lr           = {config.learning_rate}")
    print(f"   weight_decay = {config.weight_decay}")
    print(f"   patience     = {config.patience}")
    print(f"   grad_clip    = {config.grad_clip_norm}")
    print(f"   device       = {config.device}")

    last_state_dict = None

    for epoch in range(1, config.max_epochs + 1):
        train_loss, current_lr = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config.device,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_loss, _ = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=config.device,
            grad_clip_norm=config.grad_clip_norm,
        )

        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

        improved = (best_val_loss - val_loss) > config.min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        last_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val_loss_so_far": float(best_val_loss),
                "improved": bool(improved),
                "no_improve_count": int(no_improve_count),
                "lr": float(current_lr),
            }
        )

        print(
            f"Epoch [{epoch:03d}/{config.max_epochs:03d}] | "
            f"train_loss={train_loss:.8f} | val_loss={val_loss:.8f} | "
            f"best_val={best_val_loss:.8f} | lr={current_lr:.6g} | "
            f"{'*' if improved else ''}"
        )

        if no_improve_count >= config.patience:
            print(f"⏹️ 早停触发：验证集损失连续 {config.patience} 轮未显著改善。")
            break

    if best_state_dict is None:
        raise RuntimeError("训练失败：未获得有效 best_state_dict")
    if last_state_dict is None:
        raise RuntimeError("训练失败：未获得有效 last_state_dict")

    model.load_state_dict(best_state_dict)
    history_df = pd.DataFrame(history)

    best_epoch_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
    train_summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_completed": int(len(history)),
        "early_stopped": bool(len(history) < config.max_epochs),
        "train_loss_at_best_epoch": float(best_epoch_row["train_loss"]),
        "val_loss_at_best_epoch": float(best_epoch_row["val_loss"]),
        "final_train_loss": float(history_df["train_loss"].iloc[-1]),
        "final_val_loss": float(history_df["val_loss"].iloc[-1]),
    }
    return model, history_df, train_summary, best_state_dict, last_state_dict


# ===================== 6. 保存阶段结果 =====================
def save_stage3_outputs(
    config: TrainConfig,
    best_model: nn.Module,
    history_df: pd.DataFrame,
    train_summary: Dict[str, Any],
    best_state_dict: Dict[str, torch.Tensor],
    last_state_dict: Dict[str, torch.Tensor],
) -> None:
    os.makedirs(config.stage3_dir, exist_ok=True)

    best_weights_path = os.path.join(config.stage3_dir, "best_model_weights.pth")
    last_weights_path = os.path.join(config.stage3_dir, "last_model_weights.pth")
    train_log_csv_path = os.path.join(config.stage3_dir, "train_log.csv")
    train_config_json_path = os.path.join(config.stage3_dir, "train_config.json")
    train_summary_json_path = os.path.join(config.stage3_dir, "train_summary.json")

    common_payload = {
        "train_config": asdict(config),
        "summary": train_summary,
    }

    torch.save(
        {
            **common_payload,
            "model_state_dict": best_state_dict,
            "weight_type": "best",
        },
        best_weights_path,
    )

    torch.save(
        {
            **common_payload,
            "model_state_dict": last_state_dict,
            "weight_type": "last",
        },
        last_weights_path,
    )

    history_df.to_csv(train_log_csv_path, index=False, encoding="utf-8-sig")

    config_payload = asdict(config)
    config_payload.update(
        {
            "model_name": config.model_name,
            "history_len": config.history_len,
            "history_len_base": config.history_len_base,
            "effective_history_len": config.effective_history_len,
            "compare_max_lag": config.compare_max_lag,
            "horizon": config.horizon,
            "pred_mode": config.pred_mode,
            "feature_mode": config.feature_mode,
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "input_feature_names": config.input_feature_names,
            "lag_list": config.lag_list,
            "max_lag": config.max_lag,
        }
    )

    with open(train_config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=4, ensure_ascii=False)
    with open(train_summary_json_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=4, ensure_ascii=False)

    print(f"\n📁 第三阶段结果已保存至：{config.stage3_dir}")
    print("保存文件说明：")
    print("- best_model_weights.pth: 验证集最优模型权重（第四阶段测试直接读取）")
    print("- last_model_weights.pth: 最后一轮训练结束权重（用于对比保留）")
    print("- train_log.csv: 每轮训练/验证损失日志")
    print("- train_config.json: 第三阶段训练配置")
    print("- train_summary.json: 最优epoch与早停摘要")


# ===================== 7. 主流程 =====================
def main() -> None:
    config = TrainConfig()
    set_random_seed(config.random_seed)

    arrays = load_stage1_arrays(config)
    train_loader, val_loader, test_loader = build_dataloaders(config, arrays)
    print(f"\n📦 DataLoader构建完成：train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    model = build_model(config)
    load_stage2_init_weights(config, model)

    best_model, history_df, train_summary, best_state_dict, last_state_dict = train_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    save_stage3_outputs(
        config=config,
        best_model=best_model,
        history_df=history_df,
        train_summary=train_summary,
        best_state_dict=best_state_dict,
        last_state_dict=last_state_dict,
    )

    print("\n🎉 第三阶段重构版运行完成！")
    print(f"   最优epoch      = {train_summary['best_epoch']}")
    print(f"   最优val_loss   = {train_summary['best_val_loss']:.8f}")
    print(f"   实际完成轮数   = {train_summary['epochs_completed']}")
    print(f"   是否早停       = {train_summary['early_stopped']}")
    print(f"   最终train_loss = {train_summary['final_train_loss']:.8f}")
    print(f"   最终val_loss   = {train_summary['final_val_loss']:.8f}")


if __name__ == "__main__":
    main()