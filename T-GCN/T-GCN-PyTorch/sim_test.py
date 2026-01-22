import argparse
import torch
import pytorch_lightning as pl
import models
import tasks
import utils.data
import utils.data.data_nyc
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
import os

# 数据路径配置（与 main 保持一致）
DATA_PATHS = {
    "sim": {
        "feat": "data/sim/test/springs5_interp_xy_norm.pkl", 
        "adj": "data/sim/test/edges_test_springs5.npy"
    },
}

def run_test(args):
    # 1. 初始化 DataModule
    dm = utils.data.data_sim.SimSpatioTemporalPklDataModule(
        feat_path=DATA_PATHS[args.data]["feat"],
        edges_path=DATA_PATHS[args.data]["adj"],
        **vars(args)
    )
    # 调用 setup 确保 dm.adj 等属性被初始化
    dm.setup(stage="test")

    # 2. 构造基础模型架构（用于加载权重）
    # 注意：这里需要先根据 dm 确定架构维度
    if args.model_name == "GCN":
        base_model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    elif args.model_name == "GRU":
        base_model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    elif args.model_name == "TGCN":
        base_model = models.TGCN(num_nodes=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # 3. 从 ckpt 加载 Task 状态
    # 这一步会自动把权重填充进 base_model 并初始化 Task 的 hparams
    print(f"正在从加载权重: {args.ckpt_path}")
    task = tasks.SimSupervisedForecastTask.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        model=base_model,
        # 如果训练和测试时的 pre_len 等参数不一致，可以在此处覆盖
        # pre_len=args.pre_len 
    )

    # 4. 初始化 Trainer
    logger = TensorBoardLogger("lightning_logs", name=f"test_{args.data}")
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger
    )

    # 5. 执行验证/测试
    # validate 通常用于查看验证集指标，test 用于查看测试集指标
    print("开始在验证集上进行评估...")
    # val_results = trainer.validate(task, datamodule=dm)
    task.eval()
    all_preds = []
    all_y = []
    all_masks = []
    
    with torch.no_grad():
        for batch in dm.test_dataloader():
            # 移动数据到设备
            batch = [b.to(task.device) if torch.is_tensor(b) else b for b in batch]
            # 调用模型的 validation_step 逻辑
            pred, y = task.validation_step(batch, 0)
            _, _, _, y_mask = batch
            # 这里的形状通常是 (Batch, pre_len, Nodes)
            # 我们取每个 batch 的第一个时间步进行拼接
            all_preds.append(pred.cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_masks.append(y_mask.cpu().numpy())
    # 拼接所有 Batch: (Total_Samples, pre_len, Nodes)
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    if all_masks.ndim == 4:
        all_masks = np.squeeze(all_masks, axis=-1)
    # print(all_preds.shape, all_y.shape, all_masks.shape)
    # 转换形状: (Total_Samples * Nodes, pre_len)
    # 先转置为 (Total_Samples, Nodes, pre_len) 再展平前两维
    all_preds = all_preds.transpose(0, 2, 1).reshape(-1, args.pre_len)
    all_y = all_y.transpose(0, 2, 1).reshape(-1, args.pre_len)
    all_masks = all_masks.transpose(0, 2, 1).reshape(-1, args.pre_len)
    # 计算带掩码的指标
    valid_idx = all_masks > 0
    if np.any(valid_idx):
        mse = np.mean((all_preds[valid_idx] - all_y[valid_idx]) ** 2)
        mae = np.mean(np.abs(all_preds[valid_idx] - all_y[valid_idx]))
    else:
        mse, mae = 0, 0, 0
    
    print("\n" + "="*30)
    print(f"测试集评估指标:")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"预测结果形状: {all_preds.shape}")
    print(f"真实结果形状: {all_y.shape}")
    
        # 7. 绘图部分：每隔 1000 个样本点保存一张对比图
    save_dir = f"vis_results_{args.data}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n开始绘制对比图，结果将保存至: {save_dir}")
    interval = 1000
    for i in range(0, len(all_preds), interval):
        plt.figure(figsize=(10, 5))
        plt.plot(all_y[i], label="Ground Truth", color="blue", linestyle="--", marker="o")
        plt.plot(all_preds[i], label="Prediction", color="red", marker="x")
        
        plt.title(f"Prediction vs Ground Truth (Sample Index: {i})")
        plt.xlabel("Time Step (Future)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plt.savefig(os.path.join(save_dir, f"compare_sample_{i}.png"))
        plt.close() # 释放内存
        
    print(f"绘图完成，共保存 {len(range(0, len(all_preds), interval))} 张图片。")
    
    return {"mae": mae, "mse": mse}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必须提供的参数
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint 文件的路径", default="lightning_logs/sim/version_2/checkpoints/epoch=45-step=46000.ckpt")
    parser.add_argument("--data", type=str, default="sim", choices=("sim"))
    parser.add_argument("--model_name", type=str, default="TGCN", choices=("GCN", "GRU", "TGCN"))
    
    # 环境参数
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # 以下参数通常需要与训练时保持一致，以便正确构建模型结构
    parser.add_argument("--pre_len", type=int, default=59)
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)

    args = parser.parse_args()
    
    results = run_test(args)
    print("\n测试结果:")
    print(results)