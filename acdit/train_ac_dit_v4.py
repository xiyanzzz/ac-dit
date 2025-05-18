import os
import sys
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import math

# 设置环境变量，避免HuggingFace tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AC_DiT.models.acdit_future_pred_runner import ACDiTFuturePredRunner, create_acdit_future_pred_runner_from_config
from AC_DiT.datasets.sequence_dataset_v3 import CompactSequenceDataModule


class ACDiTFuturePredLightningModule(pl.LightningModule):
    """带未来图像预测功能的PyTorch Lightning模块，用于训练和评估AC_DiT模型"""
    def __init__(
        self,
        runner: ACDiTFuturePredRunner,
        train_samples_per_log: int = 10,
        val_samples_per_log: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.95),
        warmup_steps: int = 1000,
        constant_steps: int = 5000,
        decay_steps: int = 10000
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['runner'])
        self.runner = runner
        self.train_samples_per_log = train_samples_per_log
        self.val_samples_per_log = val_samples_per_log
        
        # 优化器参数
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps
        
        # 记录最佳验证损失
        self.best_val_loss = float('inf')
        self.best_img_pred_loss = float('inf')
        self.best_sample_loss = float('inf')

    
    def forward(self, batch):
        return self.runner(batch)
    
    def prepare_batch(self, batch):
        """准备批次数据，确保数据结构一致
        
        Args:
            batch: 原始批次数据
        
        Returns:
            prepared_batch: 准备好的批次数据，包含所需的所有键
        """
        prepared_batch = {}
        
        # 确保批次包含必要的键
        if "rgb_obs" not in batch or "rgb_static" not in batch["rgb_obs"] or "rgb_gripper" not in batch["rgb_obs"]:
            raise ValueError("批次数据缺少必要的RGB观察")
        
        # 对原始批次进行浅拷贝
        prepared_batch["rgb_obs"] = batch["rgb_obs"]
        
        # 处理动作数据
        if "input_actions" in batch:
            prepared_batch["input_actions"] = batch["input_actions"]
        else:
            raise ValueError("批次数据缺少输入动作:'input_actions'")
        
        # 检查是否有单独的目标动作
        if "target_actions" in batch:
            prepared_batch["target_actions"] = batch["target_actions"]
        else:
            raise ValueError("批次数据缺少目标动作:'target_actions'")
            
        # 处理语言数据
        if "lang_text" in batch:
            prepared_batch["lang_text"] = batch["lang_text"]
        else:
            raise ValueError("批次数据缺少语言文本:'lang_text'")
        
        # 处理未来图像数据，如果有的话
        if "future_rgb_static" in batch and "future_rgb_gripper" in batch and "future_step" in batch:
            prepared_batch["future_rgb_static"] = batch["future_rgb_static"]
            prepared_batch["future_rgb_gripper"] = batch["future_rgb_gripper"]
            prepared_batch["future_step"] = batch["future_step"]
        
        # 保存样本索引，如果有的话
        if "idx" in batch:
            prepared_batch["idx"] = batch["idx"]
        
        return prepared_batch
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 准备批次数据
        batch = self.prepare_batch(batch)
        
        # 计算损失
        losses = self.runner.compute_loss(batch)
        loss = losses['total_loss']
        
        # 记录训练损失
        self.log('train/action_loss', losses['action_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["input_actions"].shape[0])
        
        # 如果有图像预测损失，也记录下来
        if 'img_pred_loss' in losses:
            self.log('train/img_pred_loss', losses['img_pred_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["input_actions"].shape[0])
            self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["input_actions"].shape[0])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 准备批次数据
        batch = self.prepare_batch(batch)
        
        # 计算损失
        losses = self.runner.compute_loss(batch)
        loss = losses['total_loss']
        action_loss = losses['action_loss']
        
        # 记录验证损失
        batch_size = batch["input_actions"].shape[0]
        self.log('val_act/action_loss', action_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        # 更新最佳验证损失
        if self.best_val_loss > action_loss:
            self.best_val_loss = action_loss
        self.log('val_act/best_loss', self.best_val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        # 如果有图像预测损失，也记录下来
        if 'img_pred_loss' in losses:
            img_pred_loss = losses['img_pred_loss']
            self.log('val_img/img_pred_loss', img_pred_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log('val_total/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            
            # 更新最佳图像预测损失
            if self.best_img_pred_loss > img_pred_loss:
                self.best_img_pred_loss = img_pred_loss
            self.log('val_img/best_loss', self.best_img_pred_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        # 新增计算预测动作和真实动作的MSE，作为checkpoint的monitor
        # 用runner采样动作
        with torch.no_grad():
            sample_actions = self.runner.predict_actions(
                rgb_observations=batch["rgb_obs"],
                language_goal=batch["lang_text"] if "lang_text" in batch else batch["lang"],
                past_actions=batch["input_actions"]
            )
        # 计算采样动作和真实动作的MSE
        sample_mse = F.mse_loss(sample_actions, batch["target_actions"]).item()
        self.log('val_sample/sample_mse', sample_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # 更新最佳动作采样损失
        if self.best_sample_loss > sample_mse:
            self.best_sample_loss = sample_mse
        self.log('val_sample/best_sample_loss', self.best_sample_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # 如果是第一个批次，可视化一些样本
        # 若1个训练epoch验证3次, 则总数为 训练epoch * 3
        if batch_idx == 0 and hasattr(self.logger, "experiment") and self.logger.experiment is not None:
            try:
                # 已全局记录
                # # 用runner预测动作
                # with torch.no_grad():
                #     pred_actions = self.runner.predict_actions(
                #         rgb_observations=batch["rgb_obs"],
                #         language_goal=batch["lang_text"] if "lang_text" in batch else batch["lang"],
                #         past_actions=batch["input_actions"]
                #     )
                
                # # 计算预测动作和真实动作的MSE
                # pred_mse = F.mse_loss(pred_actions, batch["target_actions"]).item()
                # self.log('val_act/pred_mse', pred_mse, on_epoch=True, logger=True, batch_size=batch_size)
                
                # 如果runner支持未来图像预测，也可视化预测的图像
                if self.runner.use_future_prediction and "future_rgb_static" in batch:              
                    language_ann = batch["lang_text"][0]
                    future_step = batch["future_step"][0].item()  # 取第一个样本的时间步
                    future_images = self.runner.predict_future_images(
                        rgb_observations=batch["rgb_obs"],
                        language_goal=batch["lang_text"] if "lang_text" in batch else batch["lang"],
                        future_step=torch.tensor([future_step], device=self.device)
                    ) # [B, N_c, 3, H, W]
                     
                    # 保存图像到本地
                    # 创建保存目录
                    save_dir = Path("./saved_images")
                    save_dir.mkdir(exist_ok=True, parents=True)
                    
                    # 获取当前步骤（全局步骤或当前epoch）
                    current_epoch = self.current_epoch
                    global_step = self.global_step
                    
                    try:
                        # 检查并预处理图像数据
                        # 确保图像数据是正确的格式和数据类型
                        # print(f"future_images shape: {future_images.shape}, dtype: {future_images.dtype}") # [32, 2, 3, 144, 144]
                        # print(f"future_rgb_static shape: {batch['future_rgb_static'].shape}, dtype: {batch['future_rgb_static'].dtype}") # [32, 3, 144, 144]
                        # print(f"future_rgb_gripper shape: {batch['future_rgb_gripper'].shape}, dtype: {batch['future_rgb_gripper'].dtype}")
                        # print(f"rgb_obs_gripper shape: {batch['rgb_obs']['rgb_gripper'].shape}, dtype: {batch['rgb_obs']['rgb_gripper'].dtype}")
                        # 转换为float32以确保兼容性
                        static_curr = denormalize(batch["rgb_obs"]["rgb_static"][0,-1].detach()).permute(1, 2, 0).cpu().numpy() # 历史帧的最后一帧
                        static_pred = denormalize(future_images[0, 0].detach()).permute(1, 2, 0).cpu().numpy()
                        static_true = denormalize(batch["future_rgb_static"][0].detach()).permute(1, 2, 0).cpu().numpy()

                        gripper_curr = denormalize(batch["rgb_obs"]["rgb_gripper"][0,-1].detach()).permute(1, 2, 0).cpu().numpy()
                        gripper_pred = denormalize(future_images[0, 1].detach()).permute(1, 2, 0).cpu().numpy()
                        gripper_true = denormalize(batch["future_rgb_gripper"][0].detach()).permute(1, 2, 0).cpu().numpy()
                        
                        
                        # 使用matplotlib保存对比图
                        static_images = [static_curr, static_pred, static_true]
                        gripper_images = [gripper_curr, gripper_pred, gripper_true]
                        
                        # 保存所有图像在一起的对比图
                        all_images = static_images + gripper_images
                        all_titles = ["static_curr", "static_pred", "static_true", "gripper_curr", "gripper_pred", "gripper_true"]
                        all_save_path = save_dir / f"e{current_epoch}_s{global_step}_t{future_step}.png"
                        save_images(all_images, all_titles, all_save_path, language_ann)
                    except Exception as e:
                        print(f"准备图像数据时出错: {e}")
                        import traceback
                        traceback.print_exc()

                    # # 这里可以添加可视化代码，如使用wandb.Image等
                    # if isinstance(self.logger, WandbLogger) and 'wandb' in sys.modules:
                    #     import wandb
                    #     # 记录第一个样本的预测图像和真实图像
                    #     # 处理静态相机图像
                    #     # pred_static = future_images[0, 0].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                    #     # true_static = batch["future_rgb_static"][0].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                        
                    #     # # 处理抓取器相机图像
                    #     # pred_gripper = future_images[0, 1].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                    #     # true_gripper = batch["future_rgb_gripper"][0].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                        
                    #     # 记录到wandb
                    #     self.logger.experiment.log({
                    #         "val_img/static_camera_pred": wandb.Image(static_pred, caption=f"pred t+{future_step}, static"),
                    #         "val_img/static_camera_true": wandb.Image(static_true, caption=f"true t+{future_step}, static"),
                    #         "val_img/gripper_camera_pred": wandb.Image(gripper_pred, caption=f"pred t+{future_step}, gripper"),
                    #         "val_img/gripper_camera_true": wandb.Image(gripper_true, caption=f"true t+{future_step}, gripper"),
                    #     })
                
            except Exception as e:
                print(f"可视化验证样本时出错: {e}")
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 使用runner的configure_optimizers方法获取优化器
        optimizer = self.runner.configure_optimizers(
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas
        )
        
        # 创建学习率调度器
        def lr_lambda(step):
            # 热身阶段
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            # 稳定阶段
            elif step < self.warmup_steps + self.constant_steps:
                return 1.0
            # 衰减阶段
            else:
                decay_step = step - self.warmup_steps - self.constant_steps
                # return max(5e-7, 1.0 - decay_step / float(max(1, self.decay_steps)))
                # 使用余弦衰减
                min_lr_factor = 5e-6
                return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + math.cos(math.pi * min(1.0, decay_step / self.decay_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def modify_cmd_for_gpus(config):
    """
    修改命令行参数以支持多GPU训练
    """
    # 确保加载器工作进程数量是合理的
    if 'devices' in config.trainer and isinstance(config.trainer.devices, list) and len(config.trainer.devices) > 1:
        # 多GPU训练
        original_batch_size = config.data.batch_size
        gpu_count = len(config.trainer.devices)
        config.data.batch_size = original_batch_size // gpu_count
        config.trainer.strategy = 'ddp'
        
        # 根据GPU数量调整学习率
        if 'learning_rate' in config.optimizer:
            config.optimizer.learning_rate *= gpu_count
    
    return config

import matplotlib.pyplot as plt
def save_images(images, titles, save_path, language_ann):
    """显示一系列图像并可选择保存到指定路径
    
    Args:
        images: 图像列表，可以是numpy数组或PyTorch张量
        titles: 每个图像的标题列表
        save_path: 保存图像的路径，如果为None则不保存
    """
    # 计算合适的子图布局
    if len(images) == 6:
        n_cols = 3 # len(images)
        n_rows = 2
    else:
        raise ValueError(f"图像数量必须为6，当前数量为{len(images)}")
    
    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    
    # 确保axes是一个平坦的数组
    axes = np.array(axes).flatten()
    
    # 在每个子图中显示相应的图像
    for i, img in enumerate(images):
        if i >= len(axes):
            break
        # 显示图像
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB图像
            # # 确保值在合理范围内
            # if np.max(img) > 1.0 and np.max(img) <= 255.0:
            #     img = img / 255.0
            # axes[i].imshow(np.clip(img, 0, 1))
            axes[i].imshow(img)
        else:
            print(f"警告: 无法显示形状为 {img.shape} 的图像 {i}")
            # 尝试显示任何方式
            raise ValueError(f"图像 {i} 的形状不是3维: {img.shape}")
            
        # 设置标题和隐藏坐标轴
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    
    # 隐藏多余的子图
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 顶部留出5%的空间给标题
    plt.suptitle(language_ann, fontsize=16, fontweight='bold')
    
    # 如果指定了保存路径，则保存图像
    if save_path:
        try:
            # 确保保存目录存在
            save_dir = Path(save_path).parent
            save_dir.mkdir(exist_ok=True, parents=True)
            
            # 保存图像
            plt.savefig(save_path)
            print(f"图像已保存到: {save_path}")
        except Exception as e:
            print(f"保存图像时出错: {e}")
    else:
        raise ValueError("save_path is None")
    # 关闭图像以释放内存
    plt.close(fig)


def denormalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]):
    """
    将归一化后的图像恢复到原始的[0,1]范围
    
    Args:
        tensor (torch.Tensor): 归一化后的图像张量，形状为[C,H,W]
        mean: 归一化时使用的均值
        std: 归一化时使用的标准差
    """
    # 创建与输入tensor相同设备的mean和std张量
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    
    # 执行逆归一化: x = (normalized * std) + mean
    return (tensor * std + mean).clip(0, 1)


@hydra.main(config_path="configs", config_name="ac_dit_sequence_v5")
def main(config: DictConfig) -> None:
    """
    主函数，加载数据，创建模型，设置训练器并开始训练
    """
    print("配置:")
    print(OmegaConf.to_yaml(config))
    
    # 设置随机种子
    seed_everything(config.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建数据模块
    data_module = hydra.utils.instantiate(config.data)
    
    # 修改配置以适应多GPU训练
    config = modify_cmd_for_gpus(config)
    
    # 检查检查点并决定是加载还是创建新模型
    checkpoint_path = None
    if config.trainer.callbacks:
        for callback in config.trainer.callbacks:
            if "_target_" in callback and "ModelCheckpoint" in callback._target_:
                checkpoint_dir = Path(callback.get("dirpath", "checkpoints"))
                if checkpoint_dir.exists():
                    checkpoint_files = sorted(list(checkpoint_dir.glob("*.ckpt")))
                    if checkpoint_files and hasattr(config, "load_from_checkpoint") and config.load_from_checkpoint:
                        checkpoint_path = checkpoint_files[-1]
                        print(f"加载检查点: {checkpoint_path}")
    
    # 创建模型
    if checkpoint_path is not None:
        # 从检查点加载模型
        model = ACDiTFuturePredLightningModule.load_from_checkpoint(
            checkpoint_path,
            runner=create_acdit_future_pred_runner_from_config(config)
        )
    else:
        # 创建新模型
        runner = create_acdit_future_pred_runner_from_config(config)
        model = ACDiTFuturePredLightningModule(
            runner=runner,
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.transformer_weight_decay,
            betas=tuple(config.optimizer.betas),
            warmup_steps=config.lr_scheduler.warmup_steps,
            constant_steps=config.lr_scheduler.constant_steps,
            decay_steps=config.lr_scheduler.decay_steps
        )
    
    # 设置日志器
    logger = []
    if "logger" in config and config.logger:
        # 创建WandbLogger
        if "_target_" in config.logger and "WandbLogger" in config.logger._target_:
            wandb_logger = hydra.utils.instantiate(config.logger)
            wandb_logger.watch(model)
            logger.append(wandb_logger)
        # 创建TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir="logs/",
            name="tensorboard",
            version=config.logger.get("id", None)
        )
        logger.append(tb_logger)
    
    # 设置回调
    callbacks = []
    if config.trainer.callbacks:
        for callback_conf in config.trainer.callbacks:
            if "_target_" in callback_conf:
                callbacks.append(hydra.utils.instantiate(callback_conf))
    
    # 创建训练器
    trainer = pl.Trainer(
        **{k: v for k, v in config.trainer.items() if k != "callbacks"},
        callbacks=callbacks,
        logger=logger
    )
    
    # 开始训练
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main() 