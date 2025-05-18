import logging
import os
import math  # 添加math模块引用，因为在make_sample_density方法中需要使用
from typing import Any, Dict, Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import einops
import wandb

# 导入本地工具模块，替换mdt中的依赖
from AC_DiT.utils.edm_diffusion.gc_sampling import get_sigmas_karras, get_sigmas_exponential, get_sigmas_linear, sample_lms, sample_ddim, sample_euler
from AC_DiT.utils.edm_diffusion.utils import append_dims, rand_log_logistic, rand_log_uniform
from AC_DiT.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from AC_DiT.callbacks.ema import EMA

# 导入AC_DiT特定的模型组件
from AC_DiT.models.model.AC_DiT import AC_DiT
from AC_DiT.models.multimodal_encoder.vision_encoder_with_tokens import MVT_TokenFusion_Encoder
from AC_DiT.models.multimodal_encoder.text_encoder import TextEncoder

logger = logging.getLogger(__name__)


class ACDiTAgent(pl.LightningModule):
    """
    AC_DiT智能体，用于训练和推理
    """
    def __init__(
        self,
        vision_encoder: DictConfig,
        language_encoder: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        latent_dim: int = 512,
        multistep: int = 10,
        sampler_type: str = 'ddim',
        num_sampling_steps: int = 10,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80,
        noise_scheduler: str = 'exponential',
        sigma_sample_density_type: str = 'loglogistic',
        use_lr_scheduler: bool = True,
        act_window_size: int = 10,
        action_dim: int = 7,
        ckpt_path=None,
        seed: int = 42,
    ):
        super(ACDiTAgent, self).__init__()
        
        # 基本设置
        self.latent_dim = latent_dim
        self.act_window_size = act_window_size
        self.action_dim = action_dim
        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        
        # 编码器模型
        self.vision_encoder = hydra.utils.instantiate(vision_encoder)
        self.language_encoder = hydra.utils.instantiate(language_encoder) if language_encoder else None
        
        # 动作生成模型
        self.ac_dit_model = hydra.utils.instantiate(model, 
                                                   action_dim=action_dim, 
                                                   hidden_dim=latent_dim,
                                                   pred_horizon=act_window_size)
        
        # 优化器和学习率调度器配置
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        
        # 扩散模型参数
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        
        # 推理相关
        self.rollout_step_counter = 0
        self.multistep = multistep
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 加载预训练权重（如果有）
        if ckpt_path is not None:
            self.load_pretrained_parameters(ckpt_path)
    
    def load_pretrained_parameters(self, ckpt_path):
        """加载预训练参数"""
        print("正在加载预训练参数")
        checkpoint_data = torch.load(ckpt_path)
        
        if "ema_weights" in checkpoint_data.get('callbacks', {}).get('EMA', {}):
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(self.named_parameters())}
            self.load_state_dict(ema_weights_dict)
            print("成功从检查点加载EMA权重!")
        else:
            self.load_state_dict(checkpoint_data['state_dict'])
        print("成功从检查点加载权重!")
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 主模型参数组
        optim_groups = [
            {"params": self.ac_dit_model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]
        
        # 编码器参数组
        optim_groups.extend([
            {"params": self.vision_encoder.parameters(), "weight_decay": self.optimizer_config.encoder_weight_decay},
        ])
        
        # 注意：不添加文本编码器的参数，因为它们是冻结的
        # 如果需要在未来解冻文本编码器进行微调，可以有条件地添加
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=self.optimizer_config.learning_rate, 
            betas=self.optimizer_config.betas
        )
        
        # 创建学习率调度器
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler_config)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer
    
    def compute_input_embeddings(self, dataset_batch):
        """计算输入嵌入"""
        # 提取视觉观察
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'].unsqueeze(1)  # 形状：[B, 1, T_o, C, H, W]
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'].unsqueeze(1)  # 形状：[B, 1, T_o, C, H, W]
        
        # 确保数据中包含语言标注
        if "lang" not in dataset_batch:
            raise ValueError("每个样本必须包含语言标注！")
            
        # 处理语言目标
        if isinstance(self.language_encoder, TextEncoder):
            # 处理自定义文本编码器
            if "lang_text" in dataset_batch:
                # 如果有原始文本，直接使用原始文本
                batch_texts = dataset_batch["lang_text"]
                # 对批次中的每个文本进行编码
                latent_goals = []
                for text in batch_texts:
                    text_embed = self.language_encoder.encode(text)
                    latent_goals.append(text_embed)
                latent_goal = torch.cat(latent_goals, dim=0)
            else:
                # 如果没有原始文本，使用预计算的嵌入
                raise ValueError("未发现目标文本！")
                # latent_goal = dataset_batch["lang"]
            
            # 调整维度以匹配期望的输出
            if len(latent_goal.shape) == 2:
                latent_goal = latent_goal.to(self.device).to(rgb_static.dtype)
        else:
            # 处理其他语言编码器
            raise NotImplementedError("语言编码器未实现！")
            latent_goal = self.language_encoder(dataset_batch["lang"]).to(rgb_static.dtype)
        
        # 重塑视觉输入以匹配MVT_TokenFusion_Encoder的期望
        # 期望形状：[B, N_c, T, C, H, W]，其中N_c=2（static和gripper），T=2（时间步）
        B, _, T, C, H, W = rgb_static.shape
        
        # 创建包含两个相机视图的张量(static和gripper)
        visual_input = torch.cat([rgb_static, rgb_gripper], dim=1, dtype=rgb_static.dtype)
        
        # 编码视觉观察，使用语言条件
        visual_tokens = self.vision_encoder(
            visual_input,  # 形状：[B, 2, T, C, H, W]
            latent_goal  # 语言嵌入作为条件
        )
        
        return visual_tokens, latent_goal
    
    def training_step(self, batch: Dict[str, Dict], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """训练步骤"""
        total_loss = torch.tensor(0.0).to(self.device)
        action_loss = torch.tensor(0.0).to(self.device)
        total_bs = 0
        
        # 处理批次中的每个样本
        for key, dataset_batch in batch.items():
            # 计算输入嵌入
            perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)
            
            # 计算扩散损失
            act_loss, _, _ = self.diffusion_loss(
                perceptual_emb,
                latent_goal,
                dataset_batch["actions"],
            )
            
            action_loss += act_loss
            total_loss += act_loss
            total_bs += dataset_batch["actions"].shape[0]
        
        # 平均损失
        batch_len = len(batch)
        total_loss = total_loss / batch_len
        action_loss = action_loss / batch_len
        
        # 记录指标，使用安全方式检查是否在训练环境中
        try:
            self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
            self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        except (RuntimeError, ValueError, AttributeError):
            # 在测试环境中可能没有trainer，忽略日志记录错误
            pass
        
        return total_loss
    
    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        output = {}
        val_total_act_loss = torch.tensor(0.0).to(self.device)
        
        # 处理批次中的每个样本
        for key, dataset_batch in batch.items():
            # 计算输入嵌入
            perceptual_emb, latent_goal, _ = self.compute_input_embeddings(dataset_batch)
            
            # 预测下一个动作序列
            action_pred = self.denoise_actions(
                torch.zeros_like(latent_goal).to(latent_goal.device),
                perceptual_emb,
                latent_goal,
                inference=True,
            )
            
            # 计算MSE动作损失
            pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
            val_total_act_loss += pred_loss
            
            # 记录验证指标，使用安全方式检查是否在训练环境中
            try:
                self.log(f"val_act/{key}_act_loss", pred_loss, sync_dist=True)
            except (RuntimeError, ValueError, AttributeError):
                # 在测试环境中可能没有trainer，忽略日志记录错误
                pass
            
            output[f"idx_{key}"] = dataset_batch["idx"]
            
        # 记录总验证损失，使用安全方式检查是否在训练环境中
        try:
            self.log("val_act/action_loss", val_total_act_loss / len(batch), sync_dist=True)
        except (RuntimeError, ValueError, AttributeError):
            # 在测试环境中可能没有trainer，忽略日志记录错误
            pass
        
        output["validation_loss"] = val_total_act_loss
        
        return output
    
    def diffusion_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算扩散损失"""
        # 将模型设为训练模式
        self.ac_dit_model.train()
        
        # 采样噪声水平
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        
        # 生成噪声
        noise = torch.randn_like(actions).to(self.device)
        
        # 添加噪声到动作
        noised_actions = actions + noise * append_dims(sigmas, actions.ndim)
        
        # 预测噪声
        # AC_DiT模型的forward方法需要5个参数：x, x_cond, y, t, g
        # x: 噪声动作 (noised_actions)
        # x_cond: 条件动作 (actions[:, :2])
        # y: 语言目标 (latent_goal)
        # t: 时间步 (sigmas)
        # g: 视觉表征 (perceptual_emb)
        pred_noise = self.ac_dit_model(
            noised_actions,          # x: 噪声动作
            actions[:, :2],          # x_cond: 条件动作（前2个时间步）
            latent_goal,             # y: 语言目标
            sigmas,                  # t: 扩散时间步
            perceptual_emb           # g: 视觉表征
        )
        
        # 计算噪声预测损失
        loss = F.mse_loss(pred_noise, noise)
        
        return loss, sigmas, noise
    
    def denoise_actions(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        inference: Optional[bool] = False,
        extra_args={}
    ) -> torch.Tensor:
        """去噪生成动作序列"""
        # 设置采样步数
        sampling_steps = self.num_sampling_steps if inference else 10
        
        # 设为评估模式
        self.ac_dit_model.eval()
        
        # 生成噪声调度
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        
        # 初始化噪声输入 - 应该是3D张量 (B, T_a, d_action)
        x = torch.randn((len(latent_goal), self.act_window_size, self.action_dim), device=self.device) * self.sigma_max
        
        # 确保latent_plan形状是(B, T_o, d_action)，T_o默认为2
        if latent_plan is None:
            # 如果没有提供latent_plan，创建全零张量
            conditioned_plan = torch.zeros((len(latent_goal), 2, self.action_dim), device=self.device)
        else:
            # 如果提供了latent_plan但形状不对，进行调整
            if len(latent_plan.shape) == 2 and latent_plan.shape[1] == self.action_dim:
                # (B, d_action) -> (B, 1, d_action)
                conditioned_plan = latent_plan.unsqueeze(1)
                # 如果需要2个时间步，复制一次
                if conditioned_plan.shape[1] < 2:
                    conditioned_plan = conditioned_plan.expand(-1, 2, -1)
            elif len(latent_plan.shape) == 3:
                # 已经是3D，但可能需要调整时间步数
                if latent_plan.shape[1] != 2:
                    # 创建新的全零张量
                    conditioned_plan = torch.zeros((len(latent_goal), 2, self.action_dim), device=latent_plan.device, dtype=latent_plan.dtype)
                    # 复制可用的时间步
                    copy_steps = min(latent_plan.shape[1], 2)
                    conditioned_plan[:, :copy_steps] = latent_plan[:, :copy_steps]
                else:
                    conditioned_plan = latent_plan
            else:
                # 其他情况，创建全零张量
                conditioned_plan = torch.zeros((len(latent_goal), 2, self.action_dim), device=self.device)
        
        # 运行采样循环
        actions = self.sample_loop(sigmas, x, perceptual_emb, latent_goal, conditioned_plan, self.sampler_type, extra_args)
        
        return actions
    
    def make_sample_density(self):
        """创建噪声水平采样密度函数"""
        if self.sigma_sample_density_type == 'loglogistic':
            loc = math.log(self.sigma_data)
            scale = 0.5
            return partial(rand_log_logistic, loc=loc, scale=scale, 
                          min_value=self.sigma_min, max_value=self.sigma_max)
        
        elif self.sigma_sample_density_type == 'loguniform':
            return partial(rand_log_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        else:
            raise ValueError('未知的采样密度类型')
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """获取噪声调度"""
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7, self.device)
        
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        
        else:
            raise ValueError('未知的噪声调度类型')
    
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        perceptual_emb: torch.Tensor, 
        goal: torch.Tensor, 
        latent_plan: torch.Tensor,
        sampler_type: str,
        extra_args={}, 
    ):
        """采样循环
        
        参数:
            sigmas: 噪声水平序列
            x_t: 初始噪声动作，形状(B, T_a, d_action)
            perceptual_emb: 视觉表征，形状(B, T_g, D)
            goal: 语言目标，形状(B, d_clip)
            latent_plan: 条件动作，形状(B, T_o, d_action)
            sampler_type: 采样器类型，'ddim'或'普通'
            extra_args: 额外参数
        """
        # 将视觉表征作为模型的g输入参数
        extra_args_with_g = {'g': perceptual_emb}
        
        # 根据采样器类型选择采样方法
        if sampler_type == 'ddim':
            # 使用DDIM采样器
            x_0 = sample_ddim(
                model=self.ac_dit_model,     # 模型
                state=latent_plan,           # 条件动作，形状(B, T_o, d_action)
                action=x_t,                  # 噪声动作，形状(B, T_a, d_action)
                goal=goal,                   # 语言目标，形状(B, d_clip)
                sigmas=sigmas,               # 噪声水平
                disable=True,                # 禁用进度条
                extra_args=extra_args_with_g # 额外参数，包含视觉表征
            )
        else:
            # 使用普通采样器（默认）
            # 这里使用简化的采样逻辑，适合非分数扩散模型
            x_0 = self._simple_sampling(
                latent_plan=latent_plan,
                x_t=x_t,
                goal=goal,
                perceptual_emb=perceptual_emb,
                sigmas=sigmas
            )
        
        return x_0
    
    def _simple_sampling(self, latent_plan, x_t, goal, perceptual_emb, sigmas):
        """简化的采样方法，适合非分数扩散模型"""
        # 初始化当前噪声动作
        x = x_t
        
        # 遍历噪声水平，从高噪声到低噪声
        for i in range(len(sigmas) - 1):
            # 当前噪声水平
            sigma = sigmas[i]
            # 下一个噪声水平
            next_sigma = sigmas[i + 1]
            
            # 为当前批次创建噪声水平输入
            s_in = torch.ones([x.shape[0]], device=x.device) * sigma
            
            # 使用模型预测去噪后的动作
            denoised = self.ac_dit_model(
                x=x,                 # 噪声动作
                x_cond=latent_plan,  # 条件动作
                y=goal,              # 语言目标
                t=s_in,              # 当前噪声水平
                g=perceptual_emb     # 视觉表征
            )
            
            # 线性插值到下一个噪声水平
            # 这是一个简化的去噪过程，类似于DDIM但更简单
            alpha = next_sigma / sigma
            x = alpha * x + (1 - alpha) * denoised
            
        return x
    
    def reset(self):
        """在推理时开始新的rollout时调用"""
        self.rollout_step_counter = 0
    
    def forward(self, obs, goal):
        """推理时的前向传播"""
        # 确保输入包含语言标注
        if "lang" not in goal:
            raise ValueError("必须提供语言目标！每个样本必须包含语言标注！")
        
        # 处理语言编码
        if isinstance(self.language_encoder, TextEncoder):
            # 使用自定义文本编码器
            if "lang_text" in goal:
                # 如果有原始文本
                text = goal["lang_text"]
                latent_goal = self.language_encoder.encode(text).to(self.device)
            else:
                # 如果有预计算的嵌入
                latent_goal = goal["lang"].to(self.device)
        else:
            # 使用其他语言编码器
            latent_goal = self.language_encoder(goal["lang"])
        
        # 提取观察
        rgb_static = obs["rgb_obs"]['rgb_static']  # 形状：[B, T+1, C, H, W]
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']  # 形状：[B, T+1, C, H, W]
        
        # 重塑视觉输入以匹配MVT_TokenFusion_Encoder的期望
        B, T, C, H, W = rgb_static[:, :-1].shape  # 取除了最后一帧外的所有帧
        
        # 创建包含两个相机视图的张量(static和gripper)
        visual_input = torch.zeros((B, 2, T, C, H, W), device=rgb_static.device, dtype=rgb_static.dtype)
        visual_input[:, 0] = rgb_static[:, :-1]  # static相机，除了最后一帧（目标帧）
        visual_input[:, 1] = rgb_gripper[:, :-1]  # gripper相机，除了最后一帧（目标帧）
        
        # 将观察编码为tokens，使用语言条件
        visual_tokens = self.vision_encoder(
            visual_input,  # 形状：[B, 2, T, C, H, W]
            goal["lang"]  # 使用语言嵌入作为条件
        )
        
        # 生成动作序列
        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            visual_tokens,
            latent_goal,
            inference=True,
        )
        
        return act_seq
    
    def step(self, obs, goal):
        """执行一步推理，处理动作分块情况"""
        # 每隔multistep步计算一次序列
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self(obs, goal)
            self.pred_action_seq = pred_action_seq
        
        # 获取当前动作
        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        
        # 维度处理
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        
        # 更新步数计数器
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        
        return current_action
    
    def on_train_start(self) -> None:
        """训练开始时的钩子函数"""
        # 将模型移至当前设备和数据类型
        self.ac_dit_model.to(dtype=self.dtype)
        self.vision_encoder.to(dtype=self.dtype)
        if self.language_encoder is not None:
            self.language_encoder.to(dtype=self.dtype)
    
    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"开始训练周期 {self.current_epoch}")
    
    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional[Any] = None) -> None:
        logger.info(f"完成训练周期 {self.current_epoch}")
    
    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"完成验证周期 {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """仅在rank 0进程记录日志"""
    logger.info(*args, **kwargs) 