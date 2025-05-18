import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
import re

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from AC_DiT.models.multimodal_encoder.vision_encoder_with_tokens import MVT_TokenFusion_Encoder
from AC_DiT.models.multimodal_encoder.text_encoder import TextEncoder
from AC_DiT.models.model.AC_DiT import AC_DiT


class ACDiTRunner(nn.Module):
    """
    AC_DiT Runner类，类似于RDT的Runner，封装了AC_DiT模型、视觉编码器和语言编码器，
    提供用于训练和推理的方法。
    """
    def __init__(
        self,
        *,
        vision_encoder_config: dict,
        language_encoder_config: dict,
        model_config: dict,
        noise_scheduler_config: dict,
        latent_dim: int = 512,
        shared_language_projection: bool = False,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 10,
        sampler_type: str = 'ddim',
        num_sampling_steps: int = 10,
        ckpt_path: Optional[str] = None,
        device: str = 'cuda',
        # use_dropout: bool = True,
        use_l1_loss: bool = False,
    ):
        """
        初始化AC_DiT Runner

        Args:
            vision_encoder_config: 视觉编码器配置
            language_encoder_config: 语言编码器配置
            model_config: AC_DiT模型配置
            noise_scheduler_config: 噪声调度器配置
            latent_dim: 潜在维度
            shared_language_projection: 是否共享语言投影
            action_dim: 动作维度
            obs_horizon: 观察视野
            pred_horizon: 预测视野
            sampler_type: 采样器类型 ('ddim'或'dpm_solver')
            num_sampling_steps: 采样步数
            ckpt_path: 检查点路径
            device: 设备（cuda/cpu）
            # use_dropout: 是否使用dropout
            use_l1_loss: 是否使用L1损失
        """
        super(ACDiTRunner, self).__init__()
        
        self.device = torch.device(device)
        # self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_l1_loss = use_l1_loss
        self.shared_language_projection = shared_language_projection
        # 初始化视觉编码器
        self.vision_encoder = self._build_vision_encoder(vision_encoder_config, shared_language_projection, latent_dim)
        
        # 初始化语言编码器
        self.language_encoder = self._build_language_encoder(language_encoder_config, shared_language_projection, latent_dim)
        
        # 初始化AC_DiT模型
        self.ac_dit_model = self._build_ac_dit_model(model_config, action_dim, obs_horizon, pred_horizon, shared_language_projection, latent_dim)
        
        # 采样相关参数
        self.sampler_type = sampler_type # mark
        self.num_sampling_steps = num_sampling_steps # mark

        # # 如果没有提供噪声调度器配置，使用默认配置
        # if noise_scheduler_config is None:
        #     noise_scheduler_config = {
        #         'num_train_timesteps': 1000,
        #         'beta_schedule': 'linear',
        #         'beta_start': 0.0001,
        #         'beta_end': 0.02,
        #         'prediction_type': 'epsilon',
        #         'clip_sample': True,
        #         'num_inference_timesteps': num_sampling_steps
        #     }
        
        # 保存噪声调度器配置
        self.noise_scheduler_config = noise_scheduler_config
        
        # 创建噪声调度器 - 训练用DDPM
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config.get('num_train_timesteps', 1000),
            beta_schedule=noise_scheduler_config.get('beta_schedule', 'linear'),
            beta_start=noise_scheduler_config.get('beta_start', 0.0001),
            beta_end=noise_scheduler_config.get('beta_end', 0.02),
            prediction_type=noise_scheduler_config.get('prediction_type', 'epsilon'),
            clip_sample=noise_scheduler_config.get('clip_sample', True),
        )
        
        # 创建采样噪声调度器 - 根据采样类型选择调度器
        if self.sampler_type == 'ddim':
            self.noise_scheduler_sample = DDIMScheduler(
                num_train_timesteps=noise_scheduler_config.get('num_train_timesteps', 1000),
                beta_schedule=noise_scheduler_config.get('beta_schedule', 'linear'),
                beta_start=noise_scheduler_config.get('beta_start', 0.0001),
                beta_end=noise_scheduler_config.get('beta_end', 0.02),
                prediction_type=noise_scheduler_config.get('prediction_type', 'epsilon'),
                clip_sample=noise_scheduler_config.get('clip_sample', True),
            )
        elif self.sampler_type == 'dpm_solver':
            self.noise_scheduler_sample = DPMSolverMultistepScheduler(
                num_train_timesteps=noise_scheduler_config.get('num_train_timesteps', 1000),
                beta_schedule=noise_scheduler_config.get('beta_schedule', 'linear'),
                beta_start=noise_scheduler_config.get('beta_start', 0.0001),
                beta_end=noise_scheduler_config.get('beta_end', 0.02),
                prediction_type=noise_scheduler_config.get('prediction_type', 'epsilon'),
            )
        else:
            raise NotImplementedError(f"不支持的采样器类型: {self.sampler_type}")
        
        # 保存采样参数
        self.num_train_timesteps = noise_scheduler_config.get('num_train_timesteps', 1000) # mark
        # self.num_inference_timesteps = noise_scheduler_config.get('num_inference_timesteps', num_sampling_steps) # mark
        self.prediction_type = noise_scheduler_config.get('prediction_type', 'epsilon')
        
        # 是否使用dropout
        # self.use_dropout = use_dropout
        
        # 如果提供了检查点路径，加载权重
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location=self.device)['state_dict'])
        
        # 将模型移至指定设备
        self.to(self.device)
        
        # 打印模型参数数量
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"总参数数量: {total_params:,}")

        # vision_encoder_params = sum(p.numel() for p in self.vision_encoder.parameters())
        # ac_dit_model_params = sum(p.numel() for p in self.ac_dit_model.parameters())
        # trainable_params = vision_encoder_params + ac_dit_model_params
        # print(f"可训练参数数量: {trainable_params:,}")
    
    def _build_vision_encoder(self, config: dict, shared_language_projection: bool, latent_dim: int) -> MVT_TokenFusion_Encoder:
        """
        构建视觉编码器
        
        Args:
            config: 视觉编码器配置
            
        Returns:
            vision_encoder: 视觉编码器实例
        """
        return MVT_TokenFusion_Encoder(
            input_size=config.get('input_size'), # 224
            patch_size=config.get('patch_size'), # 16
            in_channels=config.get('in_channels'), # 3
            hidden_size=latent_dim, # 512
            depth=config.get('depth'), # 10
            num_heads=config.get('num_heads'), # 8
            mlp_ratio=config.get('mlp_ratio'), # 4.0
            num_frames=config.get('num_frames', self.obs_horizon), # 2
            num_cameras=config.get('num_cameras'),  # 静态相机和抓取器相机
            language_dim=config.get('language_dim'), # 512
            attn_drop=config.get('attn_drop', 0.1),
            proj_drop=config.get('proj_drop', 0.1),
            qk_norm=config.get('qk_norm', False),
            mlp_drop=config.get('mlp_drop', 0.05),
            use_token_fusion=config.get('use_token_fusion', False),
            fusion_type=config.get('fusion_type', 'cross_fusion'),
            cross_attn_drop=config.get('cross_attn_drop', 0.1),
            cross_proj_drop=config.get('cross_proj_drop', 0.1),
            cross_qk_norm=config.get('cross_qk_norm', False),
            use_independent_patch_embed=config.get('use_independent_patch_embed', False),
            shared_language_projection=shared_language_projection,
            
        )
    
    def _build_language_encoder(self, config: dict, shared_language_projection: bool, latent_dim: int) -> TextEncoder:
        """
        构建语言编码器
        
        Args:
            config: 语言编码器配置
            
        Returns:
            language_encoder: 语言编码器实例
        """
        return TextEncoder(
            clip_path=config.get('clip_path', 'openai/clip-vit-base-patch32'),
            device=self.device,
            shared_language_projection=shared_language_projection,
            hidden_dim=latent_dim,
            dropout=config.get('dropout', 0.1),
        )
    
    def _build_ac_dit_model(self, config: dict, action_dim: int, obs_horizon: int, 
                            pred_horizon: int, shared_language_projection: bool, latent_dim: int) -> AC_DiT:
        """
        构建AC_DiT模型
        
        Args:
            config: AC_DiT模型配置
            action_dim: 动作维度
            # hidden_dim: 隐藏维度
            obs_horizon: 观察视野
            pred_horizon: 预测视野
            
        Returns:
            ac_dit_model: AC_DiT模型实例
        """
        return AC_DiT(
            action_dim=action_dim,
            hidden_dim=latent_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_heads=config.get('num_heads'),
            mlp_ratio=config.get('mlp_ratio'),
            num_layers=config.get('num_layers'),
            attn_drop=config.get('attn_drop', 0.1),
            proj_drop=config.get('proj_drop', 0.1),
            qk_norm=config.get('qk_norm', False),
            cross_attn_drop=config.get('cross_attn_drop', 0.3),
            cross_proj_drop=config.get('cross_proj_drop', 0.1),
            cross_qk_norm=config.get('cross_qk_norm', True),
            mlp_drop=config.get('mlp_drop', 0.05),
            shared_language_projection=shared_language_projection,
            mlp_embedder=config.get('mlp_embedder', False),
            linear_output=config.get('linear_output', False),
        )
    
    def encode_inputs(self, batch: Dict[str, any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码输入数据
        
        Args:
            batch: 批次数据，包含以下键：
                'input_actions': 形状 [batch_size, obs_horizon, action_dim]
                'rgb_obs': 包含'rgb_static'和'rgb_gripper'的字典，形状 [batch_size, obs_horizon, 3, H, W]
                'target_actions': 形状 [batch_size, pred_horizon, action_dim]
                'lang_text': 语言标注字符串列表
                'idx': 批次中样本索引
                
        Returns:
            视觉tokens和语言目标嵌入
        """
        # 设置为训练模式
        self.vision_encoder.train()

        # 提取视觉观察
        rgb_static = batch["rgb_obs"]['rgb_static']  # 形状：[batch_size, obs_horizon, 3, H, W]
        rgb_gripper = batch["rgb_obs"]['rgb_gripper']  # 形状：[batch_size, obs_horizon, 3, H, W]
        
        # 确保数据中包含语言标注
        if "lang_text" not in batch:
            raise ValueError("每个样本必须包含语言标注！")
            
        # 处理语言目标
        batch_texts = batch["lang_text"]

        latent_goal = self.language_encoder.encode(batch_texts).to(self.device).to(rgb_static.dtype)
        
        # # 对批次中的每个文本进行编码
        # latent_goals = []
        # for text in batch_texts:
        #     text_embed = self.language_encoder.encode(text)
        #     latent_goals.append(text_embed)
        
        # # 拼接语言嵌入
        # latent_goal = torch.cat(latent_goals, dim=0)
        
        # # 调整维度以匹配期望的输出
        # latent_goal = latent_goal.to(self.device).to(rgb_static.dtype)
        
        # 重塑视觉输入以匹配MVT_TokenFusion_Encoder的期望
        # 期望形状：[B, N_c, T, C, H, W]，其中N_c=2（static和gripper），T=obs_horizon
        batch_size, obs_horizon, C, H, W = rgb_static.shape
        
        # 创建包含两个相机视图的张量(static和gripper)
        visual_input = torch.zeros((batch_size, 2, obs_horizon, C, H, W), 
                                  device=rgb_static.device, dtype=rgb_static.dtype)
        visual_input[:, 0] = rgb_static  # static相机
        visual_input[:, 1] = rgb_gripper  # gripper相机
        
        # 编码视觉观察，使用语言条件
        visual_tokens = self.vision_encoder(
            visual_input,  # 形状：[B, 2, T, C, H, W]
            latent_goal    # 语言嵌入作为条件
        )
        
        return visual_tokens, latent_goal
    
    def diffusion_loss(self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor,
                     input_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """
        使用DDPM计算扩散损失
        
        Args:
            perceptual_emb: 感知嵌入，形状 [batch_size, embed_dim]
            latent_goal: 潜在目标，形状 [batch_size, embed_dim]
            input_actions: 输入动作，形状 [batch_size, obs_horizon, action_dim]
            target_actions: 目标动作，形状 [batch_size, pred_horizon, action_dim]
            
        Returns:
            loss: 损失值
        """
        # 将模型设为训练模式
        self.ac_dit_model.train()
        
        batch_size = target_actions.shape[0]
        device = target_actions.device
        
        # 生成噪声
        noise = torch.randn_like(target_actions)
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 向干净动作添加噪声（正向扩散过程）
        noised_actions = self.noise_scheduler.add_noise(
            target_actions, noise, timesteps
        )
        
        # 使用模型预测噪声或动作
        pred = self.ac_dit_model(
            x=noised_actions,            # 噪声动作
            x_cond=input_actions,        # 条件动作
            y=latent_goal,               # 语言目标
            t=timesteps.float(),         # 扩散时间步
            g=perceptual_emb             # 视觉表征
        )
        
        # 根据预测类型计算目标
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'sample':
            target = target_actions
        else:
            raise ValueError(f"不支持的预测类型: {self.prediction_type}")
        
        # 计算损失
        if self.prediction_type == 'sample' and self.use_l1_loss:
            loss = F.l1_loss(pred, target)
        else:
            loss = F.mse_loss(pred, target)
        
        return loss
    
    def predict_actions(self, rgb_observations: Dict[str, torch.Tensor], 
                        language_goal: Union[str, torch.Tensor, List[str]],
                        past_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测动作序列
        
        Args:
            rgb_observations: RGB观察，包含rgb_static和rgb_gripper
                rgb_static: 形状 [batch_size, obs_horizon, 3, H, W] 或 [obs_horizon, 3, H, W]
                rgb_gripper: 形状 [batch_size, obs_horizon, 3, H, W] 或 [obs_horizon, 3, H, W]
            language_goal: 语言目标，可以是文本、文本列表或嵌入
            past_actions: 过去的动作，形状 [batch_size, obs_horizon, action_dim] 或 [obs_horizon, action_dim]，可选
            
        Returns:
            pred_actions: 预测的动作序列，形状 [batch_size, pred_horizon, action_dim] 或 [pred_horizon, action_dim]
        """
        # 设置为评估模式
        self.eval()
        # self.vision_encoder.eval()
        # self.ac_dit_model.eval()

        if isinstance(self.language_encoder, nn.Module):
            self.language_encoder.eval()
        
        # 提取RGB观察
        rgb_static = rgb_observations['rgb_static']
        rgb_gripper = rgb_observations['rgb_gripper']
        
        # 检查输入维度，确定是否有batch维度
        static_shape = rgb_static.shape
        # gripper_shape = rgb_gripper.shape
        
        # 检查维度数量
        has_batch_dim = len(static_shape) == 5  # [B, T, C, H, W]
        # single_sample = not has_batch_dim or static_shape[0] == 1
        
        # 添加batch维度（如果需要）
        if not has_batch_dim:
            # 将 [T, C, H, W] 转换为 [1, T, C, H, W]
            rgb_static = rgb_static.unsqueeze(0)
            rgb_gripper = rgb_gripper.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = static_shape[0]
        
        # 处理语言目标
        if isinstance(language_goal, (str, list)):
            # 字符串或字符串列表，直接用编码器处理
            latent_goal = self.language_encoder.encode(language_goal).to(self.device)
            
            # 检查输出批次大小与RGB批次大小是否匹配
            if latent_goal.shape[0] != batch_size:
                raise ValueError(f"文本嵌入批次大小({latent_goal.shape[0]})与RGB批次大小({batch_size})不匹配")
                # if latent_goal.shape[0] == 1 and batch_size > 1:
                #     # 单个文本嵌入扩展到多个样本
                #     latent_goal = latent_goal.expand(batch_size, -1)
                # elif batch_size == 1 and latent_goal.shape[0] > 1:
                #     # 如果是单个RGB样本但多个文本嵌入，只保留第一个RGB
                #     rgb_static = rgb_static.expand(latent_goal.shape[0], -1, -1, -1, -1)
                #     rgb_gripper = rgb_gripper.expand(latent_goal.shape[0], -1, -1, -1, -1) 
                #     batch_size = latent_goal.shape[0]
        else:
            # 已编码的嵌入
            if language_goal.shape[-1] == 512:
                print(f"将使用已编码的嵌入: {language_goal.shape}")
                latent_goal = language_goal.to(self.device)
            else:
                raise ValueError(f"文本嵌入维度不正确: {language_goal.shape}, 期望嵌入维度: 512")
            
            # 调整嵌入维度以匹配批次
            if len(latent_goal.shape) == 1:  # [embed_dim]
                latent_goal = latent_goal.unsqueeze(0)  # [1, embed_dim]
                
            if latent_goal.shape[0] != batch_size:
                raise ValueError(f"嵌入批次大小({latent_goal.shape[0]})与RGB批次大小({batch_size})不匹配")
                # if latent_goal.shape[0] == 1 and batch_size > 1:
                #     # 扩展单个嵌入到多个样本
                #     latent_goal = latent_goal.repeat(batch_size, 1)
                # elif batch_size == 1 and latent_goal.shape[0] > 1:
                #     # 扩展RGB观察以匹配多个嵌入
                #     rgb_static = rgb_static.expand(latent_goal.shape[0], -1, -1, -1, -1)
                #     rgb_gripper = rgb_gripper.expand(latent_goal.shape[0], -1, -1, -1, -1)
                #     batch_size = latent_goal.shape[0]
                
                    
        
        # 确保latent_goal是二维的 [batch_size, embed_dim]，且类型一致
        latent_goal = latent_goal.to(rgb_static.dtype)
        
        # 处理过去的动作
        if past_actions is None:
            # 如果没有提供过去动作，使用零向量
            input_actions = torch.zeros((batch_size, self.obs_horizon, self.action_dim), 
                                      device=self.device)
        else:
            # 检查并调整过去动作的维度
            if len(past_actions.shape) == 2:  # [T, action_dim]
                past_actions = past_actions.unsqueeze(0)  # 添加batch维度
            
            if past_actions.shape[0] != batch_size:
                raise ValueError(f"动作批次大小({past_actions.shape[0]})与RGB批次大小({batch_size})不匹配")
                # if past_actions.shape[0] == 1 and batch_size > 1:
                #     # 扩展单个动作序列到多个样本
                #     input_actions = past_actions.expand(batch_size, -1, -1)
                # elif batch_size == 1 and past_actions.shape[0] > 1:
                #     # 调整RGB观察以匹配动作批次
                #     rgb_static = rgb_static.expand(past_actions.shape[0], -1, -1, -1, -1)
                #     rgb_gripper = rgb_gripper.expand(past_actions.shape[0], -1, -1, -1, -1)
                #     batch_size = past_actions.shape[0]
                #     input_actions = past_actions
            else:
                input_actions = past_actions
            
        # 确保动作的数据类型和设备正确
        input_actions = input_actions.to(self.device)
        
        # 创建视觉输入 [batch_size, 2, obs_horizon, C, H, W]
        # 获取通道数、高度和宽度
        _, obs_horizon, C, H, W = rgb_static.shape
        
        visual_input = torch.zeros((batch_size, 2, obs_horizon, C, H, W), 
                                   device=self.device, dtype=torch.float32)
        visual_input[:, 0] = rgb_static  # static相机
        visual_input[:, 1] = rgb_gripper  # gripper相机
        
        # 编码视觉观察
        with torch.no_grad():
            visual_tokens = self.vision_encoder(
                visual_input,
                latent_goal
            )
        
        # 初始化噪声动作
        rand_device = self.device
        if 'mps' in str(self.device):
            # Apple Silicon可能需要在CPU上生成随机数
            rand_device = 'cpu'
        
        noisy_action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim), 
            device=rand_device, 
            dtype=latent_goal.dtype
        ).to(self.device)
        
        # 采样预测动作
        with torch.no_grad():
            # 设置采样时间步
            self.noise_scheduler_sample.set_timesteps(self.num_sampling_steps)
            
            # 逐步去噪
            for t in self.noise_scheduler_sample.timesteps:
                # 模型预测
                model_output = self.ac_dit_model(
                    x=noisy_action,
                    x_cond=input_actions,
                    y=latent_goal,
                    t=t.expand(batch_size).to(self.device),
                    g=visual_tokens
                )
                
                # 更新预测动作
                noisy_action = self.noise_scheduler_sample.step(
                    model_output, t, noisy_action
                ).prev_sample
        
        # 如果输入没有batch维度，则去除输出的batch维度
        if not has_batch_dim:
            noisy_action = noisy_action.squeeze(0)  # [pred_horizon, action_dim]
        
        # 切换回训练模式
        self.train()

        return noisy_action
    
    def compute_loss(self, batch: Dict[str, any]) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            batch: 批次数据
                'input_actions': 形状 [batch_size, obs_horizon, action_dim]
                'rgb_obs': 包含'rgb_static'和'rgb_gripper'的字典，形状 [batch_size, obs_horizon, 3, H, W]
                'target_actions': 形状 [batch_size, pred_horizon, action_dim]
                'lang_text': 语言标注字符串列表
                'idx': 批次索引
            
        Returns:
            loss: 训练损失
        """
        # 编码输入
        perceptual_emb, latent_goal = self.encode_inputs(batch)
        
        # 计算扩散损失
        loss = self.diffusion_loss(
            perceptual_emb,
            latent_goal,
            batch["input_actions"],
            batch["target_actions"]
        )
        
        return loss
    
    def forward(self, batch: Dict[str, any]) -> torch.Tensor:
        """
        前向传播，计算损失
        
        Args:
            batch: 批次数据
            
        Returns:
            loss: 训练损失
        """
        return self.compute_loss(batch)
    
    def configure_optimizers(self, learning_rate=1e-4, weight_decay=0.01, 
                            betas=(0.9, 0.95)):
        """
        配置优化器
        
        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减
            betas: Adam优化器的beta参数
            
        Returns:
            optimizer: 优化器
        """
        # 模型参数组
        param_groups = [
            {'params': self.ac_dit_model.parameters(), 'weight_decay': weight_decay},
            {'params': self.vision_encoder.parameters(), 'weight_decay': weight_decay * 0.5},
        ]

        if self.shared_language_projection and self.language_encoder.mlp is not None:
            param_groups.append({
                'params': self.language_encoder.mlp.parameters(),
                'weight_decay': weight_decay * 0.5,
            })
        
        # # 如果语言编码器是可训练的，添加其参数
        # if hasattr(self.language_encoder, 'parameters') and isinstance(self.language_encoder, nn.Module):
        #     param_groups.append({
        #         'params': self.language_encoder.parameters(), 
        #         'weight_decay': weight_decay * 0.5,
        #         'lr': learning_rate * 0.1  # 语言编码器使用较小的学习率
        #     })
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas
        )
        
        return optimizer
    
    def save_checkpoint(self, path):
        """
        保存检查点
        
        Args:
            path: 检查点路径
        """
        # 创建目录（如果不存在）
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'state_dict': self.state_dict(),
        }, path)
        
        print(f"模型已保存到: {path}")
    
    def load_checkpoint(self, path):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        # 加载检查点
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        
        print(f"模型已从 {path} 加载")
        
        
def create_acdit_runner_from_config(config, device='cuda'):
    """
    从配置创建ACDiTRunner实例
    
    Args:
        config: 配置
        device: 设备
        
    Returns:
        runner: ACDiTRunner实例
    """
    # 提取相关配置
    vision_encoder_config = config.model.vision_encoder
    language_encoder_config = config.model.language_encoder
    model_config = config.model.model
    
    # 提取噪声调度器配置（如果存在）
    noise_scheduler_config = None
    if hasattr(config.model, 'noise_scheduler'):
        noise_scheduler_config = config.model.noise_scheduler
    else:
        raise ValueError("没有在config.model下找到噪声调度器配置")
    
    # 创建ACDiTRunner实例
    runner = ACDiTRunner(
        vision_encoder_config=vision_encoder_config,
        language_encoder_config=language_encoder_config,
        model_config=model_config,
        noise_scheduler_config=noise_scheduler_config,
        latent_dim=config.model.get('latent_dim', 512),
        shared_language_projection=config.model.get('shared_language_projection', False),
        action_dim=config.model.get('action_dim', 7), # 7
        obs_horizon=config.model.get('obs_horizon', 2), # 2
        pred_horizon=config.model.get('pred_horizon', 10), # 10
        sampler_type=config.model.get('sampler_type', 'ddim'), # ddim
        num_sampling_steps=config.model.get('num_sampling_steps', 15), # 15
        ckpt_path=config.model.get('ckpt_path', None),
        device=device,
        # use_dropout=not config.model.get('eval_mode', False),
        use_l1_loss=config.model.get('use_l1_loss', False)
    )
    
    return runner 