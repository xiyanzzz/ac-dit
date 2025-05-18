import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from AC_DiT.models.acdit_runner import ACDiTRunner, create_acdit_runner_from_config
from AC_DiT.models.multimodal_encoder.vision_encoder_with_tokens import MVT_TokenFusion_Encoder
from AC_DiT.models.multimodal_encoder.text_encoder import TextEncoder
from AC_DiT.models.model.AC_DiT import AC_DiT
from AC_DiT.models.imgs_pred_decoder.vision_decoder_with_masks import VisionDecoderWithMasks


class ACDiTFuturePredRunner(ACDiTRunner):
    """
    带未来图像预测功能的AC_DiT Runner。
    扩展了原始的ACDiTRunner，添加了视觉解码器来预测未来的图像。
    """
    def __init__(
        self,
        *,
        vision_encoder_config: dict,
        language_encoder_config: dict,
        model_config: dict,
        noise_scheduler_config: dict,
        vision_decoder_config: Optional[dict] = None,
        latent_dim: int = 512,
        shared_language_projection: bool = False,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 10,
        sampler_type: str = 'ddim',
        num_sampling_steps: int = 15,
        ckpt_path: Optional[str] = None,
        device: str = 'cuda',
        # use_dropout: bool = True,
        use_l1_loss: bool = False,
        use_future_prediction: bool = True,
        future_pred_weight: float = 0.1,
    ):
        """
        初始化ACDiTFuturePredRunner

        Args:
            vision_encoder_config: 视觉编码器配置
            language_encoder_config: 语言编码器配置
            model_config: AC_DiT模型配置
            noise_scheduler_config: 噪声调度器配置
            vision_decoder_config: 视觉解码器配置，用于未来图像预测
            latent_dim: 潜在维度
            action_dim: 动作维度
            obs_horizon: 观察视野
            pred_horizon: 预测视野
            sampler_type: 采样器类型 ('ddim'或'dpm_solver')
            num_sampling_steps: 采样步数
            ckpt_path: 检查点路径
            device: 设备（cuda/cpu）
            use_dropout: 是否使用dropout
            use_future_prediction: 是否使用未来图像预测
            future_pred_weight: 未来预测损失权重
        """
        # 调用父类初始化方法
        super(ACDiTFuturePredRunner, self).__init__(
            vision_encoder_config=vision_encoder_config,
            language_encoder_config=language_encoder_config,
            model_config=model_config,
            noise_scheduler_config=noise_scheduler_config,
            latent_dim=latent_dim,
            shared_language_projection=shared_language_projection,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            sampler_type=sampler_type,
            num_sampling_steps=num_sampling_steps,
            ckpt_path=None,  # 暂时不加载检查点，后面会手动加载
            device=device,
            # use_dropout=use_dropout,
            use_l1_loss=use_l1_loss,
        )
        
        # 是否使用未来图像预测功能
        self.use_future_prediction = use_future_prediction
        self.future_pred_weight = future_pred_weight
        
        # 如果启用未来图像预测，初始化视觉解码器
        self.vision_decoder = None
        if self.use_future_prediction:
            if vision_decoder_config is None:
                raise ValueError("启用未来图像预测功能，但未提供视觉解码器配置")
            
            # 确保解码器配置包含编码器输出维度
            if 'encoder_hidden_size' not in vision_decoder_config:
                vision_decoder_config['encoder_hidden_size'] = 512 # latent_dim
                
            # 初始化视觉解码器
            self.vision_decoder = self._build_vision_decoder(vision_decoder_config)
            print(f"已启用未来图像预测，预测权重: {self.future_pred_weight}")
        
        # 如果提供了检查点路径，加载权重
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        
        # 将模型移至指定设备
        self.to(self.device)
        
        # 重新计算模型参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数数量: {total_params:,}")
        print('--------------------------------')
        vision_encoder_params = sum(p.numel() for p in self.vision_encoder.parameters())
        print(f"视觉编码器参数数量: {vision_encoder_params:,}")
        
        ac_dit_model_params = sum(p.numel() for p in self.ac_dit_model.parameters())
        print(f"AC_DiT模型参数数量: {ac_dit_model_params:,}")
        
        language_encoder_params = sum(p.numel() for p in self.language_encoder.clip.parameters())
        print(f"语言编码器参数数量: {language_encoder_params:,}")
        
        trainable_params = vision_encoder_params + ac_dit_model_params
        if self.vision_decoder is not None:
            vision_decoder_params = sum(p.numel() for p in self.vision_decoder.parameters())
            trainable_params += vision_decoder_params
            print(f"视觉解码器参数数量: {vision_decoder_params:,}")
        if self.shared_language_projection and self.language_encoder.mlp is not None:
            language_projector_params = sum(p.numel() for p in self.language_encoder.mlp.parameters())
            trainable_params += language_projector_params
            print(f"语言投影器参数数量: {language_projector_params:,}")
        print('--------------------------------')
        print(f"可训练参数数量: {trainable_params:,}")
    
    def _build_vision_decoder(self, config: dict) -> VisionDecoderWithMasks:
        """
        构建视觉解码器
        
        Args:
            config: 视觉解码器配置
            
        Returns:
            vision_decoder: 视觉解码器实例
        """
        return VisionDecoderWithMasks(
            input_size=config.get('input_size', 144),
            patch_size=config.get('patch_size', 16),
            in_channels=config.get('in_channels', 3),
            hidden_size=config.get('hidden_size', 256),
            encoder_hidden_size=config.get('encoder_hidden_size', 512),
            depth=config.get('depth'), # 留作检查
            num_heads=config.get('num_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            num_cameras=config.get('num_cameras', 2),
            mask_ratio=config.get('mask_ratio', 0.75),
            max_future_step=config.get('max_future_step', 10),
            attn_drop=config.get('attn_drop', 0.3),
            proj_drop=config.get('proj_drop', 0.1),
            qk_norm=config.get('qk_norm', False),
            cross_attn_drop=config.get('cross_attn_drop', 0.),
            cross_proj_drop=config.get('cross_proj_drop', 0.),
            cross_qk_norm=config.get('cross_qk_norm', True),
            mlp_drop=config.get('mlp_drop', 0.),
        )

    def prediction_loss(self, vision_tokens: torch.Tensor, future_rgb_static: torch.Tensor, 
                        future_rgb_gripper: torch.Tensor, future_step: torch.Tensor) -> torch.Tensor:
        """
        计算未来图像预测损失
        """
        # 设置为训练模式
        self.vision_decoder.train()
        # 堆叠多视角图像
        future_x = torch.stack([future_rgb_static, future_rgb_gripper], dim=1)  # [B, 2, C, H, W]
        
        # 调用前向计算预测损失
        img_pred_loss, _, _ = self.vision_decoder(x=future_x, 
                                                  vision_tokens=vision_tokens, 
                                                  future_step=future_step)
        
        return img_pred_loss

    def compute_loss(self, batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        """
        计算训练损失，包括动作预测损失和图像预测损失
        
        Args:
            batch: 批次数据
                'input_actions': 形状 [batch_size, obs_horizon, action_dim]
                'rgb_obs': 包含'rgb_static'和'rgb_gripper'的字典，形状 [batch_size, obs_horizon, 3, H, W]
                'target_actions': 形状 [batch_size, pred_horizon, action_dim]
                'lang_text': 语言标注字符串列表
                'idx': 批次索引
                'future_rgb_static': 未来静态相机图像 [batch_size, 3, H, W]
                'future_rgb_gripper': 未来抓取器相机图像 [batch_size, 3, H, W]
                'future_step': 未来时间步 [batch_size]
            
        Returns:
            losses: 包含各种损失的字典
        """
        # 编码输入
        vision_tokens, latent_goal = self.encode_inputs(batch)
        
        # 计算扩散损失
        action_loss = self.diffusion_loss(
            vision_tokens,
            latent_goal,
            batch["input_actions"],
            batch["target_actions"]
        )
        
        # 初始化损失字典
        losses = {
            'action_loss': action_loss,
            'total_loss': action_loss
        }
        
        # 如果启用未来图像预测，计算图像预测损失
        if self.use_future_prediction and self.vision_decoder is not None and 'future_rgb_static' in batch:

            img_pred_loss = self.prediction_loss(
                vision_tokens, 
                future_rgb_static = batch['future_rgb_static'], 
                future_rgb_gripper = batch['future_rgb_gripper'], 
                future_step = batch['future_step']
            )
            
            # 更新损失
            losses['img_pred_loss'] = img_pred_loss
            losses['total_loss'] = action_loss + self.future_pred_weight * img_pred_loss
        
        return losses
    
    def forward(self, batch: Dict[str, any]) -> torch.Tensor:
        """
        前向传播，计算总损失
        
        Args:
            batch: 批次数据
            
        Returns:
            total_loss: 总训练损失
        """
        losses = self.compute_loss(batch)
        return losses['total_loss']
    
    def predict_future_images(self, rgb_observations: Dict[str, torch.Tensor],
                          language_goal: Union[str, torch.Tensor, List[str]],
                          future_step: torch.Tensor) -> torch.Tensor:
        """
        预测未来图像
        
        Args:
            rgb_observations: RGB观测，包含rgb_static和rgb_gripper
                rgb_static: 形状 [batch_size, obs_horizon, 3, H, W] 或 [obs_horizon, 3, H, W]
                rgb_gripper: 形状 [batch_size, obs_horizon, 3, H, W] 或 [obs_horizon, 3, H, W]
            language_goal: 语言目标，可以是文本、文本列表或嵌入
            future_step: 未来时间步，形状 [batch_size] 或整数
            
        Returns:
            pred_imgs: 预测的未来图像，形状 [batch_size, num_cameras, 3, H, W]
        """
        if not self.use_future_prediction or self.vision_decoder is None:
            raise ValueError("未启用未来图像预测功能")
        
        # 切换到评估模式
        self.eval()
        
        # 确保RGB观测在正确的设备上
        device = self.device
        rgb_static = rgb_observations['rgb_static'].to(device)
        rgb_gripper = rgb_observations['rgb_gripper'].to(device)
        
        # 处理输入维度
        if len(rgb_static.shape) == 4:  # [B, C, H, W]
            # 添加时间维度
            rgb_static = rgb_static.unsqueeze(1)  # [B, 1, C, H, W]
            rgb_gripper = rgb_gripper.unsqueeze(1)  # [B, 1, C, H, W]
        
        # 创建批次数据
        batch = {
            'rgb_obs': {
                'rgb_static': rgb_static,
                'rgb_gripper': rgb_gripper
            }
        }
        
        # 处理语言目标
        if isinstance(language_goal, str):
            batch['lang_text'] = [language_goal]
        elif isinstance(language_goal, list):
            batch['lang_text'] = language_goal
        else:
            raise ValueError("语言目标必须是字符串或列表,暂不支持预嵌入")
            # # 假设是编码好的语言嵌入
            # latent_goal = language_goal.to(device)
            # # 创建一个假的批次数据，仅用于获取vision_tokens
            # temp_batch = batch.copy()
            # temp_batch['lang_text'] = ['dummy' for _ in range(rgb_static.shape[0])]
            
        # 处理future_step
        if isinstance(future_step, int):
            future_step = torch.tensor([future_step], device=device).expand(rgb_static.shape[0])
        else:
            future_step = future_step.to(device)
        
        # 获取视觉编码
        with torch.no_grad():
            if 'lang_text' in batch:
                vision_tokens, _ = self.encode_inputs(batch)
            else:
                raise ValueError("语言目标必须是字符串或列表,暂不支持预嵌入")
                # # 使用提供的语言嵌入
                # # 为视觉编码器准备多视角输入
                # vision_input = torch.stack([rgb_static, rgb_gripper], dim=1)  # [B, 2, T, C, H, W]
                # vision_tokens = self.vision_encoder(vision_input, latent_goal)

            
            # 生成未来图像
            future_images = self.vision_decoder.generate_images(
                vision_tokens=vision_tokens,
                future_step=future_step,
                device=device
            ) # [B, N_c, 3, H, W]
        
        # 切换回训练模式
        self.train()
        
        return future_images
    
    def configure_optimizers(self, learning_rate=1e-4, weight_decay=0.01, 
                          betas=(0.9, 0.95)):
        """
        配置优化器，包括视觉解码器的参数
        
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

        # 添加视觉解码器参数
        if self.use_future_prediction and self.vision_decoder is not None:
            param_groups.append({
                'params': self.vision_decoder.parameters(),
                'weight_decay': weight_decay * 0.5
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


def create_acdit_future_pred_runner_from_config(config, device='cuda'):
    """
    从配置创建ACDiTFuturePredRunner实例
    
    Args:
        config: 配置
        device: 设备
        
    Returns:
        runner: ACDiTFuturePredRunner实例
    """
    # 提取相关配置
    vision_encoder_config = config.model.vision_encoder
    language_encoder_config = config.model.language_encoder
    model_config = config.model.model
    
    # 提取噪声调度器配置
    noise_scheduler_config = None
    if hasattr(config.model, 'noise_scheduler'):
        noise_scheduler_config = config.model.noise_scheduler
    else:
        raise ValueError("没有在config.model下找到噪声调度器配置")
    
    # 提取视觉解码器配置（如果存在）
    vision_decoder_config = None
    if hasattr(config.model, 'vision_decoder'):
        vision_decoder_config = config.model.vision_decoder
    
    # 检查是否启用未来图像预测
    use_future_prediction = config.model.get('use_future_prediction')
    future_pred_weight = config.model.get('future_pred_weight', 0.1)
    
    # 创建ACDiTFuturePredRunner实例
    runner = ACDiTFuturePredRunner(
        vision_encoder_config=vision_encoder_config,
        language_encoder_config=language_encoder_config,
        model_config=model_config,
        noise_scheduler_config=noise_scheduler_config,
        vision_decoder_config=vision_decoder_config,
        latent_dim=config.model.get('latent_dim', 512),
        shared_language_projection=config.model.get('shared_language_projection', False),
        action_dim=config.model.get('action_dim', 7),
        obs_horizon=config.data.get('observation_horizon', 2),
        pred_horizon=config.data.get('prediction_horizon', 10),
        sampler_type=config.model.get('sampler_type', 'ddim'),
        num_sampling_steps=config.model.get('num_sampling_steps', 15),
        ckpt_path=config.model.get('ckpt_path', None),
        device=device,
        #use_dropout=not config.model.get('eval_mode', False),
        use_l1_loss=config.model.get('use_l1_loss', False),
        use_future_prediction=use_future_prediction,
        future_pred_weight=future_pred_weight
    )
    
    return runner
