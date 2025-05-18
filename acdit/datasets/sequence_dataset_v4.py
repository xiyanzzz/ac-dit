import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import threading
import time
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig
import threading
from functools import lru_cache
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class SeparateSequenceDataset(Dataset):
    """
    懒加载序列数据集，预加载动作数据，图像数据在需要时再加载
    优化:
    1. 从元数据获取total_samples
    2. 移除对不存在的episode_indices和frame_indices的引用
    3. 一次性加载两种图像
    """
    def __init__(
        self,
        data_path: str,
        transform=None,
        use_future_image: bool = True,
        future_image_mean: float = 5.0,
        future_image_std: float = 1.7,
        cache_size: int = 1000,  # 图像缓存大小
    ):
        """
        初始化懒加载序列数据集
        
        Args:
            data_path: 分离数据文件所在目录
            transform: 可选的数据变换函数字典
            use_future_image: 是否使用未来图像
            future_image_mean: 未来图像间隔时间步的正态分布均值
            future_image_std: 未来图像间隔时间步的正态分布标准差
            cache_size: 图像缓存大小
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.use_future_image = use_future_image
        self.future_image_mean = future_image_mean
        self.future_image_std = future_image_std
        self.cache_size = cache_size
        
        # 数据文件路径
        self.actions_file = os.path.join(data_path, "actions_data.npz")
        self.metadata_file = os.path.join(data_path, "metadata.npz")
        self.seq_index_file = os.path.join(data_path, "sequence_indices.npz")
        
        # 检查文件是否存在
        for file_path in [self.actions_file, self.metadata_file, self.seq_index_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到必要的数据文件: {file_path}")
        
        print(f"初始化懒加载序列数据集: {self.data_path}")
        print(f"使用未来图像: {use_future_image}, 均值: {future_image_mean}, 标准差: {future_image_std}")
        
        # 加载元数据
        self.metadata = np.load(self.metadata_file, allow_pickle=True)
        
        # 获取基本参数
        self.observation_horizon = int(self.metadata['observation_horizon'])
        self.prediction_horizon = int(self.metadata['prediction_horizon'])
        self.language_annotations = self.metadata['language_annotations']
        self.images_dir = str(self.metadata['images_dir'])
        # 直接在数据目录下查找images文件夹，避免路径拼接问题
        images_dir_candidate = os.path.join(self.data_path, "images")
        if os.path.exists(images_dir_candidate):
            self.images_dir = images_dir_candidate
        else:
            # 备用方案：使用提供的路径
            if not os.path.isabs(self.images_dir):
                self.images_dir = os.path.abspath(self.images_dir)
        self.total_samples = int(self.metadata['total_samples'])  # 从元数据获取总样本数
        
        # 确保图像目录存在
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"找不到图像目录: {self.images_dir}")
        
        print(f"观测窗口: {self.observation_horizon}, 预测窗口: {self.prediction_horizon}")
        print(f"语言标注数量: {len(self.language_annotations)}")
        print(f"图像目录: {self.images_dir}")
        
        # 加载序列索引
        seq_indices_data = np.load(self.seq_index_file)
        self.sequence_indices = seq_indices_data['sequence_indices']
        
        print(f"序列样本总数: {self.total_samples}")
        
        # 预加载动作数据
        print("预加载动作数据到内存中...")
        try:
            start_time = time.time()
            # 加载动作数据
            actions_data = np.load(self.actions_file)
            
            # 创建动作数据字典
            self.actions_data = {
                'abs_actions': actions_data['abs_actions'],
                'rel_actions': actions_data['rel_actions']
            }
            
            end_time = time.time()
            print(f"动作数据预加载完成，耗时: {end_time - start_time:.2f}秒")
            print(f"动作数据形状: abs_actions={self.actions_data['abs_actions'].shape}, rel_actions={self.actions_data['rel_actions'].shape}")
            
            # 关闭原始数据文件
            actions_data.close()
        except Exception as e:
            raise RuntimeError(f"加载动作数据文件失败: {e}")
        
        # 创建图像加载锁，避免多线程访问冲突
        self.lock = threading.Lock()
        
        # 定义LRU缓存装饰器，用于缓存图像数据
        self._get_images_cached = lru_cache(maxsize=cache_size)(self._get_images)
    
    def __len__(self) -> int:
        """数据集大小"""
        return self.total_samples
    
    def _sample_future_step(self):
        """生成以future_image_mean为中心的正态分布随机时间步，范围限制在[1, prediction_horizon]"""
        step = int(np.round(np.random.normal(self.future_image_mean, self.future_image_std)))
        # 确保在有效范围内
        return max(1, min(step, self.prediction_horizon))
    
    def _get_images(self, frame_idx):
        """
        加载指定帧的两种图像
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            rgb_static, rgb_gripper: 两种numpy数组格式的图像
        """
        with self.lock:  # 使用锁避免多线程冲突
            image_path = os.path.join(self.images_dir, f"frame_{frame_idx:06d}.npz")
            
            try:
                # 一次读取两种图像数据
                image_data = np.load(image_path)
                rgb_static = image_data['rgb_static']
                rgb_gripper = image_data['rgb_gripper']
                return rgb_static, rgb_gripper
            except Exception as e:
                raise FileNotFoundError(f"加载图像失败 [frame={frame_idx}]: {e}")
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个序列样本"""
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.total_samples}")
            
        # 获取序列索引信息
        _, lang_idx, frame_start_idx = self.sequence_indices[idx]
        
        # 获取语言标注
        if lang_idx >= 0 and lang_idx < len(self.language_annotations):
            lang_text = self.language_annotations[lang_idx]
        else:
            raise ValueError(f"样本 {idx} 的语言索引 {lang_idx} 无效")
        
        # 提取输入观测序列的索引
        frame_indices = []
        for i in range(self.observation_horizon + self.prediction_horizon):
            frame_indices.append(frame_start_idx + i)
        
        # 提取输入观测序列
        in_actions = []
        in_rgb_static = []
        in_rgb_gripper = []
        
        try:
            # 加载观测帧的动作和图像
            for i in range(self.observation_horizon):
                frame_idx = frame_indices[i]
                
                # 添加绝对动作 - 直接从预加载数据中获取
                in_actions.append(self.actions_data['abs_actions'][frame_idx])
                
                # 一次加载两种图像
                rgb_static, rgb_gripper = self._get_images_cached(int(frame_idx))
                in_rgb_static.append(rgb_static)
                in_rgb_gripper.append(rgb_gripper)
            
            # 提取目标动作序列
            target_actions = []
            for i in range(self.prediction_horizon):
                frame_idx = frame_indices[self.observation_horizon + i]
                target_actions.append(self.actions_data['rel_actions'][frame_idx])
        except Exception as e:
            print(f"访问数据时出错 [idx={idx}, frame_start={frame_start_idx}]: {e}")
            raise
        
        # 转换为numpy数组
        in_actions = np.stack(in_actions)
        in_rgb_static = np.stack(in_rgb_static)
        in_rgb_gripper = np.stack(in_rgb_gripper)
        target_actions = np.stack(target_actions)
        
        # 额外添加未来RGB观测
        future_rgb_static = None
        future_rgb_gripper = None
        future_step = 0
        
        if self.use_future_image:
            future_step = self._sample_future_step()
            future_frame_idx = frame_indices[self.observation_horizon + future_step - 1]  # 减1是因为当前时刻是从观测帧的最后一帧开始
            
            try:
                # 一次加载两种未来图像
                future_rgb_static, future_rgb_gripper = self._get_images_cached(int(future_frame_idx))
            except Exception as e:
                print(f"获取未来图像时出错 [idx={idx}, future_frame={future_frame_idx}]: {e}")
                # 出错时使用最后一个观测帧的图像
                last_frame_idx = frame_indices[self.observation_horizon - 1]
                future_rgb_static, future_rgb_gripper = self._get_images_cached(int(last_frame_idx))
        
        # 转换为PyTorch张量
        in_actions = torch.from_numpy(in_actions).float()
        in_rgb_static = torch.from_numpy(in_rgb_static).float() / 255.0
        in_rgb_gripper = torch.from_numpy(in_rgb_gripper).float() / 255.0
        target_actions = torch.from_numpy(target_actions).float()
        
        # 转换图像格式：从 [T, H, W, C] 到 [T, C, H, W]
        in_rgb_static = in_rgb_static.permute(0, 3, 1, 2)
        in_rgb_gripper = in_rgb_gripper.permute(0, 3, 1, 2)
        
        # 构建返回字典
        result = {
            'input_actions': in_actions,
            'rgb_obs': {
                'rgb_static': in_rgb_static,
                'rgb_gripper': in_rgb_gripper,
            },
            'target_actions': target_actions,
            'idx': idx,
            'lang_text': lang_text,
        }
        
        # 添加未来图像数据（如果启用）
        if self.use_future_image and future_rgb_static is not None:
            future_rgb_static = torch.from_numpy(future_rgb_static).float() / 255.0
            future_rgb_gripper = torch.from_numpy(future_rgb_gripper).float() / 255.0
            
            # 转换图像格式：从 [H, W, C] 到 [C, H, W]
            future_rgb_static = future_rgb_static.permute(2, 0, 1)
            future_rgb_gripper = future_rgb_gripper.permute(2, 0, 1)
            
            result['future_rgb_static'] = future_rgb_static
            result['future_rgb_gripper'] = future_rgb_gripper
            result['future_step'] = future_step
        
        # 应用数据变换（如果提供）
        if self.transform is not None:
            # 对各个组件分别应用对应的变换
            if 'rgb_static' in self.transform and 'rgb_obs' in result and 'rgb_static' in result['rgb_obs']:
                frames = []
                for t in range(result['rgb_obs']['rgb_static'].size(0)):
                    frame = result['rgb_obs']['rgb_static'][t]
                    frame = self.transform['rgb_static'](frame)
                    frames.append(frame)
                result['rgb_obs']['rgb_static'] = torch.stack(frames)
                
            if 'rgb_gripper' in self.transform and 'rgb_obs' in result and 'rgb_gripper' in result['rgb_obs']:
                frames = []
                for t in range(result['rgb_obs']['rgb_gripper'].size(0)):
                    frame = result['rgb_obs']['rgb_gripper'][t]
                    frame = self.transform['rgb_gripper'](frame)
                    frames.append(frame)
                result['rgb_obs']['rgb_gripper'] = torch.stack(frames)
                
            if 'actions' in self.transform and 'input_actions' in result:
                result['input_actions'] = self.transform['actions'](result['input_actions'])
                
            if 'actions' in self.transform and 'target_actions' in result:
                result['target_actions'] = self.transform['actions'](result['target_actions'])
                
            # 应用未来图像的变换
            if self.use_future_image and 'future_rgb_static' in self.transform and 'future_rgb_static' in result:
                result['future_rgb_static'] = self.transform['future_rgb_static'](result['future_rgb_static'])
                
            if self.use_future_image and 'future_rgb_gripper' in self.transform and 'future_rgb_gripper' in result:
                result['future_rgb_gripper'] = self.transform['future_rgb_gripper'](result['future_rgb_gripper'])
            
        return result
    
    def clear_cache(self):
        """清除图像缓存"""
        self._get_images_cached.cache_clear()


class SeparateSequenceDataModule(pl.LightningDataModule):
    """
    懒加载序列数据模块，用于管理训练和验证数据集
    """
    def __init__(
        self,
        dataset_path: str,
        train_folder: str = "training/separate_data",
        val_folder: str = "validation/separate_data",
        batch_size: int = 64,
        num_workers: int = 8,
        transforms: Optional[Dict[str, Any]] = None,
        use_future_image: bool = True,
        future_image_mean: float = 5.0,
        future_image_std: float = 1.7,
        cache_size: int = 1000,  # 图像缓存大小
    ):
        """
        初始化懒加载序列数据模块
        
        Args:
            dataset_path: 数据集根路径
            train_folder: 训练数据文件夹
            val_folder: 验证数据文件夹
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            transforms: 数据变换配置
            use_future_image: 是否使用未来图像
            future_image_mean: 未来图像间隔时间步的正态分布均值
            future_image_std: 未来图像间隔时间步的正态分布标准差
            cache_size: 图像缓存大小, 单位: 缓存项数量 (cache entries), 而非字节 约138MB/1000 总共86w
        """            
        super().__init__()
        self.dataset_path = dataset_path
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.use_future_image = use_future_image
        self.future_image_mean = future_image_mean
        self.future_image_std = future_image_std
        self.cache_size = cache_size
        
        self.train_dataset = None
        self.val_dataset = None
        
        # 构建数据变换
        self.train_transforms = self._build_transforms("train") if transforms else None
        self.val_transforms = self._build_transforms("val") if transforms else None
        
        print(f"初始化懒加载序列数据模块: {self.dataset_path}")
        print(f"批次大小: {self.batch_size}, 工作进程数: {self.num_workers}")
        print(f"使用未来图像: {use_future_image}, 均值: {future_image_mean}, 标准差: {future_image_std}")
        print(f"图像缓存大小: {cache_size}")
    
    def _build_transforms(self, split: str) -> Dict[str, Callable]:
        """构建数据变换函数"""
        if not self.transforms or split not in self.transforms:
            return None
            
        transform_dict = {}
        for key, transforms_list in self.transforms[split].items():
            transform_sequence = []
            for transform_cfg in transforms_list:
                # 检查是否为已实例化的对象
                if (isinstance(transform_cfg, dict) or isinstance(transform_cfg, DictConfig)) and "_target_" in transform_cfg:
                    # 字典格式的配置，需要实例化
                    transform_cfg_copy = dict(transform_cfg)
                    target = transform_cfg_copy.pop("_target_", None)
                    if target:
                        # 处理自定义变换
                        if "RandomShiftsAug" in target:
                            transform = RandomShiftsAug(**transform_cfg_copy)
                        elif "AddGaussianNoise" in target:
                            transform = AddGaussianNoise(**transform_cfg_copy)
                        else:
                            # 处理torchvision变换
                            transform_class = target.split(".")[-1]
                            transform = getattr(T, transform_class)(**transform_cfg_copy)
                        transform_sequence.append(transform)
                else:
                    # 已经是实例化的对象，直接使用
                    transform_sequence.append(transform_cfg)
                
            if transform_sequence:
                transform_dict[key] = T.Compose(transform_sequence)

        print("="*100)
        print(f"成功构建{split}的变换字典: {transform_dict}")
        print("="*100)

        return transform_dict


    def prepare_data(self):
        """准备数据，检查文件是否存在"""
        train_data_path = os.path.join(self.dataset_path, self.train_folder)
        val_data_path = os.path.join(self.dataset_path, self.val_folder)
        
        # 检查训练数据
        if not os.path.exists(train_data_path):
            logger.warning(f"找不到训练数据目录: {train_data_path}")
        
        # 检查验证数据
        if not os.path.exists(val_data_path):
            logger.warning(f"找不到验证数据目录: {val_data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集
        
        Args:
            stage: 训练阶段
        """
        # 初始化训练数据集
        if stage == "fit" or stage is None:
            train_data_path = os.path.join(self.dataset_path, self.train_folder)
            try:
                self.train_dataset = SeparateSequenceDataset(
                    data_path=train_data_path,
                    transform=self.train_transforms,
                    use_future_image=self.use_future_image,
                    future_image_mean=self.future_image_mean,
                    future_image_std=self.future_image_std,
                    cache_size=self.cache_size
                )
                print(f"训练数据集大小: {len(self.train_dataset)}")
            except FileNotFoundError as e:
                logger.error(f"找不到训练数据: {e}")
                raise
        
        # 初始化验证数据集
        if stage == "validate" or stage is None:
            val_data_path = os.path.join(self.dataset_path, self.val_folder)
            try:
                self.val_dataset = SeparateSequenceDataset(
                    data_path=val_data_path,
                    transform=self.val_transforms,
                    use_future_image=self.use_future_image,
                    future_image_mean=self.future_image_mean,
                    future_image_std=self.future_image_std,
                    cache_size=self.cache_size
                )
                print(f"验证数据集大小: {len(self.val_dataset)}")
            except FileNotFoundError as e:
                logger.error(f"找不到验证数据: {e}")
                raise
    
    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器"""
        if self.train_dataset is None:
            logger.warning("训练数据集未初始化，自动调用setup('fit')方法进行初始化...")
            self.setup(stage="fit")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True # abcd drop last
        )
    
    def val_dataloader(self) -> DataLoader:
        """返回验证数据加载器"""
        if self.val_dataset is None:
            logger.warning("验证数据集未初始化，自动调用setup('validate')方法进行初始化...")
            self.setup(stage="validate")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=False
        )
    # NotImplementedError: `LightningDataModule.on_save_checkpoint` was deprecated in v1.6 and is no longer supported as of v1.8. Use `state_dict` instead.
    # def on_save_checkpoint(self, checkpoint):
    #     """在保存检查点时清空图像缓存"""
    #     if self.train_dataset:
    #         self.train_dataset.clear_cache()
    #     if self.val_dataset:
    #         self.val_dataset.clear_cache()
    
    # def on_load_checkpoint(self, checkpoint):
    #     """在加载检查点时清空图像缓存"""
    #     if self.train_dataset:
    #         self.train_dataset.clear_cache()
    #     if self.val_dataset:
    #         self.val_dataset.clear_cache()


# 定义RandomShiftsAug类
class RandomShiftsAug(torch.nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad
        
    def forward(self, x):
        # 确保输入是正确的形状 [C, H, W]
        if len(x.shape) == 3:
            c, h, w = x.size()
            # 变换为 [1, C, H, W] 以便处理
            x = x.unsqueeze(0)
            n = 1
        elif len(x.shape) == 4:
            n, c, h, w = x.size()
        else:
            raise ValueError(f"输入张量形状不正确: {x.shape}")
            
        assert h == w, f"期望正方形图像，但获得尺寸为 {h}x{w}"
        
        padding = tuple([self.pad] * 4)
        x = torch.nn.functional.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        result = torch.nn.functional.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        
        # 如果输入是3D，则输出也应该是3D
        if len(x.shape) == 4 and n == 1:
            result = result.squeeze(0)
            
        return result

# 定义AddGaussianNoise类
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=[0.0], std=[0.01]):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        
    def forward(self, x):
        if self.training:
            # 确保mean和std的形状与x兼容
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            
            # 展平mean和std，使其能广播到任意形状的x
            while len(mean.shape) < len(x.shape):
                mean = mean.unsqueeze(0)
            while len(std.shape) < len(x.shape):
                std = std.unsqueeze(0)
                
            noise = torch.randn_like(x) * std + mean
            return x + noise
        return x

class SaltPepperNoiseWithSNR(torch.nn.Module):
    """基于SNR的椒盐噪声变换
    
    Args:
        snr (float): 信噪比，信号功率与噪声功率的比值 (默认: 0.95)
        salt_vs_pepper (float): 盐噪声(白点)相对于椒噪声(黑点)的比例 (默认: 0.5)
    """
    def __init__(self, snr=0.95, salt_vs_pepper=0.5):
        super().__init__()
        self.snr = snr
        self.salt_vs_pepper = salt_vs_pepper
        
    def forward(self, img):
        """
        Args:
            img (Tensor): 形状为 (C, H, W) 的图像张量
        
        Returns:
            Tensor: 添加椒盐噪声后的图像
        """
        if not self.training or random.random() < 0.5:  # 仅在训练时有50%的概率应用噪声
            return img
            
        img_np = img.clone()
        
        # 计算图像功率
        img_power = torch.mean(img_np ** 2)
        
        # 根据SNR计算噪声功率
        noise_power = img_power / self.snr
        
        # 计算像素被噪声影响的概率
        total_pixels = img_np[0].numel()
        affected_ratio = min(1.0, noise_power.item() * 3)  # 限制最大比例
        num_affected_pixels = int(total_pixels * affected_ratio)
        
        # 生成噪声掩码
        flat_indices = torch.randperm(total_pixels, device=img.device)[:num_affected_pixels]
        
        # 按盐椒比例分配黑白噪声点
        salt_ratio = self.salt_vs_pepper
        salt_indices = flat_indices[:int(num_affected_pixels * salt_ratio)]
        pepper_indices = flat_indices[int(num_affected_pixels * salt_ratio):]
        
        # 应用噪声到所有通道
        for c in range(img_np.shape[0]):
            img_np[c].view(-1)[salt_indices] = 1.0     # 盐噪声 (白点)
            img_np[c].view(-1)[pepper_indices] = 0.0   # 椒噪声 (黑点)
            
        return img_np

# 注册自定义变换
CUSTOM_TRANSFORMS = {
    "RandomShiftsAug": RandomShiftsAug,
    "AddGaussianNoise": AddGaussianNoise,
    "SaltPepperNoiseWithSNR": SaltPepperNoiseWithSNR
}



# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试懒加载数据集')
    parser.add_argument('--dataset_path', type=str, default='dataset/task_ABCD_D',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='工作线程数')
    parser.add_argument('--cache_size', type=int, default=1000,
                        help='图像缓存大小')
    
    args = parser.parse_args()

    from omegaconf import OmegaConf
    config = 'AC_DiT/configs/ac_dit_sequence_v4.yaml'
    if os.path.exists(config):
        cfg = OmegaConf.load(config)
        print(f"已加载配置文件: {config}")
    else:
        print(f"找不到配置文件")

    transforms=cfg.data.transforms

    if transforms:
        print(f"变换配置:")
        for key, transform in transforms.items():
            print(f"  - {key}: {transform}")
    else:
        print("未配置数据变换")

    # 创建数据模块
    data_module = SeparateSequenceDataModule(
        dataset_path=args.dataset_path,
        train_folder="training/separate_data",
        val_folder="validation/separate_data",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_future_image=True,
        future_image_mean=5.0,
        future_image_std=1.7,
        cache_size=args.cache_size,
        transforms=transforms
    )
    
    # 设置数据集
    data_module.setup()
    
    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    
    # 测试数据加载性能
    print("测试训练数据加载性能...")
    start_time = time.time()
    batch_count = 0
    try:
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 测试10个批次
                break
            batch_size = batch['input_actions'].shape[0]
            print(f"批次 {i+1}: 大小={batch_size}, 未来步数: {batch['future_step'].float().mean().item():.2f}")
            if i == 0:
                print("future_rgb_gripper:", batch['future_rgb_gripper'].shape)
                print("future_rgb_static:", batch['future_rgb_static'].shape)
            batch_count = i + 1
    except Exception as e:
        print(f"数据加载失败: {e}")
    end_time = time.time()
    print(f"成功加载{batch_count}个批次，用时: {end_time - start_time:.2f}秒")
    
    # 清理资源
    import gc
    if data_module.train_dataset:
        data_module.train_dataset.clear_cache()
    data_module = None
    train_loader = None
    gc.collect() 