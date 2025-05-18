# compact_sequence_dataset.py
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

logger = logging.getLogger(__name__)



class CompactSequenceDataset(Dataset):
    """
    紧凑格式序列数据集，从预处理后的单一数据文件中加载数据
    """
    def __init__(
        self,
        data_path: str,
        transform=None,
        use_future_image: bool = True,
        future_image_mean: float = 5.0,
        future_image_std: float = 1.7,
        use_mmap: bool = False,
        preload_data: bool = True  # 添加预加载选项
    ):
        """
        初始化紧凑序列数据集
        
        Args:
            data_path: 紧凑数据文件所在目录
            transform: 可选的数据变换函数字典
            use_future_image: 是否使用未来图像
            future_image_mean: 未来图像间隔时间步的正态分布均值
            future_image_std: 未来图像间隔时间步的正态分布标准差
            use_mmap: 是否使用内存映射加载数据
            preload_data: 是否预加载所有数据到内存
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.use_future_image = use_future_image
        self.future_image_mean = future_image_mean
        self.future_image_std = future_image_std
        self.use_mmap = use_mmap
        self.preload_data = preload_data
        
        # 数据文件路径
        self.data_file = os.path.join(data_path, "compact_data.npz")
        self.metadata_file = os.path.join(data_path, "metadata.npz")
        self.seq_index_file = os.path.join(data_path, "sequence_indices.npz")
        
        # 检查文件是否存在
        for file_path in [self.data_file, self.metadata_file, self.seq_index_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到必要的数据文件: {file_path}")
        
        print(f"初始化紧凑序列数据集: {self.data_path}")
        print(f"使用未来图像: {use_future_image}, 均值: {future_image_mean}, 标准差: {future_image_std}")
        
        # 加载元数据
        self.metadata = np.load(self.metadata_file, allow_pickle=True)
        
        # 获取基本参数
        self.observation_horizon = int(self.metadata['observation_horizon'])
        self.prediction_horizon = int(self.metadata['prediction_horizon'])
        self.language_annotations = self.metadata['language_annotations']
        
        print(f"观测窗口: {self.observation_horizon}, 预测窗口: {self.prediction_horizon}")
        print(f"语言标注数量: {len(self.language_annotations)}")
        
        # 加载序列索引
        seq_indices_data = np.load(self.seq_index_file)
        self.sequence_indices = seq_indices_data['sequence_indices']
        self.total_samples = len(self.sequence_indices)
        
        print(f"序列样本总数: {self.total_samples}")
        
        # 加载数据
        if self.preload_data:
            print("预加载所有数据到内存中...")
            try:
                start_time = time.time()
                # 直接加载整个数据文件到内存
                data_dict = np.load(self.data_file)
                
                # 创建数据字典
                self.preloaded_data = {
                    'rgb_static': data_dict['rgb_static'],
                    'rgb_gripper': data_dict['rgb_gripper'],
                    'abs_actions': data_dict['abs_actions'],
                    'rel_actions': data_dict['rel_actions']
                }
                
                # 获取数据形状
                self.data_shapes = {k: v.shape for k, v in self.preloaded_data.items()}
                
                end_time = time.time()
                print(f"数据预加载完成，耗时: {end_time - start_time:.2f}秒")
                
                # 关闭原始数据文件
                data_dict.close()
                self.data = None
            except Exception as e:
                print(f"数据预加载失败: {e}，将使用常规加载方式")
                self.preload_data = False
        
        # 如果预加载失败或未启用预加载，使用内存映射或常规加载
        if not self.preload_data:
            # 使用内存映射(mmap)加载大型数据文件
            mmap_mode = 'r' if use_mmap else None
            try:
                self.data = np.load(self.data_file, mmap_mode=mmap_mode)
                # 获取数据形状
                self.data_shapes = {
                    'rgb_static': self.data['rgb_static'].shape,
                    'rgb_gripper': self.data['rgb_gripper'].shape,
                    'abs_actions': self.data['abs_actions'].shape,
                    'rel_actions': self.data['rel_actions'].shape
                }
            except Exception as e:
                raise RuntimeError(f"加载数据文件失败: {e}")
        
        print(f"数据形状: {self.data_shapes}")
    
    def __len__(self) -> int:
        """数据集大小"""
        return self.total_samples
    
    def _sample_future_step(self):
        """生成以future_image_mean为中心的正态分布随机时间步，范围限制在[1, prediction_horizon]"""
        step = int(np.round(np.random.normal(self.future_image_mean, self.future_image_std)))
        # 确保在有效范围内
        return max(1, min(step, self.prediction_horizon))
    
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
        
        # 提取输入观测序列
        in_actions = []
        in_rgb_static = []
        in_rgb_gripper = []
        
        # 根据是否预加载选择数据源
        data_source = self.preloaded_data if self.preload_data else self.data
        
        try:
            for i in range(self.observation_horizon):
                frame_idx = frame_start_idx + i
                
                # 添加绝对动作
                in_actions.append(data_source['abs_actions'][frame_idx])
                
                # 添加RGB图像
                in_rgb_static.append(data_source['rgb_static'][frame_idx])
                in_rgb_gripper.append(data_source['rgb_gripper'][frame_idx])
            
            # 提取目标动作序列
            target_actions = []
            for i in range(self.prediction_horizon):
                frame_idx = frame_start_idx + self.observation_horizon + i
                target_actions.append(data_source['rel_actions'][frame_idx])
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
            future_frame_idx = frame_start_idx + self.observation_horizon + future_step - 1 # 减1是因为当前时刻是在观测帧的最后一帧(或者说开始帧已经占了第一帧观测帧)
            
            try:
                future_rgb_static = data_source['rgb_static'][future_frame_idx]
                future_rgb_gripper = data_source['rgb_gripper'][future_frame_idx]
            except Exception as e:
                # 示例: 获取未来图像时出错 [idx=269, future_frame=341]: index 341 is out of bounds for axis 0 with size 341
                # 起始帧idx=269, 未来帧idx=341, 总帧数=341 分析：没-1造成的，已修正
                print(f"获取未来图像时出错 [idx={idx}, future_frame={future_frame_idx}]: {e}")
                # 出错时使用最后一帧的图像
                future_rgb_static = data_source['rgb_static'][frame_start_idx + self.observation_horizon - 1]
                future_rgb_gripper = data_source['rgb_gripper'][frame_start_idx + self.observation_horizon - 1]
        
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
    
    def __del__(self):
        """析构函数，清理资源"""
        # 关闭内存映射文件
        if hasattr(self, 'data') and isinstance(self.data, np.lib.npyio.NpzFile):
            self.data.close()


class CompactSequenceDataModule(pl.LightningDataModule):
    """
    紧凑序列数据模块，用于管理训练和验证数据集
    """
    def __init__(
        self,
        dataset_path: str,
        train_folder: str = "training/compact_data",
        val_folder: str = "validation/compact_data",
        batch_size: int = 64,
        num_workers: int = 8,
        transforms: Optional[Dict[str, Any]] = None,
        use_future_image: bool = True,
        future_image_mean: float = 5.0,
        future_image_std: float = 1.7,
        use_mmap: bool = False,
        preload_data: bool = True  # 添加预加载选项
    ):
        """
        初始化紧凑序列数据模块
        
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
            use_mmap: 是否使用内存映射加载数据
            preload_data: 是否预加载所有数据到内存
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
        self.use_mmap = use_mmap
        self.preload_data = preload_data
        
        self.train_dataset = None
        self.val_dataset = None
        
        # 构建数据变换
        self.train_transforms = self._build_transforms("train") if transforms else None
        self.val_transforms = self._build_transforms("val") if transforms else None
        
        print(f"初始化紧凑序列数据模块: {self.dataset_path}")
        print(f"批次大小: {self.batch_size}, 工作进程数: {self.num_workers}")
        print(f"使用未来图像: {use_future_image}, 均值: {future_image_mean}, 标准差: {future_image_std}")
        print(f"预加载数据到内存: {preload_data}")
    
    def _build_transforms(self, split: str) -> Dict[str, Callable]:
        """构建数据变换函数"""
        if not self.transforms or split not in self.transforms:
            return None
            
        transform_dict = {}
        for key, transforms_list in self.transforms[split].items():
            transform_sequence = []
            for transform_cfg in transforms_list:
                # print(key,"获得transform_cfg:", transform_cfg)
                # print("-"*100)
               
                # if isinstance(transform_cfg, DictConfig):
                #     print("DictConfig实例判断通过")
                # if isinstance(transform_cfg, dict):
                #     print("dict实例判断通过")
                # if "_target_" in transform_cfg:
                #     print("获得target:", transform_cfg["_target_"])
                # print("-"*100)
                # 检查是否为已实例化的对象
                if (isinstance(transform_cfg, dict) or isinstance(transform_cfg, DictConfig)) and "_target_" in transform_cfg:
                    # 字典格式的配置，需要实例化
                    target = transform_cfg.pop("_target_", None)
                    # print(key,"获得target:", target)
                    if target:
                        # 处理自定义变换
                        if "RandomShiftsAug" in target:
                            transform = RandomShiftsAug(**transform_cfg)
                        elif "AddGaussianNoise" in target:
                            transform = AddGaussianNoise(**transform_cfg)
                        else:
                            # 处理torchvision变换
                            transform_class = target.split(".")[-1]
                            transform = getattr(T, transform_class)(**transform_cfg)
                        # print(key,"获得transform:", transform)
                        # print("="*100)
                        transform_sequence.append(transform)
                else:
                    # 已经是实例化的对象，直接使用
                    transform_sequence.append(transform_cfg)
                    print(key,"获得已实例化transform_cfg:", transform_cfg)
                    print("-"*100)
                    # raise ValueError(f"无法处理配置项: {transform_cfg}")
                
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
                self.train_dataset = CompactSequenceDataset(
                    data_path=train_data_path,
                    transform=self.train_transforms,
                    use_future_image=self.use_future_image,
                    future_image_mean=self.future_image_mean,
                    future_image_std=self.future_image_std,
                    use_mmap=self.use_mmap,
                    preload_data=self.preload_data
                )
                print(f"训练数据集大小: {len(self.train_dataset)}")
            except FileNotFoundError as e:
                logger.error(f"找不到训练数据: {e}")
                raise
        
        # 初始化验证数据集
        if stage == "validate" or stage is None:
            val_data_path = os.path.join(self.dataset_path, self.val_folder)
            try:
                self.val_dataset = CompactSequenceDataset(
                    data_path=val_data_path,
                    transform=self.val_transforms,
                    use_future_image=self.use_future_image,
                    future_image_mean=self.future_image_mean,
                    future_image_std=self.future_image_std,
                    use_mmap=self.use_mmap,
                    preload_data=self.preload_data
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
            drop_last=False
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


# 定义RandomShiftsAug类，参考MDT中的实现
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

# 注册自定义变换
CUSTOM_TRANSFORMS = {
    "RandomShiftsAug": RandomShiftsAug,
    "AddGaussianNoise": AddGaussianNoise
}


# 使用示例
if __name__ == "__main__":

    from omegaconf import OmegaConf
    config = 'AC_DiT/configs/ac_dit_sequence_v3.yaml'
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
    data_module = CompactSequenceDataModule(
        dataset_path="dataset",
        train_folder="calvin_debug_dataset/training/compact_data",
        val_folder="calvin_debug_dataset/validation/compact_data",
        batch_size=32,  # 设置合理的批次大小
        num_workers=2,  # 设置单线程，避免多线程冲突
        use_future_image=True,
        future_image_mean=5.0,
        future_image_std=1.7,
        use_mmap=False,
        preload_data=True,  # 启用预加载模式
        transforms=transforms
    )
    
    # 设置数据集
    data_module.setup()
    
    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # 测试数据加载性能
    print("测试训练数据加载性能...")
    start_time = time.time()
    batch_count = 0
    try:
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 减少测试批次数
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
    
    import gc
    data_module = None  # 移除引用
    train_loader = None
    val_loader = None
    gc.collect()  # 强制垃圾回收