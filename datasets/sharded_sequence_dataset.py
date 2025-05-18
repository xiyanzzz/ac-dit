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
import glob

logger = logging.getLogger(__name__)


class ShardedSequenceDataset(Dataset):
    """
    分片格式序列数据集，从多个预处理后的分片数据文件中加载数据
    """
    def __init__(
        self,
        data_path: str,
        transform=None,
        use_future_image: bool = True,
        future_image_mean: float = 5.0,
        future_image_std: float = 1.7,
        use_mmap: bool = False,
        preload_data: bool = True,  # 添加预加载选项
        max_shards_to_load: int = None,  # 最大加载分片数，为None则加载所有
        shards_to_load: List[int] = None  # 指定要加载的分片ID列表
    ):
        """
        初始化分片序列数据集
        
        Args:
            data_path: 分片数据文件所在目录
            transform: 可选的数据变换函数字典
            use_future_image: 是否使用未来图像
            future_image_mean: 未来图像间隔时间步的正态分布均值
            future_image_std: 未来图像间隔时间步的正态分布标准差
            use_mmap: 是否使用内存映射加载数据
            preload_data: 是否预加载所有数据到内存
            max_shards_to_load: 最大加载分片数，为None则加载所有
            shards_to_load: 指定要加载的分片ID列表，优先级高于max_shards_to_load
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.use_future_image = use_future_image
        self.future_image_mean = future_image_mean
        self.future_image_std = future_image_std
        self.use_mmap = use_mmap
        self.preload_data = preload_data
        self.max_shards_to_load = max_shards_to_load
        self.shards_to_load = shards_to_load
        
        # 获取核心数据文件路径
        self.metadata_file = os.path.join(data_path, "metadata.npz")
        self.seq_index_file = os.path.join(data_path, "sequence_indices.npz")
        self.shards_info_file = os.path.join(data_path, "shards_info.npz")
        
        # 检查核心文件是否存在
        for file_path in [self.metadata_file, self.seq_index_file, self.shards_info_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到必要的数据文件: {file_path}")
        
        print(f"初始化分片序列数据集: {self.data_path}")
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
        
        # 加载分片信息
        self.shards_info = np.load(self.shards_info_file, allow_pickle=True)
        self.num_shards = int(self.shards_info['num_shards'])
        
        print(f"总分片数: {self.num_shards}")
        
        # 确定要加载的分片
        if self.shards_to_load is not None:
            self.shards_to_load = [i for i in self.shards_to_load if 0 <= i < self.num_shards]
            if not self.shards_to_load:
                raise ValueError("指定的分片ID列表无效")
        elif self.max_shards_to_load is not None:
            self.shards_to_load = list(range(min(self.max_shards_to_load, self.num_shards)))
        else:
            self.shards_to_load = list(range(self.num_shards))
            
        print(f"将加载 {len(self.shards_to_load)} 个分片: {self.shards_to_load}")
        
        # 过滤序列索引，只保留在要加载的分片中的序列
        valid_indices = []
        for idx, seq_info in enumerate(self.sequence_indices):
            if len(seq_info) >= 4 and int(seq_info[3]) in self.shards_to_load:
                valid_indices.append(idx)
        
        self.valid_sequence_indices = self.sequence_indices[valid_indices]
        self.total_samples = len(self.valid_sequence_indices)
        print(f"有效序列样本总数: {self.total_samples} (原始: {len(self.sequence_indices)})")
        
        # 加载分片数据
        self.shard_data = {}
        self.preloaded_data = {}
        
        # 如果预加载数据，加载所有分片到内存
        if self.preload_data:
            print("开始预加载分片数据到内存...")
            start_time = time.time()
            
            for shard_id in self.shards_to_load:
                shard_file = os.path.join(data_path, f"shard_{shard_id:04d}.npz")
                if not os.path.exists(shard_file):
                    print(f"[警告] 分片文件不存在: {shard_file}")
                    continue
                
                print(f"加载分片 {shard_id}...")
                try:
                    data = np.load(shard_file)
                    
                    # 保存该分片的数据
                    self.preloaded_data[shard_id] = {
                        'rgb_static': data['rgb_static'],
                        'rgb_gripper': data['rgb_gripper'],
                        'abs_actions': data['abs_actions'],
                        'rel_actions': data['rel_actions'],
                        'global_start_frame': int(data['global_start_frame'])
                    }
                    
                    # 记录数据形状（用于第一个分片）
                    if not hasattr(self, 'data_shapes'):
                        self.data_shapes = {
                            'rgb_static': data['rgb_static'].shape,
                            'rgb_gripper': data['rgb_gripper'].shape,
                            'abs_actions': data['abs_actions'].shape,
                            'rel_actions': data['rel_actions'].shape
                        }
                    
                    data.close()  # 关闭文件
                    
                except Exception as e:
                    print(f"加载分片 {shard_id} 出错: {e}")
            
            end_time = time.time()
            print(f"预加载完成，加载了 {len(self.preloaded_data)} 个分片，耗时: {end_time - start_time:.2f}秒")
        else:
            # 如果不预加载，先获取数据形状
            sample_shard_id = self.shards_to_load[0]
            sample_shard_file = os.path.join(data_path, f"shard_{sample_shard_id:04d}.npz")
            try:
                data = np.load(sample_shard_file)
                self.data_shapes = {
                    'rgb_static': data['rgb_static'].shape,
                    'rgb_gripper': data['rgb_gripper'].shape,
                    'abs_actions': data['abs_actions'].shape,
                    'rel_actions': data['rel_actions'].shape
                }
                data.close()
            except Exception as e:
                raise RuntimeError(f"加载示例分片出错: {e}")
        
        # 打印数据形状信息
        print(f"数据形状: {self.data_shapes}")
    
    def __len__(self) -> int:
        """数据集大小"""
        return self.total_samples
    
    def _get_shard_data(self, shard_id: int):
        """获取指定分片的数据，如果未加载则加载"""
        if self.preload_data and shard_id in self.preloaded_data:
            return self.preloaded_data[shard_id]
        
        # 对于非预加载模式或未找到预加载数据的情况
        if shard_id not in self.shard_data:
            shard_file = os.path.join(self.data_path, f"shard_{shard_id:04d}.npz")
            if not os.path.exists(shard_file):
                raise FileNotFoundError(f"找不到分片文件: {shard_file}")
            
            # 使用内存映射模式加载数据
            mmap_mode = 'r' if self.use_mmap else None
            try:
                self.shard_data[shard_id] = np.load(shard_file, mmap_mode=mmap_mode)
            except Exception as e:
                raise RuntimeError(f"加载分片 {shard_id} 出错: {e}")
            
        return self.shard_data[shard_id]
    
    def _sample_future_step(self):
        """生成以future_image_mean为中心的正态分布随机时间步，范围限制在[1, prediction_horizon]"""
        step = int(np.round(np.random.normal(self.future_image_mean, self.future_image_std)))
        # 确保在有效范围内
        return max(1, min(step, self.prediction_horizon))
    
    def _get_global_frame_data(self, global_frame_idx, shard_id, data_key):
        """从指定分片获取全局帧索引对应的数据"""
        shard_data = self._get_shard_data(shard_id)
        
        # 计算分片内索引
        local_idx = global_frame_idx - shard_data['global_start_frame']
        
        # 确保索引在有效范围内
        if local_idx < 0 or local_idx >= len(shard_data[data_key]):
            raise IndexError(f"全局帧索引 {global_frame_idx} 转换为分片 {shard_id} 内索引 {local_idx} 超出范围")
        
        return shard_data[data_key][local_idx]
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个序列样本"""
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.total_samples}")
            
        # 获取序列索引信息
        _, lang_idx, frame_start_idx, shard_id = self.valid_sequence_indices[idx]
        shard_id = int(shard_id)
        
        # 获取语言标注
        if lang_idx >= 0 and lang_idx < len(self.language_annotations):
            lang_text = self.language_annotations[lang_idx]
        else:
            raise ValueError(f"样本 {idx} 的语言索引 {lang_idx} 无效")
        
        # 获取分片数据
        try:
            shard_data = self._get_shard_data(shard_id)
            
            # 计算在分片内的起始索引
            local_start_idx = frame_start_idx - shard_data['global_start_frame']
            
            # 提取输入观测序列
            in_actions = []
            in_rgb_static = []
            in_rgb_gripper = []
            
            for i in range(self.observation_horizon):
                local_idx = local_start_idx + i
                
                # 添加绝对动作
                in_actions.append(shard_data['abs_actions'][local_idx])
                
                # 添加RGB图像
                in_rgb_static.append(shard_data['rgb_static'][local_idx])
                in_rgb_gripper.append(shard_data['rgb_gripper'][local_idx])
            
            # 提取目标动作序列
            target_actions = []
            for i in range(self.prediction_horizon):
                local_idx = local_start_idx + self.observation_horizon + i
                target_actions.append(shard_data['rel_actions'][local_idx])
        except Exception as e:
            print(f"访问数据时出错 [idx={idx}, shard={shard_id}, local_start={local_start_idx}]: {e}")
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
            future_local_idx = local_start_idx + self.observation_horizon + future_step - 1
            
            try:
                future_rgb_static = shard_data['rgb_static'][future_local_idx]
                future_rgb_gripper = shard_data['rgb_gripper'][future_local_idx]
            except Exception as e:
                print(f"获取未来图像时出错 [idx={idx}, future_local_idx={future_local_idx}]: {e}")
                # 出错时使用最后一帧的图像
                future_rgb_static = shard_data['rgb_static'][local_start_idx + self.observation_horizon - 1]
                future_rgb_gripper = shard_data['rgb_gripper'][local_start_idx + self.observation_horizon - 1]
        
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
            'shard_id': shard_id,
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
        for shard_id, shard_data in self.shard_data.items():
            if isinstance(shard_data, np.lib.npyio.NpzFile):
                shard_data.close()


class ShardedSequenceDataModule(pl.LightningDataModule):
    """
    分片序列数据模块，用于管理训练和验证数据集
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
        preload_data: bool = True,  # 添加预加载选项
        max_train_shards: int = None,  # 训练数据最大分片数
        max_val_shards: int = None,    # 验证数据最大分片数
        train_shards: List[int] = None,  # 指定训练分片ID
        val_shards: List[int] = None,    # 指定验证分片ID
    ):
        """
        初始化分片序列数据模块
        
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
            max_train_shards: 训练数据最大分片数，为None则使用所有
            max_val_shards: 验证数据最大分片数，为None则使用所有
            train_shards: 指定要加载的训练分片ID列表
            val_shards: 指定要加载的验证分片ID列表
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
        self.max_train_shards = max_train_shards
        self.max_val_shards = max_val_shards
        self.train_shards = train_shards
        self.val_shards = val_shards
        
        self.train_dataset = None
        self.val_dataset = None
        
        # 构建数据变换
        self.train_transforms = self._build_transforms("train") if transforms else None
        self.val_transforms = self._build_transforms("val") if transforms else None
        
        print(f"初始化分片序列数据模块: {self.dataset_path}")
        print(f"批次大小: {self.batch_size}, 工作进程数: {self.num_workers}")
        print(f"使用未来图像: {use_future_image}, 均值: {future_image_mean}, 标准差: {future_image_std}")
        print(f"预加载数据到内存: {preload_data}")
        if max_train_shards:
            print(f"训练数据最大分片数: {max_train_shards}")
        if max_val_shards:
            print(f"验证数据最大分片数: {max_val_shards}")
        if train_shards:
            print(f"指定训练分片: {train_shards}")
        if val_shards:
            print(f"指定验证分片: {val_shards}")
    
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
                    target = transform_cfg.pop("_target_", None)
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
                        transform_sequence.append(transform)
                else:
                    # 已经是实例化的对象，直接使用
                    transform_sequence.append(transform_cfg)
                
            if transform_sequence:
                transform_dict[key] = T.Compose(transform_sequence)

        print(f"成功构建{split}的变换字典: {transform_dict}")
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
                self.train_dataset = ShardedSequenceDataset(
                    data_path=train_data_path,
                    transform=self.train_transforms,
                    use_future_image=self.use_future_image,
                    future_image_mean=self.future_image_mean,
                    future_image_std=self.future_image_std,
                    use_mmap=self.use_mmap,
                    preload_data=self.preload_data,
                    max_shards_to_load=self.max_train_shards,
                    shards_to_load=self.train_shards
                )
                print(f"训练数据集大小: {len(self.train_dataset)}")
            except FileNotFoundError as e:
                logger.error(f"找不到训练数据: {e}")
                raise
        
        # 初始化验证数据集
        if stage == "validate" or stage is None:
            val_data_path = os.path.join(self.dataset_path, self.val_folder)
            try:
                self.val_dataset = ShardedSequenceDataset(
                    data_path=val_data_path,
                    transform=self.val_transforms,
                    use_future_image=self.use_future_image,
                    future_image_mean=self.future_image_mean,
                    future_image_std=self.future_image_std,
                    use_mmap=self.use_mmap,
                    preload_data=self.preload_data,
                    max_shards_to_load=self.max_val_shards,
                    shards_to_load=self.val_shards
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

    transforms = cfg.data.transforms if 'data' in cfg and 'transforms' in cfg.data else None

    if transforms:
        print(f"变换配置:")
        for key, transform in transforms.items():
            print(f"  - {key}: {transform}")
    else:
        print("未配置数据变换")

    # 创建数据模块
    data_module = ShardedSequenceDataModule(
        dataset_path="dataset",
        train_folder="calvin_debug_dataset/training/compact_data",
        val_folder="calvin_debug_dataset/validation/compact_data",
        batch_size=32,
        num_workers=2,
        use_future_image=True,
        future_image_mean=5.0,
        future_image_std=1.7,
        use_mmap=False,
        preload_data=True,
        transforms=transforms,
        max_train_shards=None,  # 加载所有训练分片
        max_val_shards=None,    # 加载所有验证分片
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
            if i >= 10:  # 测试10个批次
                break
            batch_size = batch['input_actions'].shape[0]
            print(f"批次 {i+1}: 大小={batch_size}, 分片ID: {batch['shard_id'][0].item()}")
            if i == 0:
                if 'future_rgb_gripper' in batch:
                    print("future_rgb_gripper:", batch['future_rgb_gripper'].shape)
                    print("future_rgb_static:", batch['future_rgb_static'].shape)
                print("input_actions:", batch['input_actions'].shape)
                print("target_actions:", batch['target_actions'].shape)
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