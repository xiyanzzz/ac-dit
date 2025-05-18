import logging
import os
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import glob

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """
    加载为AC_DiT模型准备的序列数据
    """
    def __init__(
        self,
        data_path: str,
        transform=None,
        use_lang: bool = True,
        lazy_loading: bool = True,  # 保留参数但强制使用延迟加载
        max_samples: int = None
    ):
        """
        初始化序列数据集
        
        Args:
            data_path: 序列数据文件路径或目录
            transform: 可选的数据变换函数字典，针对不同类型的数据应用不同的变换
            use_lang: 是否使用语言条件
            lazy_loading: 此参数已弃用，始终使用延迟加载模式
            max_samples: 最大样本数量，用于限制数据集大小
        """
        if not lazy_loading:
            logger.warning("非延迟加载模式已弃用，将强制使用延迟加载模式")
            
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.use_lang = use_lang
        self.lazy_loading = True  # 强制使用延迟加载
        self.max_samples = max_samples
        
        print(f"初始化序列数据集: {self.data_path}")
        
        # 检查数据路径
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件或目录: {self.data_path}")
            
        # 确定数据集格式（单个文件或分片）
        self.is_sharded = False
        self.shard_files = []
        self.metadata = None
        self.samples_per_shard = []
        self.shard_offsets = []
        self.total_samples = 0
        
        # 检查是否存在元数据文件
        metadata_file = None
        for f in os.listdir(self.data_path):
            if f.endswith('_metadata.npz'):
                metadata_file = os.path.join(self.data_path, f)
                break
        
        if metadata_file:
            print(f"找到元数据文件: {metadata_file}")
            self.metadata = np.load(metadata_file, allow_pickle=True)
            
            # 获取分片文件列表
            self.shard_files = []
            for f in os.listdir(self.data_path):
                if f.startswith(os.path.basename(metadata_file).replace('_metadata.npz', '_shard')) and f.endswith('.npz'):
                    self.shard_files.append(os.path.join(self.data_path, f))
            
            if self.shard_files:
                print(f"找到 {len(self.shard_files)} 个分片文件")
                self.is_sharded = True
                self.shard_files.sort()  # 确保按顺序加载
            else:
                raise FileNotFoundError(f"未找到分片文件: {self.data_path}")
        else:
            raise FileNotFoundError(f"未找到元数据文件: {self.data_path}")
        
        # 加载数据
        if self.is_sharded:
            # 分片数据模式 - 计算每个分片的样本数量
            current_offset = 0
            for i, shard_file in enumerate(self.shard_files):
                print(f"正在处理分片 {i}: {os.path.basename(shard_file)}")
                if i == 0:
                    # 加载第一个分片以获取样本结构
                    shard_data = np.load(shard_file, allow_pickle=True)
                    shard_samples = len(shard_data['input_abs_actions'])
                    print(f"  - 从分片文件读取到 {shard_samples} 个样本")
                    
                    # 记录第一个分片的数据结构
                    self.sample_shape = {
                        'input_abs_actions': shard_data['input_abs_actions'][0].shape,
                        'input_rgb_static': shard_data['input_rgb_static'][0].shape,
                        'input_rgb_gripper': shard_data['input_rgb_gripper'][0].shape,
                        'target_rel_actions': shard_data['target_rel_actions'][0].shape,
                    }
                    print(f"  - 记录样本形状: {self.sample_shape}")
                    
                    # 如果需要立即加载语言数据
                    if self.use_lang and 'language_annotations' in self.metadata:
                        self.language_annotations = self.metadata['language_annotations']
                        self.has_language = True
                        print(f"  - 从元数据加载语言标注: {len(self.language_annotations)} 条")
                    else:
                        self.has_language = False
                        print("  - 未找到语言标注数据")
                    
                    self.samples_per_shard.append(shard_samples) # 统计每个分片的样本数量
                    self.shard_offsets.append(current_offset) # 统计每个分片的偏移量
                    current_offset += shard_samples # 更新总样本数
                    print(f"  - 当前总样本数: {current_offset}")
                    
                    # 延迟加载模式下，释放内存
                    del shard_data
                else:
                    # 对于其他分片，只获取样本数量
                    shard_data = np.load(shard_file, allow_pickle=True)
                    shard_samples = len(shard_data['input_abs_actions'])
                    self.samples_per_shard.append(shard_samples)
                    self.shard_offsets.append(current_offset)
                    current_offset += shard_samples
                    print(f"  - 当前总样本数: {current_offset}")
                    del shard_data
            
            self.total_samples = current_offset
            print(f"总样本数: {self.total_samples}")
            
            # 限制样本数量
            if self.max_samples and self.max_samples < self.total_samples:
                print(f"限制样本数量为 {self.max_samples}")
                self.total_samples = self.max_samples
                
                # 更新分片偏移量
                new_offsets = []
                current_offset = 0
                for samples in self.samples_per_shard:
                    if current_offset >= self.max_samples:
                        break
                    new_offsets.append(current_offset)
                    current_offset += min(samples, self.max_samples - current_offset)
                self.shard_offsets = new_offsets
        else:
            raise FileNotFoundError(f"出错，未找到分片文件，且暂不支持单文件模式")
        # 以下注释，暂不考虑单文件模式
        # else:
        #     # 单文件模式 - 直接加载
        #     self.data = np.load(self.data_path, allow_pickle=True)
            
        #     # 提取数据
        #     self.input_abs_actions = self.data['input_abs_actions']  # [N, T_in, 7]
        #     self.input_rgb_static = self.data['input_rgb_static']    # [N, T_in, H, W, 3]
        #     self.input_rgb_gripper = self.data['input_rgb_gripper']  # [N, T_in, H, W, 3]
        #     self.target_rel_actions = self.data['target_rel_actions'] # [N, T_pred, 7]
        #     self.sequence_indices = self.data['sequence_indices']    # 序列对应的文件索引
        #     self.lang_indices = self.data['lang_indices']            # 语言标注索引
            
        #     # 如果使用语言条件且数据中有语言标注
        #     self.has_language = False
        #     if self.use_lang and 'language_annotations' in self.data:
        #         self.language_annotations = self.data['language_annotations']  # 语言描述
        #         self.has_language = True
        #         print(f"数据集包含语言标注: {len(self.language_annotations)} 条")
                
        #         # 统计有效语言标注的数量
        #         valid_lang_count = sum(1 for idx in self.lang_indices if idx >= 0)
        #         print(f"有效语言标注的序列数量: {valid_lang_count}/{len(self.lang_indices)}")
            
        #     # 限制样本数量
        #     self.total_samples = len(self.input_abs_actions)
        #     if self.max_samples and self.max_samples < self.total_samples:
        #         print(f"限制样本数量为 {self.max_samples}")
        #         self.total_samples = self.max_samples
            
        #     print(f"序列数据加载完成。总序列数: {self.total_samples}")
        #     print(f"  - 输入动作形状: {self.input_abs_actions.shape}")
        #     print(f"  - 输入RGB静态图像形状: {self.input_rgb_static.shape}")
        #     print(f"  - 输入RGB夹爪图像形状: {self.input_rgb_gripper.shape}")
        #     print(f"  - 目标动作形状: {self.target_rel_actions.shape}")
        
    def _locate_shard(self, idx: int) -> Tuple[int, int]:
        """定位样本在哪个分片及其局部索引"""
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.total_samples}")
            
        # 二分查找定位分片
        left, right = 0, len(self.shard_offsets) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.shard_offsets[mid] <= idx:
                if mid == len(self.shard_offsets) - 1 or self.shard_offsets[mid + 1] > idx:
                    # 找到了正确的分片
                    shard_idx = mid
                    local_idx = idx - self.shard_offsets[mid]
                    return shard_idx, local_idx
                left = mid + 1
            else:
                right = mid - 1
                
        # 这不应该发生
        raise RuntimeError(f"无法定位索引 {idx} 所在的分片")
        
    def __len__(self) -> int:
        """数据集大小"""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个数据项"""
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.total_samples}")
            
        if self.is_sharded:
            # 延迟加载模式 - 从分片加载数据
            shard_idx, local_idx = self._locate_shard(idx)
            shard_file = self.shard_files[shard_idx]
            
            # 加载分片数据
            with np.load(shard_file, allow_pickle=True) as shard_data:
                # 提取样本数据
                in_actions = torch.from_numpy(shard_data['input_abs_actions'][local_idx]).float()
                in_rgb_static = torch.from_numpy(shard_data['input_rgb_static'][local_idx]).float() / 255.0
                in_rgb_gripper = torch.from_numpy(shard_data['input_rgb_gripper'][local_idx]).float() / 255.0
                target_actions = torch.from_numpy(shard_data['target_rel_actions'][local_idx]).float()
                lang_idx = int(shard_data['lang_indices'][local_idx])
        else:
            raise FileNotFoundError(f"__getitem__()方法出错，未找到分片文件，且暂不支持单文件模式")
            # # 单文件模式 - 直接从内存获取数据
            # in_actions = torch.from_numpy(self.input_abs_actions[idx]).float()
            # in_rgb_static = torch.from_numpy(self.input_rgb_static[idx]).float() / 255.0
            # in_rgb_gripper = torch.from_numpy(self.input_rgb_gripper[idx]).float() / 255.0
            # target_actions = torch.from_numpy(self.target_rel_actions[idx]).float()
            # lang_idx = int(self.lang_indices[idx]) if hasattr(self, 'lang_indices') else -1
        
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
        }
        
        # 如果有语言条件并且该序列有对应的语言标注
        if hasattr(self, 'has_language') and self.has_language and lang_idx >= 0:
            lang_text = self.language_annotations[lang_idx]
            result['lang_text'] = lang_text
        elif hasattr(self, 'has_language') and self.has_language:
            raise FileNotFoundError(f"__getitem__()方法出错，未找到正确语言标注数据")
            # # 如果没有对应的语言标注，使用随机一个
            # lang_idx = np.random.randint(0, len(self.language_annotations))
            # lang_text = self.language_annotations[lang_idx]
            # result['lang_text'] = lang_text
            # result['is_random_lang'] = True
        
        # 应用数据变换（如果提供）
        if self.transform is not None:
            # 对各个组件分别应用对应的变换
            if 'rgb_static' in self.transform and 'rgb_obs' in result and 'rgb_static' in result['rgb_obs']:
                # 对每个时间步的静态RGB图像应用变换
                frames = []
                for t in range(result['rgb_obs']['rgb_static'].size(0)):
                    # 确保每帧图像是 [C, H, W] 格式
                    frame = result['rgb_obs']['rgb_static'][t]  # [C, H, W]
                    # 应用变换
                    try:
                        frame = self.transform['rgb_static'](frame)
                        frames.append(frame)
                    except Exception as e:
                        print(f"应用rgb_static变换时出错: {e}, 形状: {frame.shape}")
                        frames.append(frame)  # 使用原始帧
                
                # 堆叠所有帧
                if frames:
                    result['rgb_obs']['rgb_static'] = torch.stack(frames)
                
            if 'rgb_gripper' in self.transform and 'rgb_obs' in result and 'rgb_gripper' in result['rgb_obs']:
                # 对每个时间步的夹爪RGB图像应用变换
                frames = []
                for t in range(result['rgb_obs']['rgb_gripper'].size(0)):
                    # 确保每帧图像是 [C, H, W] 格式
                    frame = result['rgb_obs']['rgb_gripper'][t]  # [C, H, W]
                    # 应用变换
                    try:
                        frame = self.transform['rgb_gripper'](frame)
                        frames.append(frame)
                    except Exception as e:
                        print(f"应用rgb_gripper变换时出错: {e}, 形状: {frame.shape}")
                        frames.append(frame)  # 使用原始帧
                
                # 堆叠所有帧
                if frames:
                    result['rgb_obs']['rgb_gripper'] = torch.stack(frames)
                
            if 'actions' in self.transform and 'input_actions' in result:
                try:
                    result['input_actions'] = self.transform['actions'](result['input_actions'])
                except Exception as e:
                    print(f"应用input_actions变换时出错: {e}, 形状: {result['input_actions'].shape}")
                
            if 'actions' in self.transform and 'target_actions' in result:
                try:
                    result['target_actions'] = self.transform['actions'](result['target_actions'])
                except Exception as e:
                    print(f"应用target_actions变换时出错: {e}, 形状: {result['target_actions'].shape}")
                
            if 'language' in self.transform and 'lang_text' in result:
                # 对语言标注只应用噪声，不改变文本内容
                pass
            
        return result


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


class SequenceDataModule(pl.LightningDataModule):
    """
    序列数据模块，用于管理训练和验证数据集
    """
    def __init__(
        self,
        dataset_path: str,
        train_suffix: str = "seq_data_in2_pred10_lang.npz",
        val_suffix: str = "seq_data_in2_pred10_lang.npz",
        train_folder: str = "training/sequence_data",
        val_folder: str = "validation/sequence_data",
        env: str = "debug",
        batch_size: int = 32,
        num_workers: int = 8,
        use_lang: bool = True,
        lazy_loading: bool = True,
        max_train_samples: int = None,
        max_val_samples: int = None,
        transforms: Optional[Dict[str, Any]] = None
    ):
        """
        初始化序列数据模块
        
        Args:
            dataset_path: 数据集根路径
            train_suffix: 训练数据文件名后缀
            val_suffix: 验证数据文件名后缀
            train_folder: 训练数据文件夹
            val_folder: 验证数据文件夹
            env: 环境名称
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            use_lang: 是否使用语言条件
            lazy_loading: 是否延迟加载数据（节省内存）
            max_train_samples: 最大训练样本数量
            max_val_samples: 最大验证样本数量
            transforms: 数据变换配置
        """
        if not lazy_loading:
            logger.warning("非延迟加载模式已弃用，将强制使用延迟加载模式")
            
        super().__init__()
        self.dataset_path = dataset_path
        self.train_suffix = train_suffix
        self.val_suffix = val_suffix
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.env = env
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_lang = use_lang
        self.lazy_loading = True  # 强制使用延迟加载
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.transforms = transforms
        
        self.train_dataset = None
        self.val_dataset = None
        
        # 构建数据变换
        self.train_transforms = self._build_transforms("train") if transforms else None
        self.val_transforms = self._build_transforms("val") if transforms else None
        
        print(f"初始化序列数据模块: {self.dataset_path}")
        print(f"环境: {self.env}, 批次大小: {self.batch_size}")
        print(f"使用语言条件: {self.use_lang}")
        print(f"延迟加载: {self.lazy_loading}")
        print(f"使用数据变换: {transforms is not None}")
    
    def _build_transforms(self, split: str) -> Dict[str, Callable]:
        """构建数据变换函数"""
        if not self.transforms or split not in self.transforms:
            return None
            
        transform_dict = {}
        for key, transforms_list in self.transforms[split].items():
            transform_sequence = []
            for transform_cfg in transforms_list:
                # 检查是否为已实例化的对象
                if isinstance(transform_cfg, dict) and "_target_" in transform_cfg:
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
                
        return transform_dict

    def prepare_data(self):
        """准备数据，检查文件是否存在"""
        train_data_path = os.path.join(self.dataset_path, self.train_folder)
        val_data_path = os.path.join(self.dataset_path, self.val_folder)
        
        # 检查训练数据
        if not os.path.exists(train_data_path):
            logger.warning(f"找不到训练数据目录: {train_data_path}")
            logger.warning(f"请先运行预处理脚本: python preprocess/extract_sequence_data.py")
        
        # 检查验证数据
        if not os.path.exists(val_data_path):
            logger.warning(f"找不到验证数据目录: {val_data_path}")
            logger.warning(f"请先运行预处理脚本: python preprocess/extract_sequence_data.py")
    
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
                self.train_dataset = SequenceDataset(
                    data_path=train_data_path,
                    use_lang=self.use_lang,
                    lazy_loading=self.lazy_loading,
                    max_samples=self.max_train_samples,
                    transform=self.train_transforms
                )
                print(f"训练数据集大小: {len(self.train_dataset)}")
            except FileNotFoundError:
                logger.error(f"找不到训练数据目录: {train_data_path}")
                logger.error(f"请先运行预处理脚本: python preprocess/extract_sequence_data.py")
                raise
        
        # 初始化验证数据集
        if stage == "validate" or stage is None:
            val_data_path = os.path.join(self.dataset_path, self.val_folder)
            try:
                self.val_dataset = SequenceDataset(
                    data_path=val_data_path,
                    use_lang=self.use_lang,
                    lazy_loading=self.lazy_loading,
                    max_samples=self.max_val_samples,
                    transform=self.val_transforms
                )
                print(f"验证数据集大小: {len(self.val_dataset)}")
            except FileNotFoundError:
                logger.error(f"找不到验证数据目录: {val_data_path}")
                logger.error(f"请先运行预处理脚本: python preprocess/extract_sequence_data.py")
                raise
    
    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器"""
        if self.train_dataset is None:
            logger.warning("训练数据集未初始化，自动调用setup('fit')方法进行初始化...")
            try:
                self.setup(stage="fit")
            except Exception as e:
                logger.error(f"训练数据集初始化失败: {e}")
                raise RuntimeError("训练数据集初始化失败，请检查数据路径和配置。") from e
            
            if self.train_dataset is None:
                raise RuntimeError("训练数据集初始化后仍然为None，请检查setup方法的实现。")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False  # 确保使用所有样本
        )
    
    def val_dataloader(self) -> DataLoader:
        """返回验证数据加载器"""
        if self.val_dataset is None:
            logger.warning("验证数据集未初始化，自动调用setup('validate')方法进行初始化...")
            try:
                self.setup(stage="validate")
            except Exception as e:
                logger.error(f"验证数据集初始化失败: {e}")
                raise RuntimeError("验证数据集初始化失败，请检查数据路径和配置。") from e
            
            if self.val_dataset is None:
                raise RuntimeError("验证数据集初始化后仍然为None，请检查setup方法的实现。")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False  # 确保使用所有样本
        ) 