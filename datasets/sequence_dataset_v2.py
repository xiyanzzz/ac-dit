# optimized_sequence_dataset.py
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import threading
import queue
import time

logger = logging.getLogger(__name__)

class OptimizedSequenceDataset(Dataset):
    """
    优化版序列数据集，实现分片缓存和预加载
    """
    def __init__(
        self,
        data_path: str,
        transform=None,
        shard_cache_size: int = 3,      # 缓存分片数量
        preload_next: bool = True,      # 是否预加载下一个可能的分片
        prefetch_queue_size: int = 2    # 预取队列大小
    ):
        """
        初始化优化的序列数据集
        
        Args:
            data_path: 序列数据文件路径或目录
            transform: 可选的数据变换函数字典
            shard_cache_size: 缓存多少个分片在内存中
            preload_next: 是否预加载下一个可能的分片
            prefetch_queue_size: 预取队列大小
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.shard_cache_size = shard_cache_size
        self.preload_next = preload_next
        
        # 初始化缓存和预取相关变量
        self.shard_cache = {}  # 分片缓存: {shard_idx: shard_data}
        self.shard_cache_lock = threading.RLock()  # 线程锁保护缓存访问
        self.shard_access_history = []  # 最近访问的分片索引
        self.prefetch_queue = queue.Queue(maxsize=prefetch_queue_size)
        self.prefetch_thread = None
        self.stop_prefetch = False
        
        print(f"初始化优化序列数据集: {self.data_path}")
        print(f"分片缓存大小: {self.shard_cache_size}, 预加载下一分片: {self.preload_next}")
        
        # 检查数据路径
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件或目录: {self.data_path}")
            
        # 查找并加载元数据文件
        metadata_file = None
        for f in os.listdir(self.data_path):
            if f.endswith('_metadata.npz'):
                metadata_file = os.path.join(self.data_path, f)
                break
        
        if not metadata_file:
            raise FileNotFoundError(f"未找到元数据文件: {self.data_path}")
            
        print(f"找到元数据文件: {metadata_file}")
        self.metadata = np.load(metadata_file, allow_pickle=True)
        
        # 加载语言标注数据
        if 'language_annotations' in self.metadata:
            self.language_annotations = self.metadata['language_annotations']
            print(f"加载了 {len(self.language_annotations)} 条语言标注")
        else:
            raise ValueError("元数据中缺少语言标注数据，该优化版要求必须有语言标注")
            
        # 获取分片文件列表
        self.shard_files = []
        for f in os.listdir(self.data_path):
            if f.startswith(os.path.basename(metadata_file).replace('_metadata.npz', '_shard')) and f.endswith('.npz'):
                self.shard_files.append(os.path.join(self.data_path, f))
        
        if not self.shard_files:
            raise FileNotFoundError(f"未找到分片文件: {self.data_path}")
            
        self.shard_files.sort()  # 确保按顺序排列
        print(f"找到 {len(self.shard_files)} 个分片文件")
        
        # 计算每个分片的样本数和偏移量
        self.samples_per_shard = []
        self.shard_offsets = []
        current_offset = 0
        
        # 加载第一个分片以获取样本数量和形状
        print(f"加载第一个分片以初始化数据集信息")
        first_shard = np.load(self.shard_files[0], allow_pickle=True)
        first_shard_samples = len(first_shard['input_abs_actions'])
        
        # 记录样本形状
        self.sample_shape = {
            'input_abs_actions': first_shard['input_abs_actions'][0].shape,
            'input_rgb_static': first_shard['input_rgb_static'][0].shape,
            'input_rgb_gripper': first_shard['input_rgb_gripper'][0].shape,
            'target_rel_actions': first_shard['target_rel_actions'][0].shape,
        }
        print(f"样本形状: {self.sample_shape}")
        
        # 初始化第一个分片缓存
        with self.shard_cache_lock:
            self.shard_cache[0] = {
                'input_abs_actions': first_shard['input_abs_actions'],
                'input_rgb_static': first_shard['input_rgb_static'],
                'input_rgb_gripper': first_shard['input_rgb_gripper'],
                'target_rel_actions': first_shard['target_rel_actions'],
                'lang_indices': first_shard['lang_indices'],
            }
            self.shard_access_history.append(0)
        
        # 记录第一个分片信息
        self.samples_per_shard.append(first_shard_samples)
        self.shard_offsets.append(current_offset)
        current_offset += first_shard_samples
        
        # 对其余分片，只获取样本数量
        for i in range(1, len(self.shard_files)):
            try:
                # 使用mmap_mode='r'以节省内存
                shard_data = np.load(self.shard_files[i], allow_pickle=True, mmap_mode='r')
                shard_samples = len(shard_data['input_abs_actions'])
                self.samples_per_shard.append(shard_samples)
                self.shard_offsets.append(current_offset)
                current_offset += shard_samples
                del shard_data  # 释放内存
            except Exception as e:
                print(f"警告: 读取分片 {i} 时出错: {e}")
                # 假定与前一分片相同大小
                self.samples_per_shard.append(self.samples_per_shard[-1])
                self.shard_offsets.append(current_offset)
                current_offset += self.samples_per_shard[-1]
        
        self.total_samples = current_offset
        print(f"总样本数: {self.total_samples}")
        
        # 启动预取线程
        if self.preload_next:
            self._start_prefetch_thread()
    
    def _start_prefetch_thread(self):
        """启动预取线程"""
        def prefetch_worker():
            while not self.stop_prefetch:
                try:
                    next_shard_idx = self._predict_next_shard()
                    if next_shard_idx is None:
                        # 没有需要预取的分片，等待一段时间
                        time.sleep(0.5)
                        continue
                        
                    if next_shard_idx < len(self.shard_files):
                        # 预测下一个可能需要的分片
                        next_shard_idx = self._predict_next_shard()
                        if next_shard_idx is not None and next_shard_idx < len(self.shard_files):
                            # 检查是否已在缓存中
                            with self.shard_cache_lock:
                                if next_shard_idx not in self.shard_cache:
                                    # 放入预取队列
                                    if not self.prefetch_queue.full():
                                        self.prefetch_queue.put(next_shard_idx, block=False)
                                        # 实际加载分片
                                        try:
                                            shard_file = self.shard_files[next_shard_idx]
                                            shard_data = np.load(shard_file, allow_pickle=True)
                                            # 准备数据字典
                                            cache_data = {
                                                'input_abs_actions': shard_data['input_abs_actions'],
                                                'input_rgb_static': shard_data['input_rgb_static'],
                                                'input_rgb_gripper': shard_data['input_rgb_gripper'],
                                                'target_rel_actions': shard_data['target_rel_actions'],
                                                'lang_indices': shard_data['lang_indices'],
                                            }
                                            
                                            # 更新缓存
                                            with self.shard_cache_lock:
                                                if len(self.shard_cache) >= self.shard_cache_size:
                                                    # LRU策略：删除最早访问的分片
                                                    for old_idx in self.shard_access_history:
                                                        if old_idx in self.shard_cache and old_idx != next_shard_idx:
                                                            del self.shard_cache[old_idx]
                                                            break
                                                self.shard_cache[next_shard_idx] = cache_data
                                        except Exception as e:
                                            print(f"预取分片 {next_shard_idx} 时出错: {e}")
                                        finally:
                                            # 从队列移除
                                            try:
                                                self.prefetch_queue.get(block=False)
                                            except queue.Empty:
                                                pass
                except Exception as e:
                    print(f"预取线程出错: {e}")
                    time.sleep(1)  # 出错后短暂暂停
                
        # 创建并启动线程
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _predict_next_shard(self) -> Optional[int]:
        with self.shard_cache_lock:
            if not self.shard_access_history:
                return 0
            
            last_shard = self.shard_access_history[-1]
            next_shard = last_shard + 1
            
            # 添加循环检查
            if next_shard >= len(self.shard_files):
                return None  # 或者返回0重新开始
            
            # 检查是否已缓存
            if next_shard in self.shard_cache:
                return None  # 已缓存则不需预加载
                
            return next_shard
    
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
    
    def _get_shard_data(self, shard_idx: int) -> Dict:
        """获取分片数据，优先从缓存读取"""
        with self.shard_cache_lock:
            # 更新访问历史
            if shard_idx in self.shard_access_history:
                self.shard_access_history.remove(shard_idx)
            self.shard_access_history.append(shard_idx)
            # 如果历史记录过长，删除旧记录
            if len(self.shard_access_history) > self.shard_cache_size * 2:
                self.shard_access_history = self.shard_access_history[-self.shard_cache_size:]
                
            # 检查缓存
            if shard_idx in self.shard_cache:
                return self.shard_cache[shard_idx]
        
        # 如果不在缓存中，加载分片
        shard_file = self.shard_files[shard_idx]
        shard_data = np.load(shard_file, allow_pickle=True)
        
        # 准备数据字典
        cache_data = {
            'input_abs_actions': shard_data['input_abs_actions'],
            'input_rgb_static': shard_data['input_rgb_static'],
            'input_rgb_gripper': shard_data['input_rgb_gripper'],
            'target_rel_actions': shard_data['target_rel_actions'],
            'lang_indices': shard_data['lang_indices'],
        }
        
        # 更新缓存
        with self.shard_cache_lock:
            if len(self.shard_cache) >= self.shard_cache_size:
                # LRU策略：删除最早访问的分片
                for old_idx in self.shard_access_history[:-1]:  # 除了最新访问的
                    if old_idx in self.shard_cache and old_idx != shard_idx:
                        del self.shard_cache[old_idx]
                        break
            self.shard_cache[shard_idx] = cache_data
            
        return cache_data
        
    def __len__(self) -> int:
        """数据集大小"""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个数据项"""
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.total_samples}")
            
        # 定位分片和局部索引
        shard_idx, local_idx = self._locate_shard(idx)
        
        # 获取分片数据（优先从缓存读取）
        shard_data = self._get_shard_data(shard_idx)
        
        # 提取样本数据
        in_actions = torch.from_numpy(shard_data['input_abs_actions'][local_idx]).float()
        in_rgb_static = torch.from_numpy(shard_data['input_rgb_static'][local_idx]).float() / 255.0
        in_rgb_gripper = torch.from_numpy(shard_data['input_rgb_gripper'][local_idx]).float() / 255.0
        target_actions = torch.from_numpy(shard_data['target_rel_actions'][local_idx]).float()
        lang_idx = int(shard_data['lang_indices'][local_idx])
        
        # 转换图像格式：从 [T, H, W, C] 到 [T, C, H, W]
        in_rgb_static = in_rgb_static.permute(0, 3, 1, 2)
        in_rgb_gripper = in_rgb_gripper.permute(0, 3, 1, 2)
        
        # 获取语言标注
        if lang_idx >= 0 and lang_idx < len(self.language_annotations):
            lang_text = self.language_annotations[lang_idx]
        else:
            raise ValueError(f"样本 {idx} 的语言索引 {lang_idx} 无效")

        
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
            
        return result
    
    def __del__(self):
        """析构函数，清理资源"""
        self.stop_prefetch = True
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)


class OptimizedSequenceDataModule(pl.LightningDataModule):
    """
    优化版序列数据模块，用于管理训练和验证数据集
    """
    def __init__(
        self,
        dataset_path: str,
        train_folder: str = "training/sequence_data",
        val_folder: str = "validation/sequence_data",
        batch_size: int = 128,
        num_workers: int = 8,
        shard_cache_size: int = 3,
        preload_next: bool = True,
        transforms: Optional[Dict[str, Any]] = None
    ):
        """
        初始化优化序列数据模块
        
        Args:
            dataset_path: 数据集根路径
            train_folder: 训练数据文件夹
            val_folder: 验证数据文件夹
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            shard_cache_size: 缓存分片数量
            preload_next: 是否预加载下一个可能的分片
            transforms: 数据变换配置
        """            
        super().__init__()
        self.dataset_path = dataset_path
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shard_cache_size = shard_cache_size
        self.preload_next = preload_next
        self.transforms = transforms
        
        self.train_dataset = None
        self.val_dataset = None
        
        # 构建数据变换
        self.train_transforms = self._build_transforms("train") if transforms else None
        self.val_transforms = self._build_transforms("val") if transforms else None
        
        print(f"初始化优化序列数据模块: {self.dataset_path}")
        print(f"批次大小: {self.batch_size}, 工作进程数: {self.num_workers}")
        print(f"分片缓存大小: {self.shard_cache_size}, 预加载下一分片: {self.preload_next}")
    
    def _build_transforms(self, split: str) -> Dict[str, Callable]:
        """构建数据变换函数"""
        # 此处可以复用原始数据集的变换构建逻辑
        # 简化起见，这里假设transforms字典已经包含了需要的变换
        if not self.transforms or split not in self.transforms:
            return None
        return self.transforms[split]

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
                self.train_dataset = OptimizedSequenceDataset(
                    data_path=train_data_path,
                    transform=self.train_transforms,
                    shard_cache_size=self.shard_cache_size,
                    preload_next=self.preload_next
                )
                print(f"训练数据集大小: {len(self.train_dataset)}")
            except FileNotFoundError as e:
                logger.error(f"找不到训练数据目录: {e}")
                raise
        
        # 初始化验证数据集
        if stage == "validate" or stage is None:
            val_data_path = os.path.join(self.dataset_path, self.val_folder)
            try:
                self.val_dataset = OptimizedSequenceDataset(
                    data_path=val_data_path,
                    transform=self.val_transforms,
                    shard_cache_size=self.shard_cache_size,
                    preload_next=self.preload_next
                )
                print(f"验证数据集大小: {len(self.val_dataset)}")
            except FileNotFoundError as e:
                logger.error(f"找不到验证数据目录: {e}")
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
            persistent_workers=True,  # 保持工作进程存活
            prefetch_factor=2,        # 预取因子
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
            persistent_workers=True,  # 保持工作进程存活
            prefetch_factor=2,        # 预取因子
            drop_last=False
        )


# 使用示例
if __name__ == "__main__":
    # 创建数据模块
    data_module = OptimizedSequenceDataModule(
        dataset_path="dataset",
        train_folder="task_D_D/training/sequence_data",
        val_folder="task_D_D/validation/sequence_data",
        batch_size=128,
        num_workers=8,
        shard_cache_size=3,
        preload_next=True
    )
    
    # 设置数据集
    data_module.setup()
    
    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # 测试数据加载性能
    print("测试训练数据加载性能...")
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 测试10个批次
            break
        print(f"批次 {i+1}: {batch['input_actions'].shape}")
    end_time = time.time()
    print(f"加载10个批次用时: {end_time - start_time:.2f}秒")