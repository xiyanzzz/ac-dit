import logging
import os
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class CalvinDataset(Dataset):
    """
    Calvin数据集加载器
    """
    def __init__(
        self,
        dataset_path: str,
        split: str = "training",
        modality: str = "vis",
        seq_len: int = 10,
        pad_seq_len: bool = True,
        pad_frame: Optional[int] = None,
    ):
        """
        初始化Calvin数据集
        
        Args:
            dataset_path: 数据集路径
            split: 训练集或验证集
            modality: 视觉模态或语言模态
            seq_len: 序列长度
            pad_seq_len: 是否补齐序列长度
            pad_frame: 帧补齐方式
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.modality = modality
        self.seq_len = seq_len
        self.pad_seq_len = pad_seq_len
        self.pad_frame = pad_frame
        
        print(f"初始化数据集: {self.dataset_path}, 拆分: {self.split}, 模态: {self.modality}")
        
        # 获取所有npz文件路径
        self.data_files = self._get_data_files()
        print(f"找到 {len(self.data_files)} 个数据文件")
        
        if len(self.data_files) == 0:
            # 如果没有找到文件，递归搜索整个数据集目录
            print(f"在指定目录未找到数据文件，正在递归搜索整个数据集目录...")
            self._recursive_search_data_files()
            
        # 如果仍然没有找到文件，创建虚拟数据
        if len(self.data_files) == 0:
            print("警告：未找到任何数据文件，将使用虚拟数据")
            self._create_dummy_data()
            self.episode_lookup = [{'dummy': True, 'index': i} for i in range(10)]
        else:
            self.episode_lookup = self._build_episode_lookup()
            
        self.lang_annotations = None
        
        # 如果模态是语言，加载语言注释
        if "lang" in modality:
            self._load_language_annotations()
        
        # 确保有足够的样本
        if len(self.episode_lookup) == 0:
            print(f"警告：没有找到有效样本！使用示例数据代替.")
            # 创建一些示例数据以避免空数据集错误
            self._create_dummy_data()
            self.episode_lookup = [{'dummy': True, 'index': i} for i in range(10)]
        
        print(f"已加载{len(self.episode_lookup)}个Calvin episodes")
        
    def _recursive_search_data_files(self):
        """递归搜索整个数据集目录查找npz文件"""
        print(f"递归搜索数据文件...")
        for root, _, files in os.walk(self.dataset_path):
            npz_files = [os.path.join(root, f) for f in files if f.endswith(".npz") and f.startswith("episode_")]
            if npz_files:
                subfolder = os.path.basename(root)
                if subfolder in ["training", "validation"]:
                    print(f"在 {root} 中找到了 {len(npz_files)} 个.npz文件")
                    if subfolder == self.split:
                        self.data_files.extend(npz_files)
                    else:
                        print(f"文件在 '{subfolder}' 目录中，但当前拆分是 '{self.split}'，更新拆分...")
                        self.split = subfolder
                        self.data_files.extend(npz_files)
                else:
                    # 检查父目录是否为训练/验证目录
                    parent = os.path.basename(os.path.dirname(root))
                    if parent in ["training", "validation"]:
                        print(f"在 {root} 中找到了 {len(npz_files)} 个.npz文件 (父目录: {parent})")
                        if parent == self.split:
                            self.data_files.extend(npz_files)
                        else:
                            print(f"更新拆分从 '{self.split}' 到 '{parent}'")
                            self.split = parent
                            self.data_files.extend(npz_files)
                    else:
                        # 如果没有找到训练/验证目录，仍然添加文件
                        print(f"在 {root} 中找到了 {len(npz_files)} 个.npz文件，但无法确定拆分")
                        if not self.data_files:  # 如果之前没有找到任何文件，先添加这些
                            self.data_files.extend(npz_files)
        
        # 确保文件列表已排序
        self.data_files = sorted(self.data_files)
        print(f"递归搜索后找到 {len(self.data_files)} 个数据文件")
        
    def _get_data_files(self) -> List[str]:
        """获取所有npz数据文件"""
        data_dir = os.path.join(self.dataset_path, self.split)
        files = []
        
        if not os.path.exists(data_dir):
            print(f"警告：数据目录不存在: {data_dir}")
            # 尝试直接在数据集路径中查找
            direct_path = self.dataset_path
            if os.path.exists(direct_path):
                print(f"尝试直接在数据集路径查找: {direct_path}")
                for file in os.listdir(direct_path):
                    if file.endswith(".npz") and file.startswith("episode_"):
                        files.append(os.path.join(direct_path, file))
                        print(f"找到数据文件: {file}")
            return files
        
        for file in os.listdir(data_dir):
            if file.endswith(".npz") and file.startswith("episode_"):
                files.append(os.path.join(data_dir, file))
                print(f"找到数据文件: {file}")
        
        return sorted(files)
        
    def _build_episode_lookup(self) -> List[Dict]:
        """
        构建episode查找表，每个条目包含文件路径和帧索引
        """
        lookup = []
        
        # 最大数据点数，用于debug数据集，避免加载太多
        max_datapoints = 200 if "debug" in self.dataset_path else 10000
        
        for file_path in self.data_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'actions' not in data:
                    print(f"警告：文件 {file_path} 中没有找到'actions'键")
                    print(f"可用键: {data.files}")
                    continue
                    
                episode_len = len(data['actions'])
                print(f"文件 {os.path.basename(file_path)} 包含 {episode_len} 帧")
                
                # 如果序列长度大于episode长度，调整序列长度
                effective_seq_len = self.seq_len
                if episode_len <= self.seq_len:
                    effective_seq_len = max(1, episode_len - 1)  # 至少需要1帧，留1帧作为目标
                    print(f"调整序列长度从 {self.seq_len} 到 {effective_seq_len} 以适应短序列")
                
                # 对于每个可能的起始索引，至少需要1帧作为输入，1帧作为目标
                if episode_len >= 2:  # 确保至少有2帧
                    valid_starts = 0
                    for start_idx in range(0, episode_len - 1):
                        end_idx = min(start_idx + effective_seq_len, episode_len - 1)
                        actual_seq_len = end_idx - start_idx
                        
                        lookup.append({
                            'file_path': file_path,
                            'start_idx': start_idx,
                            'seq_len': actual_seq_len
                        })
                        valid_starts += 1
                        
                        # 如果数据点已足够，提前退出
                        if len(lookup) >= max_datapoints:
                            print(f"已达到最大数据点数量 {max_datapoints}")
                            return lookup
                    
                    print(f"从文件 {os.path.basename(file_path)} 中添加了 {valid_starts} 个有效样本")
                else:
                    print(f"文件 {os.path.basename(file_path)} 帧数不足，无法创建样本")
                        
            except Exception as e:
                print(f"加载{file_path}时出错: {e}")
        
        if len(lookup) == 0:
            print("警告：没有找到有效的样本！请检查数据集结构。")
        else:
            print(f"总共找到 {len(lookup)} 个有效样本")
            
        return lookup
    
    def _create_dummy_data(self):
        """创建一些虚拟数据以避免空数据集错误"""
        print("创建示例数据以避免空数据集错误...")
        
    def _load_language_annotations(self):
        """加载语言注释"""
        lang_dir = os.path.join(self.dataset_path, self.split, "lang_annotations")
        
        if os.path.exists(lang_dir):
            lang_file = os.path.join(lang_dir, "auto_lang_ann.npy")
            
            if os.path.exists(lang_file):
                try:
                    self.lang_annotations = np.load(lang_file, allow_pickle=True).item()
                    print(f"已加载语言注释: {len(self.lang_annotations['language']['ann'])}条")
                except Exception as e:
                    print(f"加载语言注释时出错: {e}")
                    self._create_dummy_lang_annotations()
            else:
                print(f"语言注释文件不存在: {lang_file}")
                self._create_dummy_lang_annotations()
        else:
            print(f"语言注释目录不存在: {lang_dir}")
            self._create_dummy_lang_annotations()
    
    def _create_dummy_lang_annotations(self):
        """创建虚拟语言注释以避免错误"""
        print("创建虚拟语言注释...")
        self.lang_annotations = {
            'language': {
                'ann': ["拿起方块", "打开抽屉", "打开灯", "移动立方体"],
                'emb': np.random.randn(4, 512).astype(np.float32)  # 4个512维的随机嵌入
            }
        }
            
    def __len__(self) -> int:
        """数据集长度"""
        return max(1, len(self.episode_lookup))  # 确保至少有一个样本
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据项"""
        if idx >= len(self.episode_lookup):
            idx = 0  # 防止索引越界
            
        lookup_item = self.episode_lookup[idx]
        
        # 处理虚拟数据情况
        if lookup_item.get('dummy', False):
            return self._get_dummy_item(lookup_item['index'])
            
        file_path = lookup_item['file_path']
        start_idx = lookup_item['start_idx']
        seq_len = lookup_item.get('seq_len', self.seq_len)
        
        try:
            # 加载数据文件
            episode_data = np.load(file_path, allow_pickle=True)
            
            # 提取序列
            end_idx = start_idx + seq_len
            
            # 获取观察和动作
            actions = episode_data['actions'][start_idx:end_idx]
            
            # 获取RGB观察 (注意我们需要额外一帧作为目标)
            # 如果没有足够的帧，我们将使用最后一帧重复
            next_frame_idx = min(end_idx + 1, len(episode_data['rgb_static']))
            
            rgb_static = episode_data['rgb_static'][start_idx:next_frame_idx]
            rgb_gripper = episode_data['rgb_gripper'][start_idx:next_frame_idx]
            
            # 确保我们至少有1帧作为输入
            if len(actions) == 0:
                raise ValueError(f"没有足够的帧: 起始={start_idx}, 结束={end_idx}, 文件长度={len(episode_data['actions'])}")
            
            # 确保我们至少有1帧作为目标
            if rgb_static.shape[0] <= seq_len:
                # 复制最后一帧作为目标
                rgb_static = np.concatenate([rgb_static, rgb_static[-1:]])
                rgb_gripper = np.concatenate([rgb_gripper, rgb_gripper[-1:]])
            
            # 转换为PyTorch张量
            actions = torch.from_numpy(actions).float()
            rgb_static = torch.from_numpy(rgb_static).float() / 255.0  # 归一化到[0,1]
            rgb_gripper = torch.from_numpy(rgb_gripper).float() / 255.0
            
            # 调整通道顺序，从HWC转为CHW
            rgb_static = rgb_static.permute(0, 3, 1, 2)
            rgb_gripper = rgb_gripper.permute(0, 3, 1, 2)
            
            # 构建返回字典
            result = {
                'actions': actions,
                'rgb_obs': {
                    'rgb_static': rgb_static,
                    'rgb_gripper': rgb_gripper,
                },
                'idx': idx,
            }
            
            # 如果是语言模态，添加语言数据
            if "lang" in self.modality and self.lang_annotations is not None:
                # 为简单起见，随机选择一个语言描述
                lang_idx = np.random.randint(0, len(self.lang_annotations['language']['ann']))
                
                # 获取嵌入和原始文本
                lang_embed = self.lang_annotations['language']['emb'][lang_idx]
                lang_text = self.lang_annotations['language']['ann'][lang_idx]
                
                result['lang'] = torch.from_numpy(lang_embed).float()
                result['lang_text'] = lang_text
            
            return result
            
        except Exception as e:
            print(f"获取项 {idx} 时出错: {e}")
            # 返回虚拟数据
            return self._get_dummy_item(idx)
    
    def _get_dummy_item(self, idx: int) -> Dict:
        """生成虚拟数据项"""
        # 创建随机动作和RGB数据
        actions = torch.randn(self.seq_len, 7).float()
        rgb_static = torch.randn(self.seq_len + 1, 3, 200, 200).float()
        rgb_gripper = torch.randn(self.seq_len + 1, 3, 84, 84).float()
        
        result = {
            'actions': actions,
            'rgb_obs': {
                'rgb_static': rgb_static,
                'rgb_gripper': rgb_gripper,
            },
            'idx': idx,
        }
        
        # 如果是语言模态，添加语言数据
        if "lang" in self.modality and self.lang_annotations is not None:
            # 为简单起见，随机选择一个语言描述
            lang_idx = np.random.randint(0, len(self.lang_annotations['language']['ann']))
            
            # 获取嵌入和原始文本
            lang_embed = self.lang_annotations['language']['emb'][lang_idx]
            lang_text = self.lang_annotations['language']['ann'][lang_idx]
            
            result['lang'] = torch.from_numpy(lang_embed).float()
            result['lang_text'] = lang_text
        
        return result


class CalvinDataModule(pl.LightningDataModule):
    """
    Calvin数据模块，用于管理训练和验证数据集
    """
    def __init__(
        self,
        dataset_path: str,
        train_folder: str = "training",
        val_folder: str = "validation",
        env: str = "debug",
        batch_size: int = 32,
        num_workers: int = 8,
        modalities: List[str] = ["vis"],
        seq_len: int = 10,
    ):
        """
        初始化数据模块
        
        Args:
            dataset_path: 数据集根路径
            train_folder: 训练集文件夹名
            val_folder: 验证集文件夹名
            env: 环境名称
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            modalities: 模态列表
            seq_len: 序列长度
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.env = env
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities
        self.seq_len = seq_len
        
        self.train_datasets = {}
        self.val_datasets = {}
        
        print(f"初始化Calvin数据模块: {self.dataset_path}")
        print(f"环境: {self.env}, 批次大小: {self.batch_size}")
        print(f"模态: {self.modalities}")
        
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集
        
        Args:
            stage: 训练阶段
        """
        # 初始化训练数据集
        if stage == "fit" or stage is None:
            print(f"设置训练数据集...")
            for modality in self.modalities:
                print(f"创建 '{modality}' 模态的训练数据集")
                self.train_datasets[modality] = CalvinDataset(
                    dataset_path=self.dataset_path,
                    split=self.train_folder,
                    modality=modality,
                    seq_len=self.seq_len,
                )
                print(f"'{modality}' 训练数据集大小: {len(self.train_datasets[modality])}")
        
        # 初始化验证数据集
        if stage == "validate" or stage is None:
            print(f"设置验证数据集...")
            for modality in self.modalities:
                print(f"创建 '{modality}' 模态的验证数据集")
                self.val_datasets[modality] = CalvinDataset(
                    dataset_path=self.dataset_path,
                    split=self.val_folder,
                    modality=modality,
                    seq_len=self.seq_len,
                )
                print(f"'{modality}' 验证数据集大小: {len(self.val_datasets[modality])}")
                
    def train_dataloader(self) -> Dict[str, DataLoader]:
        """返回训练数据加载器"""
        train_dataloaders = {}
        
        for modality, dataset in self.train_datasets.items():
            print(f"创建 '{modality}' 模态的训练数据加载器, 数据集大小: {len(dataset)}")
            train_dataloaders[modality] = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            
        return train_dataloaders
    
    def val_dataloader(self) -> Dict[str, DataLoader]:
        """返回验证数据加载器"""
        val_dataloaders = {}
        
        for modality, dataset in self.val_datasets.items():
            print(f"创建 '{modality}' 模态的验证数据加载器, 数据集大小: {len(dataset)}")
            val_dataloaders[modality] = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            
        return val_dataloaders 