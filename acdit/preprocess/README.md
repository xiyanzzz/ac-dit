# CALVIN数据集分离格式预处理

该文档详细说明了CALVIN数据集分离格式预处理的数据结构和使用方法。(old ver.)

## 背景

CALVIN数据集非常大（如ABCD_D任务为656GB），在训练过程中将所有数据加载到内存会导致内存不足。为解决这个问题，我们开发了分离格式预处理方法，将动作数据和图像数据分开存储，只预加载小体积的动作数据到内存，而大体积的图像数据按需加载。

## 数据结构

分离格式的数据组织结构如下：

```
separate_data/
├── actions_data.npz      # 包含所有动作数据
├── metadata.npz          # 包含元数据和语言标注
├── sequence_indices.npz  # 包含预计算的序列索引
└── images/               # 包含所有图像数据
    ├── frame_000000.npz  # 包含单帧的两种图像
    ├── frame_000001.npz
    └── ...
```

### 文件内容详解

#### 1. actions_data.npz

包含所有动作数据，预加载到内存中：

- `abs_actions`: 形状为[N, 7]的数组，包含所有绝对动作 (x,y,z, euler_x, euler_y, euler_z, gripper)
- `rel_actions`: 形状为[N, 7]的数组，包含所有相对动作

#### 2. metadata.npz

包含元数据和语言标注：

- `language_annotations`: 所有语言标注文本
- `lang_to_frame_indices`: 语言标注和对应帧区间的映射，格式为(语言索引, 起始帧, 结束帧)
- `observation_horizon`: 观测窗口大小(输入序列长度)
- `prediction_horizon`: 预测窗口大小(目标序列长度)
- `seq_stride`: 序列采样步长
- `creation_time`: 数据创建时间戳
- `input_action_dim`: 输入动作维度(7)
- `target_action_dim`: 目标动作维度(7)
- `images_dir`: 图像数据目录路径
- `total_samples`: 总样本数

#### 3. sequence_indices.npz

预计算的序列索引信息：

- `sequence_indices`: 预计算的所有可能序列索引，格式为(样本索引, 语言索引, 起始帧索引)

#### 4. images目录

图像数据目录，按需加载：

- 每个帧存储在一个.npz文件中，命名为`frame_{索引:06d}.npz`
- 每个.npz文件包含两种图像：
  - `rgb_static`: 静态相机图像，形状为[H, W, 3]
  - `rgb_gripper`: 机械手相机图像，形状为[H, W, 3]

## 数据提取方法

使用`extract_separate_data.py`脚本从原始CALVIN数据集提取分离格式数据：

```bash
# 示例：处理debug数据集
python AC_DiT/preprocess/extract_separate_data.py --task calvin_debug_dataset

# 示例：处理D任务训练集
python AC_DiT/preprocess/extract_separate_data.py --task task_D_D --split training

# 示例：处理所有任务的所有分割
python AC_DiT/preprocess/extract_separate_data.py --task all --split all
```

### 参数说明

- `--root_dir`: 数据集根目录，默认为'dataset'
- `--task`: 要处理的任务，可选['calvin_debug_dataset', 'task_D_D', 'task_ABC_D', 'task_ABCD_D', 'all']
- `--split`: 数据分割，可选['training', 'validation', 'all']
- `--observation_horizon`: 输入观测的时间步数，默认为2
- `--prediction_horizon`: 预测动作的时间步数，默认为10
- `--seq_stride`: 序列采样步长，默认为1
- `--out_dir`: 输出目录，默认为{root_dir}/{task}/{split}/separate_data/
- `--force`: 强制覆盖已存在的文件

## 使用方法

使用`lazy_sequence_dataset.py`中的`LazySequenceDataModule`加载分离格式数据：

```python
from AC_DiT.datasets.lazy_sequence_dataset import LazySequenceDataModule

# 创建数据模块
data_module = LazySequenceDataModule(
    dataset_path="dataset/task_D_D",
    train_folder="training/separate_data",
    val_folder="validation/separate_data",
    batch_size=64,
    num_workers=8,
    use_future_image=True,
    future_image_mean=5.0,
    future_image_std=1.7,
    cache_size=1000  # 图像缓存大小
)

# 设置数据集
data_module.setup()

# 获取数据加载器
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

### 参数说明

- `dataset_path`: 数据集根路径
- `train_folder`: 训练数据文件夹，默认为"training/separate_data"
- `val_folder`: 验证数据文件夹，默认为"validation/separate_data"
- `batch_size`: 批次大小，默认为64
- `num_workers`: 数据加载工作进程数，默认为8
- `use_future_image`: 是否使用未来图像，默认为True
- `future_image_mean`: 未来图像间隔时间步的正态分布均值，默认为5.0
- `future_image_std`: 未来图像间隔时间步的正态分布标准差，默认为1.7
- `cache_size`: 图像缓存大小，默认为1000

## 优势

分离格式数据相比于原始紧凑格式数据具有以下优势：

1. **内存高效**：只将较小的动作数据(~几MB)加载到内存中，而较大的图像数据(~几GB)按需加载
2. **支持超大数据集**：即使是ABCD_D任务(656GB)也可以在有限内存的环境中训练
3. **随机访问高效**：可以直接索引任意帧的图像，无需加载整个数据集
4. **缓存友好**：使用LRU缓存保留最近访问的图像，进一步提高训练效率
5. **接口兼容**：数据加载器返回的格式与原始数据集完全相同，不需要修改训练代码

## 注意事项

1. 确保有足够的磁盘空间，分离格式数据与原始数据大小相近
2. 适当调整`cache_size`参数可优化内存使用和加载性能
3. 使用SSD而非HDD存储数据会大幅提高加载速度
4. 适当增加`num_workers`可提高数据加载并行度 