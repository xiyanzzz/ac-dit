import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def extract_compact_data(
        root_dir,
        task='calvin_debug_dataset',
        split='training',
        observation_horizon=2,     # 过去的观测帧数
        prediction_horizon=10,     # 未来的预测帧数
        seq_stride=1,              # 序列采样步长
        out_dir=None,
        force_overwrite=False
):
    """
    提取紧凑格式的数据，只保存语言标注区间内的原始观测和动作数据，
    并预先计算好所有可能的序列索引
    """
    
    # 1. 获取数据目录路径
    data_dir = os.path.join(root_dir, task, split)
    if not os.path.exists(data_dir):
        print(f"[警告] {data_dir} 不存在，跳过!")
        return
        
    # 确保输出目录存在
    if out_dir is None:
        out_dir = os.path.join(root_dir, task, split, 'compact_data')
    os.makedirs(out_dir, exist_ok=True)
    
    # 检查是否已存在数据文件
    data_file = os.path.join(out_dir, "compact_data.npz")
    metadata_file = os.path.join(out_dir, "metadata.npz")
    seq_index_file = os.path.join(out_dir, "sequence_indices.npz")
    
    existing_files = [f for f in [data_file, metadata_file, seq_index_file] if os.path.exists(f)]
    if existing_files:
        if force_overwrite:
            for file in existing_files:
                print(f"删除已存在的文件: {file}")
                os.remove(file)
        else:
            print(f"[警告] 已存在数据文件。如果要重新处理，请使用 -f 参数覆盖。")
            return
    
    # 2. 获取所有episode文件并排序
    ep_files = []
    for f in os.listdir(data_dir):
        if f.startswith("episode_") and f.endswith(".npz"):
            ep_num = int(f.split("_")[1].split(".")[0])
            ep_files.append((ep_num, os.path.join(data_dir, f)))
    
    ep_files.sort()  # 按episode编号排序
    ep_indices = [idx for idx, _ in ep_files]
    print(f"找到 {len(ep_files)} 个episode文件，索引范围: [{min(ep_indices) if ep_indices else 'N/A'}, {max(ep_indices) if ep_indices else 'N/A'}]")
    
    if len(ep_files) == 0:
        print(f"[错误] 在 {data_dir} 中没有找到episode文件!")
        return
    
    # 3. 加载语言标注
    lang_ann_file = os.path.join(data_dir, "lang_annotations", "auto_lang_ann.npy")
    if not os.path.exists(lang_ann_file):
        print(f"[错误] 找不到语言标注文件: {lang_ann_file}")
        return
    
    print(f"加载语言标注: {lang_ann_file}")
    language_data = np.load(lang_ann_file, allow_pickle=True).item()
    
    if 'language' not in language_data or 'info' not in language_data:
        print("[错误] 语言标注文件格式不正确")
        return
    
    language_annotations = language_data['language']['ann']
    lang_start_end_indices = language_data['info']['indx']
    
    print(f"找到 {len(language_annotations)} 条语言标注")
    for i, (start, end) in enumerate(lang_start_end_indices):
        print(f"语言标注 #{i}: 区间 [{start}, {end}], 文本: {language_annotations[i]}")
    
    # 4. 确定需要提取的episode范围
    all_episodes = set()
    for start_ep, end_ep in lang_start_end_indices:
        all_episodes.update(range(start_ep, end_ep + 1))
    
    needed_episodes = sorted(list(all_episodes))
    print(f"需要提取 {len(needed_episodes)} 个episode")
    
    # 构建episode索引到文件路径的映射
    ep_idx_to_file = {idx: path for idx, path in ep_files}
    
    # 5. 提取所有必要的数据
    all_rgb_static = []
    all_rgb_gripper = []
    all_abs_actions = []
    all_rel_actions = []
    episode_indices = []  # 记录每个帧对应的episode索引
    
    print("正在提取所有需要的数据...")
    for ep_idx in tqdm(needed_episodes):
        if ep_idx not in ep_idx_to_file:
            print(f"[警告] 找不到episode {ep_idx}，跳过")
            continue
            
        file_path = ep_idx_to_file[ep_idx]
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # 提取RGB图像
            all_rgb_static.append(data['rgb_static'])
            all_rgb_gripper.append(data['rgb_gripper'])
            
            # 提取绝对动作 (从robot_obs提取)
            robot_obs = data['robot_obs']
            # 提取前6个元素(位置和姿态)和最后一个元素(夹爪状态)
            abs_action = np.concatenate([robot_obs[:6], robot_obs[-1:]])
            all_abs_actions.append(abs_action)
            
            # 提取相对动作
            all_rel_actions.append(data['rel_actions'])
            
            # 记录episode索引
            episode_indices.append(ep_idx)
            
        except Exception as e:
            print(f"处理episode {ep_idx} 时出错: {e}")
    
    # 6. 转换为numpy数组
    all_rgb_static = np.stack(all_rgb_static)
    all_rgb_gripper = np.stack(all_rgb_gripper)
    all_abs_actions = np.stack(all_abs_actions)
    all_rel_actions = np.stack(all_rel_actions)
    episode_indices = np.array(episode_indices)
    
    print(f"数据形状:")
    print(f"  RGB静态图像: {all_rgb_static.shape}")
    print(f"  RGB机械手图像: {all_rgb_gripper.shape}")
    print(f"  绝对动作: {all_abs_actions.shape}")
    print(f"  相对动作: {all_rel_actions.shape}")
    print(f"  Episode索引: {episode_indices.shape}")
    
    # 7. 创建语言区间到帧索引的映射
    lang_to_frame_indices = []
    frame_to_ep_mapping = {ep_idx: i for i, ep_idx in enumerate(episode_indices)}
    
    for lang_idx, (start_ep, end_ep) in enumerate(lang_start_end_indices):
        start_frame = frame_to_ep_mapping.get(start_ep)
        end_frame = frame_to_ep_mapping.get(end_ep)
        
        if start_frame is not None and end_frame is not None:
            lang_to_frame_indices.append((lang_idx, start_frame, end_frame))
        else:
            print(f"[警告] 语言标注 #{lang_idx} 的区间 [{start_ep}, {end_ep}] 部分或全部不在提取的数据中")
    
    print(f"创建了 {len(lang_to_frame_indices)} 个语言区间到帧索引的映射")
    
    # 8. 预先计算所有可能的序列索引
    seq_length = observation_horizon + prediction_horizon
    sequence_indices = []  # 格式: [(样本索引, 语言索引, 起始帧索引), ...]
    
    print(f"计算可能的序列索引 (观测窗口={observation_horizon}, 预测窗口={prediction_horizon}, 步长={seq_stride})...")
    sample_idx = 0
    
    for lang_idx, start_frame, end_frame in lang_to_frame_indices:
        # 计算该语言区间内有多少个可能的序列
        frames_in_interval = end_frame - start_frame + 1
        
        if frames_in_interval < seq_length:
            print(f"[警告] 语言标注 #{lang_idx} 的区间长度 {frames_in_interval} 小于序列长度 {seq_length}，跳过")
            continue
        
        # 计算可以提取多少个序列
        num_sequences = (frames_in_interval - seq_length) // seq_stride + 1
        
        for seq_idx in range(num_sequences):
            frame_start_idx = start_frame + seq_idx * seq_stride
            sequence_indices.append((sample_idx, lang_idx, frame_start_idx))
            sample_idx += 1
    
    sequence_indices = np.array(sequence_indices)
    print(f"共生成 {len(sequence_indices)} 个序列索引")
    
    # 9. 保存数据和元数据
    print(f"保存数据到 {data_file}")
    np.savez_compressed(
        data_file,
        rgb_static=all_rgb_static,
        rgb_gripper=all_rgb_gripper,
        abs_actions=all_abs_actions,
        rel_actions=all_rel_actions,
        episode_indices=episode_indices
    )
    
    print(f"保存元数据到 {metadata_file}")
    np.savez_compressed(
        metadata_file,
        language_annotations=np.array(language_annotations, dtype=object),
        lang_to_frame_indices=np.array(lang_to_frame_indices),
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        seq_stride=seq_stride,
        creation_time=time.time(),
        input_action_dim=7,  # 输入动作维度 (6+1)
        target_action_dim=7,  # 目标动作维度 (7)
    )
    
    print(f"保存序列索引到 {seq_index_file}")
    np.savez_compressed(
        seq_index_file,
        sequence_indices=sequence_indices,
        total_samples=len(sequence_indices)
    )
    
    print("数据提取完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='提取紧凑格式的序列数据')
    parser.add_argument('-i', '--root_dir', type=str, default='dataset',
                        help='数据集根目录，包含任务子目录')
    parser.add_argument('--task', type=str, default='task_ABCD_D',
                        choices=['calvin_debug_dataset', 'task_D_D', 'task_ABC_D', 'task_ABCD_D', 'all'],
                        help='要处理的任务')
    parser.add_argument('--split', type=str, default='all',
                        choices=['training', 'validation', 'all'],
                        help='数据分割')
    parser.add_argument('--observation_horizon', type=int, default=2,
                        help='输入观测的时间步数')
    parser.add_argument('--prediction_horizon', type=int, default=10,
                        help='预测动作的时间步数')
    parser.add_argument('--seq_stride', type=int, default=1,
                        help='序列采样步长')
    parser.add_argument('-o', '--out_dir', type=str, default=None,
                        help='输出目录，默认为{root_dir}/{task}/{split}/compact_data/')
    parser.add_argument('-f', '--force', action='store_true',
                        help='强制覆盖已存在的文件')
    
    args = parser.parse_args()
    
    # 处理任务和分割参数
    tasks = ['calvin_debug_dataset', 'task_D_D', 'task_ABC_D', 'task_ABCD_D'] if args.task == 'all' else [args.task]
    splits = ['training', 'validation'] if args.split == 'all' else [args.split]
    
    # 为每个任务和分割提取数据
    for task in tasks:
        for split in splits:
            print(f"处理 {task}/{split}")
            if args.out_dir is None:
                out_dir = os.path.join(args.root_dir, task, split, 'compact_data')
            else:
                out_dir = os.path.join(args.out_dir, task, split, 'compact_data')
                
            extract_compact_data(
                args.root_dir,
                task=task,
                split=split,
                observation_horizon=args.observation_horizon,
                prediction_horizon=args.prediction_horizon,
                seq_stride=args.seq_stride,
                out_dir=out_dir,
                force_overwrite=args.force,
            )