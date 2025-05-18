import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from copy import deepcopy

# 添加项目根目录到系统路径
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from AC_DiT.models.acdit_runner import ACDiTRunner, create_acdit_runner_from_config
from AC_DiT.train_ac_dit_v2 import ACDiTLightningModule

from AC_DiT.models.acdit_future_pred_runner import ACDiTFuturePredRunner, create_acdit_future_pred_runner_from_config
from AC_DiT.train_ac_dit_v3 import ACDiTFuturePredLightningModule


# 使用calvin_agent的评估工具
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)

import hydra
import numpy as np
import pytorch_lightning
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
from torchvision import transforms

logger = logging.getLogger(__name__)

# 评估参数
EP_LEN = 360
NUM_SEQUENCES = 10 # 已由输入参数控制

PATH_TO_CALVIN_ENV = "/home/cyan/Projects/mdt_policy/calvin_env/calvin_env/__init__.py"

def make_env(dataset_path, debug=False):
    """
    创建Calvin环境
    
    Args:
        dataset_path: 数据集路径
        debug: 是否开启调试模式
        
    Returns:
        env: Calvin环境
    """
    # 猴子补丁修复calvin_env.__file__为None的问题
    import calvin_env
    if not hasattr(calvin_env, '__file__') or calvin_env.__file__ is None:
        calvin_env.__file__ = PATH_TO_CALVIN_ENV  # 提供一个有效路径

    from calvin_env.calvin_env.envs.play_table_env import get_env
    
    val_folder = Path(dataset_path) / "validation"
    obs_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"]
    }
    env = get_env(val_folder, show_gui=False, obs_space=obs_space)
    
    return env


def load_checkpoint(checkpoint_path: str, config_path: str, sampler_type: str, num_sampling_steps: int, future_pred: bool):
    """
    从LightningModule封装中加载runner模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        config_path: 训练时保存的配置文件路径
    """
    # 加载训练时保存的配置
    config = OmegaConf.load(config_path)

    # 根据输入参数修改采样器类型与采样步数
    config.model.sampler_type = sampler_type
    config.model.num_sampling_steps = num_sampling_steps

    print("加载配置:")
    # print(OmegaConf.to_yaml(config))

    print("-"*100)
    if future_pred:
        model = ACDiTFuturePredLightningModule.load_from_checkpoint(
            checkpoint_path,
            runner=create_acdit_future_pred_runner_from_config(config)
        )
        print("已加载带图像预测的AC_DiT智能体")
    else:
        model = ACDiTLightningModule.load_from_checkpoint(
            checkpoint_path,
            runner=create_acdit_runner_from_config(config)
        )
        print("已加载普通AC_DiT智能体")
    print("-"*100)
    
    return model.runner, config


class ACDiTAgent:
    """
    ACDiT模型适配器，用于评估ACDiT模型
    """
    def __init__(self, cfg, device_id=0):
        """
        初始化ACDiT模型
        
        Args:
            cfg: 配置对象
            device_id: 设备ID
        """
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"从 {cfg.ckpt_path} 加载模型")

        # 每次预测执行的步数
        self.steps_per_pred = getattr(cfg, 'steps_per_pred', 1)
        logger.info(f"每次预测执行步数: {self.steps_per_pred}")

        # 设置采样参数
        # if hasattr(cfg, 'sampler_type'):
        #     self.runner.sampler_type = cfg.sampler_type # mark
        # if hasattr(cfg, 'num_sampling_steps'):
        #     self.runner.num_sampling_steps = cfg.num_sampling_steps # mark
        
        # 加载检查点和配置
        self.runner, self.config = load_checkpoint(cfg.ckpt_path, cfg.config_path, 
                                                   cfg.sampler_type, cfg.num_sampling_steps,
                                                   cfg.future_pred)
        self.runner.eval()
        
        # 初始化状态
        self.pred_horizon = getattr(self.runner, 'pred_horizon', 10)
        self.obs_horizon = getattr(self.runner, 'obs_horizon', 2)
        
        # 确保steps_per_pred不超过pred_horizon
        self.steps_per_pred = min(self.steps_per_pred, self.pred_horizon)
        
        # 是否使用相对观测
        self.rel_obs = getattr(cfg, 'rel_obs')
        
        # 历史观测存储
        self.rgb_static_history = []
        self.rgb_gripper_history = []
        self.action_history = []
        
        # 存储上一时刻的绝对动作，用于计算相对动作
        self.last_abs_action = None
        
        # 设置图像预处理变换
        self.transform = self._get_transforms()
            
        logger.info(f"采样器: {self.runner.sampler_type}, 采样步数: {self.runner.num_sampling_steps}")
        logger.info(f"使用相对观测: {self.rel_obs}")
    
    def _get_transforms(self):
        """获取配置中的图像预处理变换"""
        # if hasattr(self.config, 'data') and hasattr(self.config.data, 'transforms') and hasattr(self.config.data.transforms, 'val'):
        #     # 从配置中获取验证集变换
        #     val_transforms = hydra.utils.instantiate(self.config.data.transforms.val)
        #     return val_transforms
        # else:
        #     # 使用默认变换
        #     print("找不到变换配置，将使用默认变换")
        #     return transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        #     ])
        return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
    
    def reset(self):
        """每个序列开始时重置状态"""
        self.rgb_static_history = []
        self.rgb_gripper_history = []
        self.action_history = []
        self.last_abs_action = None
    
    def absolute_to_relative_action(self, current_abs_action):
        """
        将绝对动作转换为相对动作
        
        Args:
            current_abs_action: 当前时刻的绝对动作 [x,y,z,euler_x,euler_y,euler_z,gripper]
        
        Returns:
            相对动作 [rel_x,rel_y,rel_z,rel_euler_x,rel_euler_y,rel_euler_z,gripper]
        """
        # 如果没有上一时刻的动作，返回零向量（位置和姿态）和当前夹爪状态
        if self.last_abs_action is None:
            rel_action = np.zeros(7)
            rel_action[6] = current_abs_action[6]  # 保持夹爪状态
            return rel_action
        
        # 位置部分：(当前 - 上一时刻) * 50，并裁剪到[-1, 1]
        rel_position = (current_abs_action[:3] - self.last_abs_action[:3]) * 50.0
        rel_position = np.clip(rel_position, -1.0, 1.0)
        
        # 姿态部分：(当前 - 上一时刻) * 20，并裁剪到[-1, 1]
        rel_orientation = (current_abs_action[3:6] - self.last_abs_action[3:6]) * 20.0
        rel_orientation = np.clip(rel_orientation, -1.0, 1.0)
        
        # 夹爪部分：保持当前值
        gripper_action = current_abs_action[6]
        
        # 组合为相对动作
        rel_action = np.concatenate([rel_position, rel_orientation, [gripper_action]])
        
        return rel_action
    
    def update_history(self, obs):
        """
        更新历史观测队列
        
        Args:
            obs: 环境观察
        """
        # 提取RGB观察
        rgb_static = obs['rgb_obs']['rgb_static'].astype(np.float32) / 255.0
        rgb_gripper = obs['rgb_obs']['rgb_gripper'].astype(np.float32) / 255.0
        
        # 转换格式 [H, W, C] -> [C, H, W]
        rgb_static = torch.from_numpy(rgb_static).permute(2, 0, 1)
        rgb_gripper = torch.from_numpy(rgb_gripper).permute(2, 0, 1)
        
        # 应用图像变换
        rgb_static = self.transform(rgb_static)
        rgb_gripper = self.transform(rgb_gripper)
        
        # 添加到历史记录中
        self.rgb_static_history.append(rgb_static)
        self.rgb_gripper_history.append(rgb_gripper)
        
        # 保持历史记录长度为obs_horizon
        if len(self.rgb_static_history) > self.obs_horizon:
            self.rgb_static_history.pop(0)
            self.rgb_gripper_history.pop(0)
        
        # 如果历史记录不足，重复当前观察
        while len(self.rgb_static_history) < self.obs_horizon:
            self.rgb_static_history.append(rgb_static)
            self.rgb_gripper_history.append(rgb_gripper)
        
        # 提取机器人观察
        robot_obs = obs['robot_obs']
        
        # 提取绝对动作（位置、方向和夹爪）
        current_abs_action = np.concatenate([
            robot_obs[:6],    # 位置和方向 [x,y,z,euler_x,euler_y,euler_z]
            [robot_obs[-1]]   # 夹爪状态
        ])
        
        # 如果使用相对观测，转换为相对动作
        if self.rel_obs:
            # 使用新的转换方法
            rel_action = self.absolute_to_relative_action(current_abs_action)
            robot_action = rel_action
        else:
            robot_action = current_abs_action
        
        # 更新上一时刻的绝对动作
        self.last_abs_action = current_abs_action
        
        # 添加到动作历史
        self.action_history.append(torch.from_numpy(robot_action).float())
        
        # 保持动作历史长度为obs_horizon
        if len(self.action_history) > self.obs_horizon:
            self.action_history.pop(0)
        
        # 如果动作历史不足，重复当前动作
        while len(self.action_history) < self.obs_horizon:
            self.action_history.append(torch.from_numpy(robot_action).float())
    
    def prepare_inputs(self):
        """
        准备模型输入
        
        Returns:
            rgb_obs: 处理后的RGB观察
            robot_action_tensor: 处理后的机器人动作
        """
        # 构建RGB观察字典
        rgb_obs = {
            'rgb_static': torch.stack(self.rgb_static_history).unsqueeze(0).to(self.device),  # [1, T, C, H, W]
            'rgb_gripper': torch.stack(self.rgb_gripper_history).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        }
        
        # 转换为张量
        robot_action_tensor = torch.stack(self.action_history).unsqueeze(0).to(self.device)  # [1, T, action_dim]
        
        return rgb_obs, robot_action_tensor
    
    def step(self, lang_annotation):
        """
        执行一步预测，返回多步动作
        
        Args:
            lang_annotation: 语言注释
            
        Returns:
            actions: 预测的动作序列，形状为[steps_per_pred, action_dim]
        """
        
        # 准备模型输入
        rgb_obs, past_actions = self.prepare_inputs()
        
        # 预测动作
        pred_actions = self.runner.predict_actions(
            rgb_observations=rgb_obs,
            language_goal=lang_annotation,
            past_actions=past_actions
        )
        
        # 处理预测动作
        # 假设pred_actions已经没有batch_size维度，形状为[pred_horizon, action_dim]
        if len(pred_actions.shape) > 2:  # 如果有batch_size维度，去掉它
            pred_actions = pred_actions[0]  # [pred_horizon, action_dim]
        
        # 获取需要执行的步数的动作
        exec_actions = pred_actions[:self.steps_per_pred].cpu().numpy()
        
        # 量化夹爪动作为1或-1
        for i in range(len(exec_actions)):
            if exec_actions[i, -1] > 0:
                exec_actions[i, -1] = 1.0
            else:
                exec_actions[i, -1] = -1.0
                
        return exec_actions


def evaluate_policy(model, env, cfg, eval_log_dir=None, debug=False):
    """
    评估策略
    
    Args:
        model: ACDiTModel对象
        env: Calvin环境
        cfg: 配置
        eval_log_dir: 评估日志目录
        debug: 是否开启调试模式
        
    Returns:
        results: 评估结果
    """
    # 加载任务配置
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    if eval_log_dir is None:
        eval_log_dir = get_log_dir()
    else:
        # 确保eval_log_dir是Path对象
        eval_log_dir = Path(eval_log_dir)
        eval_log_dir.mkdir(parents=True, exist_ok=True)

    # 获取评估序列
    eval_sequences = get_sequences(cfg.num_sequences if hasattr(cfg, 'num_sequences') else NUM_SEQUENCES)
    results = []

    # 使用tqdm显示进度
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    # 评估每个序列
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug
        )
        results.append(result)
        
        # 更新进度条
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        
    # 打印并保存结果
    try:
        print_and_save(results, eval_sequences, eval_log_dir, 
                      epoch=cfg.epoch if hasattr(cfg, 'epoch') else "")
    except TypeError as e:
        # 处理路径拼接错误
        print(f"保存结果时出错: {e}")
        # 尝试手动保存结果
        results_dict = {
            "success_rates": count_success(results).tolist(),
            "epoch": cfg.epoch if hasattr(cfg, 'epoch') else ""
        }
        results_file = os.path.join(str(eval_log_dir), "results.json")
        with open(results_file, "w") as f:
            json.dump(results_dict, f)
        print(f"结果已保存到: {results_file}")

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug):
    """
    评估单个序列
    
    Args:
        env: Calvin环境
        model: 模型
        task_checker: 任务检查器
        initial_state: 初始状态
        eval_sequence: 评估序列
        val_annotations: 验证注释
        debug: 是否开启调试模式
        
    Returns:
        success_counter: 成功任务数量
    """
    # 根据初始状态重置环境
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

    # # 添加检查代码
    # if scene_obs is None or (isinstance(scene_obs, np.ndarray) and scene_obs.size == 0):
    #     print(f"警告: 场景观察为空，可能是数据集问题，跳过序列")
    #     return 0
    
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    
    # 重置模型状态
    model.reset()
    
    # 调试输出
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"评估序列: {' -> '.join(eval_sequence)}")
        print("子任务: ", end="")
    
    # 执行序列中的每个任务
    success_counter = 0
    for subtask in eval_sequence:
        # 执行单个任务
        success = rollout(env, model, task_checker, subtask, val_annotations, debug)
        
        # 计数成功任务
        if success:
            success_counter += 1
        else:
            return success_counter
            
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug):
    """
    执行单个任务
    
    Args:
        env: Calvin环境
        model: 模型
        task_oracle: 任务检查器
        subtask: 子任务
        val_annotations: 验证注释
        debug: 是否开启调试模式
        
    Returns:
        success: 是否成功完成任务
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    
    # 获取初始观察和信息
    obs = env.get_obs()

    # 记录初始环境状态
    start_info = env.get_info()
    
    # 初始更新历史队列
    model.update_history(obs)

    # 获取语言注释
    lang_annotation = val_annotations[subtask][0]
    # lang_annotation =  'move the light switch to turn on the yellow light'
    #'move the light switch to turn on the yellow light' # test
    # 'pick up the red block on the table' # 'sweep the pink block to the right' 
    print('任务语言标注: ', lang_annotation) # 打印语言注释
    
    # 执行EP_LEN步或直到任务完成
    step_count = 0
    while step_count < EP_LEN:
        # 使用模型预测多步动作
        actions = model.step(lang_annotation)
        
        # 顺序执行预测的多步动作
        for i, action in enumerate(actions):
            # 更新步数计数
            step_count += 1
            
            # 执行动作
            obs, _, _, info = env.step(action)
            
            # 立即更新历史观测队列，确保连续性
            model.update_history(obs)
            
            # 调试模式：显示环境和语言指令
            if debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
            
            # 检查任务是否完成
            current_task_info = task_oracle.get_task_info_for_set(start_info, info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("成功", "green"), end=" ")
                return True
                
            # 如果达到最大步数，退出
            if step_count >= EP_LEN:
                break
    
    # 任务未在限定步数内完成
    if debug:
        print(colored("失败", "red"), end=" ")
    return False


def main():
    """主函数"""
    seed_everything(0, workers=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="在Calvin环境中评估AC_DiT模型")
    
    # 数据集和结果保存路径
    parser.add_argument("--dataset_path", type=str, default = "/home/cyan/Projects/mdt_policy/dataset/calvin_debug_dataset", help="数据集路径")
    parser.add_argument("--eval_log_dir", default="/home/cyan/Projects/mdt_policy/evaluation/logs", type=str, help="评估日志保存位置")
    
    # 模型加载参数
    parser.add_argument("--config_path", type=str, default = "outputs/2025-04-02/17-24-34/.hydra/config.yaml", help="训练配置路径")
    parser.add_argument("--ckpt_path", type=str, default="outputs/2025-04-02/17-24-34/logs/AC_DiT/ki31d1pj/checkpoints/last.ckpt", help="检查点路径")
    
    # 评估参数
    parser.add_argument("--num_sequences", type=int, default=5, help="评估序列数量")
    parser.add_argument("--debug", action="store_true", help="开启渲染模式")
    parser.add_argument("--device", default=0, type=int, help="CUDA设备ID")
    
    # 采样参数
    parser.add_argument("--sampler_type", type=str, default="ddim", help="采样器类型 (ddim, dpm_solver)")
    parser.add_argument("--num_sampling_steps", type=int, default=15, help="采样步数")
    
    # 是否使用相对观测
    parser.add_argument("--rel_obs", type=bool, default=False, help="是否使用相对观测")
    
    # 每次预测执行的步数
    parser.add_argument("--steps_per_pred", type=int, default=10, help="每次预测执行的步数 (1-10)")
    
        # 每次预测执行的步数
    parser.add_argument("--future_pred", action="store_true", help="加载模型是否带图像预测")

    args = parser.parse_args()
    
    # 设置CUDA设备
    torch.cuda.set_device(args.device)
 
    # 创建配置
    cfg = OmegaConf.create({
        "calvin_env_path": args.dataset_path,
        "num_sequences": args.num_sequences,
        "debug": args.debug,
        "device": args.device,
        "sampler_type": args.sampler_type,
        "num_sampling_steps": args.num_sampling_steps,
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "rel_obs": args.rel_obs,
        "steps_per_pred": args.steps_per_pred,
        "future_pred": args.future_pred
    })
    
    # 创建环境和模型
    env = make_env(args.dataset_path, debug=args.debug)
    model = ACDiTAgent(cfg, device_id=args.device)
    
    
    # 评估
    print(f"\n评估模型 (采样步数: {args.num_sampling_steps}, 采样器: {args.sampler_type}, 每次预测执行步数: {args.steps_per_pred}, 相对观测: {args.rel_obs})")
    evaluate_policy(
        model, env, cfg, 
        eval_log_dir=args.eval_log_dir, 
        debug=args.debug
    )


if __name__ == "__main__":
    main() 