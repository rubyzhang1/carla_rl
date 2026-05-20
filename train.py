#!/usr/bin/env python
"""
CARLA端到端强化学习训练脚本
无感知真值模式，直接从图像学习控制

使用方法:
1. 先启动CARLA服务器:
   cd /home/zyc/carla
   ./CarlaUE4.sh --quality-level=Low -RenderOffscreen

2. 然后运行训练:
   python train.py
"""
import os
import sys
import logging
import time
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.carla_env import CarlaEndToEndEnv
from models.end_to_end_cnn import CarlaEndToEndCNN
from algorithms.ppo import PPO
from configs.default_config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_checkpoint_in_dir(dir_path):
    """在目录中找到最新的完整检查点，优先找carla_ppo_final.pth
    如果最新文件损坏，自动往前找"""
    import glob
    if os.path.exists(os.path.join(dir_path, "carla_ppo_final.pth")) and os.path.getsize(os.path.join(dir_path, "carla_ppo_final.pth")) > 1000000:
        return os.path.join(dir_path, "carla_ppo_final.pth")
    # 收集所有完整检查点，过滤掉空文件
    valid_pths = []
    for pth in glob.glob(os.path.join(dir_path, "carla_ppo_ep*.pth")):
        if os.path.getsize(pth) > 1000000:  # 大于1MB认为文件完整
            valid_pths.append(pth)
    if not valid_pths:
        return None
    # 按episode排序
    valid_pths.sort(key=lambda x: int(os.path.basename(x).split('_')[2].replace('ep', '').split('_')[0]))
    # 从最新往后找，第一个完整的就是要找的
    for pth in reversed(valid_pths):
        return pth
    return None

def main(resume_checkpoint=None):
    config = CONFIG
    env_config = config['env']
    model_config = config['model']
    ppo_config = config['ppo']
    train_config = config['train']

    # 创建环境
    env = CarlaEndToEndEnv(env_config)
    if not env.connect():
        logger.error("无法连接到CARLA服务器，请先启动CARLA: ./CarlaUE4.sh")
        sys.exit(1)

    # 创建模型 - 使用改进的CNN+LSTM时序模型
    from models.end_to_end_cnn_lstm import CarlaEndToEndCNN_LSTM
    model = CarlaEndToEndCNN_LSTM(
        input_shape=model_config['input_shape'],
        action_dim=model_config['action_dim']
    )

    # 创建PPO
    device = torch.device(train_config['device'])
    ppo = PPO(model, ppo_config, device=device)

    # 如果指定了检查点，恢复训练
    if resume_checkpoint is not None:
        if os.path.isdir(resume_checkpoint):
            # 输入是目录，自动找检查点
            ckpt = find_checkpoint_in_dir(resume_checkpoint)
            if ckpt is None:
                logger.error(f"目录 {resume_checkpoint} 中找不到检查点文件")
                sys.exit(1)
            resume_checkpoint = ckpt
        logger.info(f"从检查点恢复训练: {resume_checkpoint}")
        ppo.load(resume_checkpoint)

    # 创建保存目录：每次训练单独一个文件夹，用时间戳命名
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    train_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_checkpoint_dir = os.path.join(train_config['checkpoint_dir'], train_timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # 创建TensorBoard日志
    log_dir = os.path.join("logs", train_timestamp)
    os.makedirs(log_dir, exist_ok=True)
    ppo.init_tensorboard(log_dir)
    logger.info(f"TensorBoard logs saved to: {log_dir}")
    logger.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")

    logger.info(f"开始训练... run_id: {train_timestamp}")

    for episode in range(1, train_config['total_episodes'] + 1):
        obs, info = env.reset()
        # 新回合重置LSTM隐状态
        if hasattr(model, 'reset_hidden'):
            model.reset_hidden()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            # 选择动作
            action, log_prob, value = ppo.select_action(obs)

            # 热身：前200个episodes强制刹车=0，让模型充分体验一直开得到高奖励
            # 建立正确的直觉"不刹车总奖励更高"后再放开让它自己学
            if episode <= 200:
                action[2] = -1.0  # 转换后就是0，不刹车

            # 调整动作范围: throttle和brake都是[0,1]
            # action[0] steer: [-1, 1] already
            action[1] = (action[1] + 1) / 2  # [-1, 1] -> [0, 1]
            action[2] = (action[2] + 1) / 2

            # 执行一步
            next_obs, reward, done, _, info = env.step(action)

            # 存储转换
            if log_prob is not None:
                ppo.store_transition(obs, action, log_prob.item(), value.item(), reward, done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # 定期更新
            if total_steps % train_config['update_interval'] == 0:
                logger.info(f"更新模型 (total steps: {total_steps})")
                ppo.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)

        # 日志输出
        if episode % train_config['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-train_config['log_interval']:])
            avg_length = np.mean(episode_lengths[-train_config['log_interval']:])
            elapsed = time.time() - start_time
            logger.info(
                f"Episode {episode}/{train_config['total_episodes']} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.1f} | "
                f"Total Steps: {total_steps} | "
                f"Time: {elapsed:.1f}s"
            )
            # 记录到TensorBoard
            if ppo.writer is not None:
                ppo.writer.add_scalar('train/avg_reward', avg_reward, episode)
                ppo.writer.add_scalar('train/avg_length', avg_length, episode)
                ppo.writer.add_scalar('train/total_steps', total_steps, episode)

        # 记录每个episode的奖励
        if ppo.writer is not None:
            ppo.writer.add_scalar('train/episode_reward', episode_reward, episode)
            ppo.writer.add_scalar('train/episode_length', episode_steps, episode)

        # 保存模型到本次训练的单独文件夹
        if episode % train_config['save_interval'] == 0:
            save_path = os.path.join(
                run_checkpoint_dir,
                f"carla_ppo_ep{episode}_rew{avg_reward:.0f}.pth"
            )
            ppo.save(save_path)

    # 保存最终模型
    final_path = os.path.join(run_checkpoint_dir, "carla_ppo_final.pth")
    ppo.save(final_path)
    logger.info(f"所有checkpoints保存到: {run_checkpoint_dir}")

    logger.info("训练完成!")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=None,
                        help='总训练episodes，默认使用配置文件中的值')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复继续训练，例如: --resume /mnt/d/checkpoints/20260514_xxxxxx/carla_ppo_ep250_rew80.pth')
    args = parser.parse_args()

    # 如果命令行指定了episodes，覆盖配置
    if args.episodes is not None:
        CONFIG['train']['total_episodes'] = args.episodes

    main(args.resume)
