#!/usr/bin/env python
"""
测试训练好的端到端模型
"""
import os
import sys
import logging
import time
import numpy as np
import torch
import cv2

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


def test_model(checkpoint_path, num_episodes=5, render=True):
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

    # 创建PPO并加载权重
    device = torch.device(train_config['device'])
    ppo = PPO(model, ppo_config, device=device)
    ppo.load(checkpoint_path)
    model.eval()

    logger.info(f"加载模型: {checkpoint_path}")
    logger.info(f"开始测试，共 {num_episodes} 个episodes")

    # 统计数据
    all_rewards = []
    all_steps = []
    collision_count = 0
    out_road_count = 0
    stopped_count = 0

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        # 新回合重置LSTM隐状态
        if hasattr(model, 'reset_hidden'):
            model.reset_hidden()
        done = False
        episode_reward = 0
        step = 0
        collided_this = False
        out_road_this = False
        stopped_this = False

        logger.info(f"\nEpisode {episode} 开始")

        while not done:
            # 确定性动作选择
            action, _, _ = ppo.select_action(obs, deterministic=True)

            # 执行
            next_obs, reward, done, _, info = env.step(action)

            if render:
                env.render('human')
                time.sleep(0.05)

            obs = next_obs
            episode_reward += reward
            step += 1

            if done:
                # 判断终止原因
                if info.get('collided', False):
                    collision_count += 1
                    collided_this = True
                elif info.get('speed', 0) < 0.5:
                    stopped_count += 1
                    stopped_this = True
                else:
                    out_road_count += 1
                    out_road_this = True

                all_rewards.append(episode_reward)
                all_steps.append(step)

                logger.info(
                    f"Episode {episode} 结束: "
                    f"Reward={episode_reward:.2f}, "
                    f"Steps={step}, "
                    f"Collided={info['collided']}, "
                    f"Final Speed={info['speed']:.2f} m/s, "
                    f"Reason: {'Collision' if collided_this else 'Low Speed' if stopped_this else 'Out of Road'}"
                )
                break

    # 输出统计汇总
    logger.info("\n" + "="*60)
    logger.info("测试结果汇总评价")
    logger.info("="*60)
    logger.info(f"总测试episodes: {len(all_rewards)}")
    logger.info(f"平均奖励: {np.mean(all_rewards):.2f}")
    logger.info(f"平均每局步数: {np.mean(all_steps):.1f}  ≈ {np.mean(all_steps)*0.05:.1f} 秒")
    logger.info(f"中位数步数: {np.median(all_steps):.0f}")
    logger.info(f"最长步数: {np.max(all_steps)}")
    logger.info(f"最短步数: {np.min(all_steps)}")
    logger.info("-"*60)
    logger.info(f"终止原因统计:")
    logger.info(f"  碰撞终止: {collision_count}  ({collision_count/len(all_rewards)*100:.1f}%)")
    logger.info(f"  停车终止: {stopped_count}  ({stopped_count/len(all_rewards)*100:.1f}%)")
    logger.info(f"  出道路终止: {out_road_count}  ({out_road_count/len(all_rewards)*100:.1f}%)")
    logger.info("="*60)

    # 评价结论
    avg_steps = np.mean(all_steps)
    if avg_steps > 200:
        conclusion = "优秀 - 模型能开很远，驾驶策略稳定"
    elif avg_steps > 100:
        conclusion = "良好 - 模型开得不错，有一定泛化能力"
    elif avg_steps > 50:
        conclusion = "一般 - 能开一段距离，但需要改进"
    else:
        conclusion = "较差 - 模型很快就停或撞，需要更多训练"
    logger.info(f"评价结论: {conclusion}")
    logger.info("="*60)

    env.close()
    cv2.destroyAllWindows()


def find_latest_checkpoint():
    """自动找到最新训练的最终模型"""
    import glob
    checkpoint_root = 'checkpoints'
    # 查找所有子文件夹，按修改时间排序
    dirs = [d for d in glob.glob(f"{checkpoint_root}/*") if os.path.isdir(d)]
    if not dirs:
        # 没有子文件夹，看根目录有没有
        if os.path.exists(f"{checkpoint_root}/carla_ppo_final.pth"):
            return f"{checkpoint_root}/carla_ppo_final.pth"
        return None

    # 按修改时间排序，最新的在最后
    dirs.sort(key=lambda d: os.path.getmtime(d))
    latest_dir = dirs[-1]
    latest_final = os.path.join(latest_dir, "carla_ppo_final.pth")
    if os.path.exists(latest_final):
        return latest_final
    # 如果final不存在，找最大episode
    pths = glob.glob(f"{latest_dir}/carla_ppo_ep*.pth")
    if not pths:
        return None
    pths.sort()
    return pths[-1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径，默认自动找最新训练的final')
    parser.add_argument('--episodes', type=int, default=20,
                        help='测试多少个episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='不渲染图像')
    args = parser.parse_args()

    if args.checkpoint is None:
        # 自动找最新
        ckpt = find_latest_checkpoint()
        if ckpt is None:
            logger.error("找不到检查点，请指定--checkpoint路径")
            sys.exit(1)
        logger.info(f"自动加载最新检查点: {ckpt}")
        args.checkpoint = ckpt

    test_model(args.checkpoint, args.episodes, render=not args.no_render)
