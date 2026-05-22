#!/usr/bin/env python
"""
RAD-2 评估脚本
加载训练好的Generator+Discriminator，在CARLA中闭环评估

使用方法:
1. 启动CARLA服务器
2. python test_rad2.py --checkpoint /mnt/d/checkpoints/xxx/rad2_phase3_best.pth
"""
import os
import sys
import logging
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.carla_env import CarlaEndToEndEnv
from models.shared_cnn import SharedCNN
from models.trajectory_generator import TrajectoryGenerator
from models.trajectory_discriminator import TrajectoryDiscriminator
from controllers.pure_pursuit import PurePursuitController
from configs.default_config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate(env, shared_cnn, generator, discriminator, controller,
             num_episodes=10, device='cpu'):
    """评估RAD-2模型"""
    shared_cnn.eval()
    generator.eval()
    discriminator.eval()

    episode_rewards = []
    episode_lengths = []
    collision_count = 0
    offroad_count = 0

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            ego_state = info.get('ego_state', np.zeros(3, dtype=np.float32))
            obs_tensor = torch.tensor(obs, dtype=torch.uint8).to(device)
            ego_tensor = torch.tensor(ego_state, dtype=torch.float32).to(device)

            with torch.no_grad():
                visual_feat, _ = shared_cnn(obs_tensor)
                visual_feat = visual_feat.squeeze(0)
                candidates = generator.generate(visual_feat, ego_tensor, N=5)

                scores, _ = discriminator.score_candidates(
                    visual_feat, ego_tensor, candidates
                )
                best_idx = scores.squeeze(0).argmax(dim=0).item()
                selected_traj = candidates[0, best_idx].cpu().numpy()

            speed = info.get('speed', 0.0)
            action = controller.compute_control(selected_traj, speed)
            next_obs, reward, done, _, info = env.step(action, raw_control=True)

            if info.get('collided', False):
                collision_count += 1
            if done and not info.get('collided', False):
                # 检查是否出道路
                if reward < -100:
                    offroad_count += 1

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            # 可视化
            if hasattr(env, 'render'):
                env.render(mode='human')

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        logger.info(f"Episode {ep}/{num_episodes} | "
                     f"Reward: {episode_reward:.2f} | "
                     f"Length: {episode_steps} | "
                     f"Collided: {info.get('collided', False)}")

    # 统计
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    collision_rate = collision_count / num_episodes
    safety_rate = 1.0 - collision_rate

    logger.info("=" * 50)
    logger.info(f"评估结果 ({num_episodes} episodes):")
    logger.info(f"  Avg Reward: {avg_reward:.2f}")
    logger.info(f"  Avg Length: {avg_length:.1f}")
    logger.info(f"  Collision Rate: {collision_rate:.3f}")
    logger.info(f"  Safety Rate: {safety_rate:.3f}")
    logger.info("=" * 50)

    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'collision_rate': collision_rate,
        'safety_rate': safety_rate,
    }


def main():
    parser = argparse.ArgumentParser(description='RAD-2 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint路径')
    parser.add_argument('--episodes', type=int, default=10,
                        help='评估episodes数')
    args = parser.parse_args()

    config = CONFIG
    device = torch.device(config['rad2_train']['device'])

    # 创建模型
    shared_cnn = SharedCNN(input_shape=(80, 160, 3))
    generator = TrajectoryGenerator(config['generator'])
    discriminator = TrajectoryDiscriminator(config['discriminator'])
    controller = PurePursuitController(config['controller'])

    # 加载checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    shared_cnn.load_state_dict(ckpt['shared_cnn'])
    generator.load_state_dict(ckpt['generator'])
    if 'discriminator' in ckpt:
        discriminator.load_state_dict(ckpt['discriminator'])
    logger.info(f"Checkpoint loaded: {args.checkpoint}")

    shared_cnn.to(device)
    generator.to(device)
    discriminator.to(device)

    # 创建环境
    env = CarlaEndToEndEnv(config['env'])
    if not env.connect():
        logger.error("无法连接CARLA服务器")
        sys.exit(1)

    # 评估
    results = evaluate(env, shared_cnn, generator, discriminator,
                       controller, num_episodes=args.episodes, device=device)

    env.close()


if __name__ == "__main__":
    main()
