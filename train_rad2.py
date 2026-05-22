#!/usr/bin/env python
"""
RAD-2 训练脚本: Generator-Discriminator框架
三阶段训练:
  Phase 1: IL预训练Generator (用CARLA waypoint真值)
  Phase 2: RL训练Discriminator (CARLA闭环)
  Phase 3: OGO在线微调Generator (交替训练)

使用方法:
1. 启动CARLA服务器:
   cd /home/zyc/carla
   ./CarlaUE4.sh --quality-level=Low -RenderOffscreen

2. 运行训练:
   python train_rad2.py
   python train_rad2.py --phase 2 --resume /mnt/d/checkpoints/xxx/gen_ep500.pth
"""
import os
import sys
import logging
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.carla_env import CarlaEndToEndEnv
from models.shared_cnn import SharedCNN
from models.trajectory_generator import TrajectoryGenerator
from models.trajectory_discriminator import TrajectoryDiscriminator
from controllers.pure_pursuit import PurePursuitController
from algorithms.tc_grpo import TCGRPO
from configs.default_config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_checkpoint(shared_cnn, generator, discriminator, optimizer, phase,
                    episode, checkpoint_dir, timestamp, extra=None):
    """保存checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"rad2_phase{phase}_ep{episode}.pth")
    state = {
        'shared_cnn': shared_cnn.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'phase': phase,
        'episode': episode,
        'timestamp': timestamp,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")
    return path


# ========== Phase 1: IL预训练Generator ==========
def train_phase1(env, shared_cnn, generator, controller, config):
    """IL预训练Generator，用CARLA waypoint API做真值"""
    gen_config = config['generator']
    train_config = config['rad2_train']
    device = torch.device(train_config['device'])

    shared_cnn.to(device)
    generator.to(device)
    shared_cnn.eval()  # CNN特征提取器在Phase 1冻结
    for p in shared_cnn.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(generator.parameters(), lr=train_config['il_lr'])
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    checkpoint_dir = os.path.join(train_config['checkpoint_dir'], timestamp)
    log_dir = os.path.join(train_config['log_dir'], f"rad2_phase1_{timestamp}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)

    total_episodes = train_config['il_episodes']
    batch_size = train_config['il_batch_size']

    logger.info(f"=== Phase 1: IL预训练Generator ({total_episodes} episodes) ===")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        episode_loss = 0.0
        episode_steps = 0
        episode_data = []

        done = False
        while not done:
            ego_state = info.get('ego_state', np.zeros(3, dtype=np.float32))
            gt_waypoints = info.get('gt_waypoints', np.zeros((6, 2), dtype=np.float32))

            obs_tensor = torch.tensor(obs, dtype=torch.uint8).to(device)
            ego_tensor = torch.tensor(ego_state, dtype=torch.float32).to(device)
            gt_tensor = torch.tensor(gt_waypoints, dtype=torch.float32).to(device)

            with torch.no_grad():
                visual_feat, _ = shared_cnn(obs_tensor)
                visual_feat = visual_feat.squeeze(0)

            episode_data.append((visual_feat.cpu(), ego_tensor.cpu(), gt_tensor.cpu()))

            # 生成轨迹并用控制器执行（收集下一步数据）
            with torch.no_grad():
                candidates = generator.generate(visual_feat, ego_tensor, N=1)
            selected_traj = candidates[0, 0].cpu().numpy()
            speed = info.get('speed', 0.0)
            action = controller.compute_control(selected_traj, speed)
            next_obs, reward, done, _, info = env.step(action, raw_control=True)
            obs = next_obs
            episode_steps += 1

        # 用episode数据训练Generator
        if len(episode_data) >= batch_size:
            np.random.shuffle(episode_data)
            for start in range(0, len(episode_data), batch_size):
                batch = episode_data[start:start + batch_size]
                vf = torch.stack([d[0] for d in batch]).to(device)
                es = torch.stack([d[1] for d in batch]).to(device)
                gt = torch.stack([d[2] for d in batch]).to(device)

                loss = generator.training_loss(vf, es, gt)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
                optimizer.step()
                episode_loss += loss.item()

        avg_loss = episode_loss / max(1, len(episode_data) // batch_size)
        writer.add_scalar('loss/il_loss', avg_loss, episode)
        writer.add_scalar('episode/steps', episode_steps, episode)

        if episode % train_config['log_interval'] == 0:
            logger.info(f"Phase1 Ep {episode}/{total_episodes} | "
                         f"IL Loss: {avg_loss:.4f} | Steps: {episode_steps}")

        if episode % train_config['save_interval'] == 0:
            save_checkpoint(shared_cnn, generator,
                            TrajectoryDiscriminator(config['discriminator']),
                            optimizer, 1, episode, checkpoint_dir, timestamp,
                            {'il_loss': avg_loss})

    # Phase 1结束保存
    save_checkpoint(shared_cnn, generator,
                    TrajectoryDiscriminator(config['discriminator']),
                    optimizer, 1, total_episodes, checkpoint_dir, timestamp)
    writer.close()
    logger.info("Phase 1 完成!")
    return checkpoint_dir, timestamp


# ========== Phase 2: RL训练Discriminator ==========
def train_phase2(env, shared_cnn, generator, discriminator, controller, config,
                 checkpoint_dir=None, timestamp=None, start_episode=0):
    """RL训练Discriminator，Generator冻结"""
    gen_config = config['generator']
    train_config = config['rad2_train']
    grpo_config = config['tc_grpo']
    device = torch.device(train_config['device'])

    shared_cnn.to(device)
    generator.to(device)
    discriminator.to(device)

    # Generator冻结
    for p in generator.parameters():
        p.requires_grad = False
    generator.eval()

    tc_grpo = TCGRPO(shared_cnn, discriminator, controller, grpo_config, device=device)

    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%m%S", time.localtime())
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(train_config['checkpoint_dir'], timestamp)
    log_dir = os.path.join(train_config['log_dir'], f"rad2_phase2_{timestamp}")
    tc_grpo.init_tensorboard(log_dir)

    total_episodes = train_config['rl_episodes']
    update_interval = train_config['rl_update_interval']

    logger.info(f"=== Phase 2: RL训练Discriminator ({total_episodes} episodes) ===")

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = -float('inf')

    for episode in range(start_episode + 1, start_episode + total_episodes + 1):
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
                candidates = generator.generate(visual_feat, ego_tensor,
                                                 N=gen_config['num_candidates'])

            # Discriminator选择轨迹
            selected_idx, selected_traj, scores, log_prob, value = \
                tc_grpo.select_trajectory(visual_feat, ego_tensor, candidates[0])

            # 控制器执行
            speed = info.get('speed', 0.0)
            action = controller.compute_control(selected_traj, speed)
            next_obs, reward, done, _, info = env.step(action, raw_control=True)

            # 存储转换
            candidates_np = candidates[0].cpu().numpy()
            tc_grpo.store_transition(
                obs, ego_state, candidates_np, scores, selected_idx,
                log_prob, value, reward, done
            )

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # 定期更新
            if total_steps % update_interval == 0 and len(tc_grpo.buffer) >= grpo_config['batch_size']:
                tc_grpo.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)

        if episode % train_config['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-train_config['log_interval']:])
            avg_length = np.mean(episode_lengths[-train_config['log_interval']:])
            logger.info(f"Phase2 Ep {episode} | Avg Reward: {avg_reward:.2f} | "
                         f"Avg Length: {avg_length:.1f} | Steps: {total_steps}")

        if episode % train_config['save_interval'] == 0:
            save_checkpoint(shared_cnn, generator, discriminator,
                            tc_grpo.optimizer, 2, episode, checkpoint_dir, timestamp,
                            {'avg_reward': np.mean(episode_rewards[-10:])})

            # 保存最优模型
            recent_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                best_path = os.path.join(checkpoint_dir, "rad2_phase2_best.pth")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'shared_cnn': shared_cnn.state_dict(),
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'avg_reward': recent_avg,
                    'episode': episode,
                }, best_path)
                logger.info(f"Best model saved: reward={recent_avg:.2f}")

    # Phase 2结束保存
    save_checkpoint(shared_cnn, generator, discriminator,
                    tc_grpo.optimizer, 2,
                    start_episode + total_episodes, checkpoint_dir, timestamp)
    logger.info("Phase 2 完成!")
    return tc_grpo


# ========== Phase 3: OGO在线微调Generator ==========
def train_phase3(env, shared_cnn, generator, discriminator, controller, tc_grpo, config,
                 checkpoint_dir, timestamp, start_episode=0):
    """OGO在线微调Generator，与Phase 2交替训练"""
    gen_config = config['generator']
    train_config = config['rad2_train']
    grpo_config = config['tc_grpo']
    device = torch.device(train_config['device'])

    # 解冻Generator
    for p in generator.parameters():
        p.requires_grad = True
    generator.train()

    gen_optimizer = torch.optim.Adam(
        list(generator.parameters()) + list(shared_cnn.parameters()),
        lr=train_config['ogo_lr']
    )

    ogo_interval = train_config['ogo_interval']
    lambda_div = train_config['ogo_lambda_div']

    total_episodes = train_config.get('ogo_episodes', 400)

    logger.info(f"=== Phase 3: OGO在线微调Generator ({total_episodes} episodes) ===")

    total_steps = 0
    episode_rewards = []
    gen_update_count = 0
    best_avg_reward = -float('inf')

    for episode in range(start_episode + 1, start_episode + total_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_gen_data = []

        done = False
        while not done:
            ego_state = info.get('ego_state', np.zeros(3, dtype=np.float32))
            obs_tensor = torch.tensor(obs, dtype=torch.uint8).to(device)
            ego_tensor = torch.tensor(ego_state, dtype=torch.float32).to(device)

            visual_feat, _ = shared_cnn(obs_tensor)
            visual_feat = visual_feat.squeeze(0)

            with torch.no_grad():
                candidates = generator.generate(visual_feat, ego_tensor,
                                                 N=gen_config['num_candidates'])

            # Discriminator选择轨迹
            selected_idx, selected_traj, scores, log_prob, value = \
                tc_grpo.select_trajectory(visual_feat, ego_tensor, candidates[0])

            # 控制器执行
            speed = info.get('speed', 0.0)
            action = controller.compute_control(selected_traj, speed)
            next_obs, reward, done, _, info = env.step(action, raw_control=True)

            # 存储到Discriminator buffer
            candidates_np = candidates[0].cpu().numpy()
            tc_grpo.store_transition(
                obs, ego_state, candidates_np, scores, selected_idx,
                log_prob, value, reward, done
            )

            # 保存Generator训练数据
            episode_gen_data.append((visual_feat.detach(), ego_tensor.detach(),
                                      candidates[0].detach()))

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Discriminator更新
            if total_steps % train_config['rl_update_interval'] == 0 and \
               len(tc_grpo.buffer) >= grpo_config['batch_size']:
                tc_grpo.update()

                # OGO: 每隔ogo_interval次Discriminator更新，更新一次Generator
                if gen_update_count % ogo_interval == 0 and len(episode_gen_data) > 0:
                    _ogo_update(shared_cnn, generator, discriminator,
                                gen_optimizer, episode_gen_data, lambda_div, device)
                gen_update_count += 1

        episode_rewards.append(episode_reward)

        if episode % train_config['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-train_config['log_interval']:])
            logger.info(f"Phase3 Ep {episode} | Avg Reward: {avg_reward:.2f} | "
                         f"Gen Updates: {gen_update_count}")

        if episode % train_config['save_interval'] == 0:
            save_checkpoint(shared_cnn, generator, discriminator,
                            gen_optimizer, 3, episode, checkpoint_dir, timestamp,
                            {'avg_reward': np.mean(episode_rewards[-10:])})

            recent_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                best_path = os.path.join(checkpoint_dir, "rad2_phase3_best.pth")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'shared_cnn': shared_cnn.state_dict(),
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'avg_reward': recent_avg,
                    'episode': episode,
                }, best_path)

    logger.info("Phase 3 完成!")


def _ogo_update(shared_cnn, generator, discriminator, gen_optimizer,
                gen_data, lambda_div, device):
    """OGO单步更新Generator"""
    if len(gen_data) < 4:
        return

    # 采样batch
    indices = np.random.choice(len(gen_data), size=min(16, len(gen_data)), replace=False)
    visual_feats = torch.stack([gen_data[i][0] for i in indices]).to(device)
    ego_states = torch.stack([gen_data[i][1] for i in indices]).to(device)

    # Generator生成候选（保留梯度）
    N = generator.num_candidates
    B = visual_feats.size(0)

    # 需要梯度，不能用@torch.no_grad()
    condition = generator.encode_condition(visual_feats, ego_states)
    condition_expanded = condition.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

    # DDPM采样（简化版: 只用少量步数加速）
    x = torch.randn(B * N, generator.traj_dim, device=device)
    for t_idx in reversed(range(5)):  # OGO用更少步数
        t = torch.full((B * N,), t_idx, device=device, dtype=torch.long)
        pred_noise = generator.forward(x, t, condition_expanded)
        alpha_t = generator.alphas_cumprod[t_idx]
        x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x = x0_pred

    candidates = x.reshape(B, N, generator.num_waypoints, 2)

    # Discriminator打分（不更新Discriminator）
    with torch.no_grad():
        scores, _ = discriminator.score_candidates(visual_feats, ego_states, candidates)

    # Generator loss: 最大化discriminator得分
    gen_loss = -scores.mean()

    # 多样性损失: 鼓励候选轨迹之间有差异
    candidates_flat = candidates.reshape(B, N, -1)
    pairwise_dist = torch.cdist(candidates_flat, candidates_flat, p=2)
    # 排除自身距离
    mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
    diversity_loss = -pairwise_dist[mask].mean()

    total_loss = gen_loss + lambda_div * diversity_loss

    gen_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(generator.parameters()) + list(shared_cnn.parameters()), 0.5
    )
    gen_optimizer.step()

    logger.debug(f"OGO update: gen_loss={gen_loss.item():.4f}, div_loss={diversity_loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description='RAD-2 Training')
    parser.add_argument('--phase', type=int, default=0,
                        help='从哪个Phase开始 (0=全部, 1/2/3)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复')
    parser.add_argument('--episodes', type=int, default=None,
                        help='覆盖Phase 1的episode数')
    args = parser.parse_args()

    config = CONFIG
    env_config = config['env']
    gen_config = config['generator']
    disc_config = config['discriminator']
    ctrl_config = config['controller']
    train_config = config['rad2_train']
    device = torch.device(train_config['device'])

    # 创建环境
    env = CarlaEndToEndEnv(env_config)
    if not env.connect():
        logger.error("无法连接CARLA服务器")
        sys.exit(1)

    # 创建模型
    shared_cnn = SharedCNN(input_shape=(80, 160, 3))
    generator = TrajectoryGenerator(gen_config)
    discriminator = TrajectoryDiscriminator(disc_config)
    controller = PurePursuitController(ctrl_config)

    # 恢复checkpoint
    start_phase = args.phase if args.phase > 0 else 1
    start_episode = 0
    checkpoint_dir = None
    timestamp = None
    tc_grpo = None

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        shared_cnn.load_state_dict(ckpt['shared_cnn'])
        generator.load_state_dict(ckpt['generator'])
        if 'discriminator' in ckpt:
            discriminator.load_state_dict(ckpt['discriminator'])
        start_phase = ckpt.get('phase', 1)
        start_episode = ckpt.get('episode', 0)
        logger.info(f"从checkpoint恢复: Phase {start_phase}, Episode {start_episode}")

    if args.episodes is not None:
        train_config['il_episodes'] = args.episodes

    # Phase 1: IL预训练
    if start_phase <= 1:
        checkpoint_dir, timestamp = train_phase1(
            env, shared_cnn, generator, controller, config
        )
        start_phase = 2

    # Phase 2: RL训练Discriminator
    if start_phase <= 2:
        tc_grpo = train_phase2(
            env, shared_cnn, generator, discriminator, controller, config,
            checkpoint_dir, timestamp
        )
        start_phase = 3

    # Phase 3: OGO微调Generator
    if start_phase <= 3 and train_config['ogo_enabled']:
        if tc_grpo is None:
            # 需要创建tc_grpo
            tc_grpo = TCGRPO(shared_cnn, discriminator, controller,
                              config['tc_grpo'], device=device)
        train_phase3(
            env, shared_cnn, generator, discriminator, controller, tc_grpo, config,
            checkpoint_dir, timestamp
        )

    logger.info("RAD-2 训练全部完成!")
    env.close()


if __name__ == "__main__":
    main()
