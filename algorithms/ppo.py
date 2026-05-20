"""
PPO (Proximal Policy Optimization) 实现
适合连续控制任务，端到端训练
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random
import os
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    经验回放缓冲区 - 用于收集PPO样本
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def push(self, obs, action, log_prob, value, reward, done):
        """添加一个转换"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

        if len(self.observations) > self.capacity:
            self.observations.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

    def get_batches(self, batch_size):
        """将收集的数据分成小批次"""
        n_samples = len(self.observations)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_idx = indices[start_idx:end_idx]

            obs_batch = torch.tensor(np.array([self.observations[i] for i in batch_idx]), dtype=torch.uint8)
            action_batch = torch.tensor(np.array([self.actions[i] for i in batch_idx]), dtype=torch.float32)
            log_prob_batch = torch.tensor(np.array([self.log_probs[i] for i in batch_idx]), dtype=torch.float32)
            value_batch = torch.tensor(np.array([self.values[i] for i in batch_idx]), dtype=torch.float32)

            yield obs_batch, action_batch, log_prob_batch, value_batch

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """使用GAE计算优势和回报"""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv = 0

        # 添加最后一个value
        if self.dones[-1]:
            last_val = 0
        else:
            last_val = self.values[-1]

        for t in reversed(range(n)):
            next_val = self.values[t+1] if t+1 < n else last_val
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_adv = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_adv

        # 计算回报
        returns = advantages + np.array(self.values, dtype=np.float32)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def clear(self):
        """清空缓冲区"""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.observations)


class PPO:
    """
    PPO 近端策略优化算法
    """

    def __init__(self, model, config, device='cpu'):
        """
        参数:
            model: Actor-Critic模型
            config: 配置字典
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.config = config

        # 超参数
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.update_epochs = config.get('update_epochs', 10)
        self.batch_size = config.get('batch_size', 64)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 经验缓冲区
        self.buffer_size = config.get('buffer_size', 2048)
        self.buffer = ReplayBuffer(self.buffer_size)

        # 移动模型到设备
        self.model.to(self.device)

        # TensorBoard日志
        self.writer = None

        logger.info(f"PPO initialized on device: {device}")
        logger.info(f"Learning rate: {self.lr}, Clip epsilon: {self.clip_epsilon}")

    def init_tensorboard(self, log_dir):
        """初始化TensorBoard"""
        self.writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard initialized, log dir: {log_dir}")

    def select_action(self, obs, deterministic=False):
        """选择动作"""
        obs_tensor = torch.tensor(obs, dtype=torch.uint8).to(self.device)
        action, log_prob, value = self.model.get_action(obs_tensor, deterministic=deterministic)
        return action, log_prob, value

    def store_transition(self, obs, action, log_prob, value, reward, done):
        """存储转换"""
        self.buffer.push(obs, action, log_prob, value, reward, done)

    def update(self):
        """更新策略和价值函数"""
        if len(self.buffer) < self.batch_size:
            return None

        # 计算回报和优势
        returns_array, advantages_array = self.buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda
        )

        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        n_samples = len(self.buffer)
        all_obs = torch.tensor(np.array(self.buffer.observations), dtype=torch.uint8).to(self.device)
        all_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        all_old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32).to(self.device)
        all_old_values = torch.tensor(np.array(self.buffer.values), dtype=torch.float32).to(self.device)
        all_returns = torch.tensor(returns_array, dtype=torch.float32).to(self.device)
        all_advantages = torch.tensor(advantages_array, dtype=torch.float32).to(self.device)

        # 标准化优势
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            # 随机打乱索引
            indices = torch.randperm(n_samples).to(self.device)
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]

                obs_batch = all_obs[batch_idx]
                action_batch = all_actions[batch_idx]
                old_log_prob_batch = all_old_log_probs[batch_idx]
                old_value_batch = all_old_values[batch_idx]
                batch_returns = all_returns[batch_idx]
                batch_advantages = all_advantages[batch_idx]

                # 评估动作
                new_log_prob, entropy, new_value = self.model.evaluate_actions(obs_batch, action_batch)

                # 比率
                ratio = torch.exp(new_log_prob - old_log_prob_batch)

                # clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss with clipping
                value_clipped = old_value_batch + torch.clamp(
                    new_value - old_value_batch,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_original = (new_value - batch_returns).pow(2)
                value_loss_clipped = (value_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_original, value_loss_clipped).mean()

                # total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 清空缓冲区
        self.buffer.clear()

        avg_policy_loss = total_policy_loss / n_updates if n_updates > 0 else 0
        avg_value_loss = total_value_loss / n_updates if n_updates > 0 else 0
        avg_entropy = total_entropy / n_updates if n_updates > 0 else 0

        logger.info(f"Update done: policy_loss={avg_policy_loss:.4f}, "
                   f"value_loss={avg_value_loss:.4f}, entropy={avg_entropy:.4f}")

        # 记录到TensorBoard
        if self.writer is not None:
            global_step = len(self.buffer)  # 使用总样本数作为global step
            self.writer.add_scalar('loss/policy', avg_policy_loss, global_step)
            self.writer.add_scalar('loss/value', avg_value_loss, global_step)
            self.writer.add_scalar('entropy', avg_entropy, global_step)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }

    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")
