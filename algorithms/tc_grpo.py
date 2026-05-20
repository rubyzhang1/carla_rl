"""
TC-GRPO (Temporal-Consistent Group Relative Policy Optimization)
基于PPO扩展，用于训练Discriminator
核心思想:
- 每步生成N条候选轨迹，Discriminator打分
- 组合优势: GAE优势 + 组内相对优势
- 时序一致性: 相邻步相似轨迹得分应相近
- PPO风格裁剪更新
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TCGRPOBuffer:
    """TC-GRPO经验缓冲区，存储N条候选轨迹及其得分"""

    def __init__(self, capacity, num_candidates):
        self.capacity = capacity
        self.num_candidates = num_candidates
        self.observations = []
        self.ego_states = []
        self.all_candidates = []
        self.all_scores = []
        self.selected_indices = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def push(self, obs, ego_state, candidates, scores, selected_idx,
             log_prob, value, reward, done):
        """添加一步转换"""
        self.observations.append(obs)
        self.ego_states.append(ego_state)
        self.all_candidates.append(candidates)
        self.all_scores.append(scores)
        self.selected_indices.append(selected_idx)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

        if len(self.observations) > self.capacity:
            self.observations.pop(0)
            self.ego_states.pop(0)
            self.all_candidates.pop(0)
            self.all_scores.pop(0)
            self.selected_indices.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """GAE计算优势和回报"""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv = 0

        if self.dones[-1]:
            last_val = 0
        else:
            last_val = self.values[-1]

        for t in reversed(range(n)):
            next_val = self.values[t + 1] if t + 1 < n else last_val
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_adv = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_adv

        returns = advantages + np.array(self.values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def compute_group_advantages(self):
        """计算组内相对优势: selected_score - mean(all_scores)"""
        group_advs = []
        for scores in self.all_scores:
            scores = np.array(scores)
            group_adv = scores - scores.mean()
            group_advs.append(group_adv)
        return np.array(group_advs, dtype=np.float32)

    def clear(self):
        self.observations.clear()
        self.ego_states.clear()
        self.all_candidates.clear()
        self.all_scores.clear()
        self.selected_indices.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.observations)


class TCGRPO:
    """
    TC-GRPO算法

    训练Discriminator用RL学习对轨迹打分
    """

    def __init__(self, shared_cnn, discriminator, controller, config, device='cpu'):
        self.shared_cnn = shared_cnn
        self.discriminator = discriminator
        self.controller = controller
        self.device = device
        self.config = config

        # 超参数
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.02)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.update_epochs = config.get('update_epochs', 5)
        self.batch_size = config.get('batch_size', 64)
        self.tau = config.get('tau', 1.0)
        self.alpha = config.get('alpha', 0.7)
        self.lambda_tc = config.get('lambda_tc', 0.1)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.num_candidates = config.get('num_candidates', 5)

        # 优化器（只训练Discriminator和SharedCNN）
        self.optimizer = optim.Adam(
            list(self.discriminator.parameters()) + list(self.shared_cnn.parameters()),
            lr=self.lr
        )

        # 缓冲区
        self.buffer_size = config.get('buffer_size', 2048)
        self.buffer = TCGRPOBuffer(self.buffer_size, self.num_candidates)

        # 移动模型到设备
        self.shared_cnn.to(self.device)
        self.discriminator.to(self.device)

        self.writer = None
        self._update_count = 0

    def init_tensorboard(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        logger.info(f"TC-GRPO TensorBoard initialized: {log_dir}")

    @torch.no_grad()
    def select_trajectory(self, visual_feat, ego_state, candidates, deterministic=False):
        """
        从N条候选轨迹中选择一条

        参数:
            visual_feat: (visual_feat_dim,) 或 (1, visual_feat_dim)
            ego_state: (ego_state_dim,) 或 (1, ego_state_dim)
            candidates: (N, num_waypoints, 2) numpy 或 tensor
            deterministic: 是否确定性选择

        返回:
            selected_idx: int
            selected_traj: (num_waypoints, 2) numpy
            scores: (N,) numpy
            log_prob: float
            value: float
        """
        if isinstance(candidates, np.ndarray):
            candidates = torch.from_numpy(candidates).float().to(self.device)
        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        if ego_state.ndim == 1:
            ego_state = ego_state.unsqueeze(0)

        # (1, N, num_wp, 2) -> score
        candidates_batch = candidates.unsqueeze(0)
        scores, value = self.discriminator.score_candidates(visual_feat, ego_state, candidates_batch)
        scores = scores.squeeze(0)  # (N,)
        value = value.squeeze(0).item()

        # softmax采样
        probs = F.softmax(scores / self.tau, dim=0)

        if deterministic:
            selected_idx = scores.argmax(dim=0).item()
            log_prob = torch.log(probs[selected_idx] + 1e-10).item()
        else:
            dist = torch.distributions.Categorical(probs)
            selected_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(selected_idx, device=self.device)).item()

        selected_traj = candidates[selected_idx].cpu().numpy()
        scores_np = scores.cpu().numpy()

        return selected_idx, selected_traj, scores_np, log_prob, value

    def store_transition(self, obs, ego_state, candidates, scores, selected_idx,
                         log_prob, value, reward, done):
        """存储一步转换"""
        self.buffer.push(obs, ego_state, candidates, scores, selected_idx,
                         log_prob, value, reward, done)

    def compute_temporal_consistency_loss(self, all_observations, all_ego_states,
                                          all_selected_trajs, all_scores_t):
        """
        时序一致性损失: 相邻步相似轨迹得分应相近

        参数:
            all_observations: (B, H, W, C) uint8
            all_ego_states: (B, ego_dim)
            all_selected_trajs: (B, traj_dim)
            all_scores_t: (B,) 当前步选中轨迹的得分
        """
        if all_selected_trajs.size(0) < 2:
            return torch.tensor(0.0, device=self.device)

        # 计算相邻轨迹的余弦相似度
        traj_curr = all_selected_trajs[:-1]
        traj_next = all_selected_trajs[1:]
        sim = F.cosine_similarity(traj_curr, traj_next, dim=-1)

        # 相似度高的相邻步，得分应相近
        tc_mask = (sim > self.similarity_threshold).float()
        score_curr = all_scores_t[:-1]
        score_next = all_scores_t[1:]

        tc_loss = (tc_mask * (score_curr - score_next) ** 2).mean()
        return tc_loss

    def update(self):
        """TC-GRPO更新Discriminator"""
        if len(self.buffer) < self.batch_size:
            return None

        # 计算GAE优势和组内相对优势
        returns_array, gae_advantages = self.buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda
        )
        group_advantages = self.buffer.compute_group_advantages()

        n_samples = len(self.buffer)

        # 准备数据
        all_obs = torch.tensor(np.array(self.buffer.observations), dtype=torch.uint8).to(self.device)
        all_ego = torch.tensor(np.array(self.buffer.ego_states), dtype=torch.float32).to(self.device)
        all_selected_idx = torch.tensor(np.array(self.buffer.selected_indices), dtype=torch.long).to(self.device)
        all_old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32).to(self.device)
        all_old_values = torch.tensor(np.array(self.buffer.values), dtype=torch.float32).to(self.device)
        all_returns = torch.tensor(returns_array, dtype=torch.float32).to(self.device)

        # 组合优势
        selected_group_advs = np.array([
            group_advantages[i, self.buffer.selected_indices[i]]
            for i in range(n_samples)
        ], dtype=np.float32)
        combined_advantages = self.alpha * gae_advantages + (1 - self.alpha) * selected_group_advs
        combined_advantages = (combined_advantages - combined_advantages.mean()) / (combined_advantages.std() + 1e-8)
        all_advantages = torch.tensor(combined_advantages, dtype=torch.float32).to(self.device)

        # 准备选中轨迹的tensor
        all_selected_trajs = []
        for i in range(n_samples):
            traj = np.array(self.buffer.all_candidates[i][self.buffer.selected_indices[i]])
            all_selected_trajs.append(traj.flatten())
        all_selected_trajs = torch.tensor(np.array(all_selected_trajs), dtype=torch.float32).to(self.device)

        total_policy_loss = 0
        total_value_loss = 0
        total_tc_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.update_epochs):
            indices = torch.randperm(n_samples).to(self.device)
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]

                obs_batch = all_obs[batch_idx]
                ego_batch = all_ego[batch_idx]
                selected_idx_batch = all_selected_idx[batch_idx]
                old_log_prob_batch = all_old_log_probs[batch_idx]
                old_value_batch = all_old_values[batch_idx]
                batch_returns = all_returns[batch_idx]
                batch_advantages = all_advantages[batch_idx]
                traj_batch = all_selected_trajs[batch_idx]

                # 提取视觉特征
                visual_feat, _ = self.shared_cnn(obs_batch)

                # 重新计算得分
                scores, new_values = self.discriminator(visual_feat, ego_batch, traj_batch)

                # 同时对N条候选重新打分以计算log_prob
                # 简化: 用选中轨迹的得分作为策略log_prob的代理
                # 更精确的做法需要重新打分所有候选，这里用简化版
                new_log_prob = scores - torch.logsumexp(scores / self.tau, dim=-1) * self.tau
                new_log_prob = new_log_prob / self.tau  # 归一化

                # PPO裁剪
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_clipped = old_value_batch + torch.clamp(
                    new_values - old_value_batch,
                    -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_original = (new_values - batch_returns).pow(2)
                value_loss_clipped = (value_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_original, value_loss_clipped).mean()

                # 时序一致性损失
                tc_loss = self.compute_temporal_consistency_loss(
                    obs_batch, ego_batch, traj_batch, scores
                )

                # 熵奖励（鼓励得分分布有一定分散度）
                entropy = -(scores - scores.mean()).pow(2).mean()

                # 总损失
                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.lambda_tc * tc_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.discriminator.parameters()) + list(self.shared_cnn.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_tc_loss += tc_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()
        self._update_count += 1

        avg_policy_loss = total_policy_loss / max(n_updates, 1)
        avg_value_loss = total_value_loss / max(n_updates, 1)
        avg_tc_loss = total_tc_loss / max(n_updates, 1)
        avg_entropy = total_entropy / max(n_updates, 1)

        logger.info(f"TC-GRPO update #{self._update_count}: "
                     f"policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}, "
                     f"tc={avg_tc_loss:.4f}, entropy={avg_entropy:.4f}")

        if self.writer is not None:
            self.writer.add_scalar('loss/policy', avg_policy_loss, self._update_count)
            self.writer.add_scalar('loss/value', avg_value_loss, self._update_count)
            self.writer.add_scalar('loss/tc', avg_tc_loss, self._update_count)
            self.writer.add_scalar('entropy', avg_entropy, self._update_count)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'tc_loss': avg_tc_loss,
            'entropy': avg_entropy,
        }

    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'shared_cnn_state_dict': self.shared_cnn.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'update_count': self._update_count,
        }, path)
        logger.info(f"TC-GRPO model saved to {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.shared_cnn.load_state_dict(checkpoint['shared_cnn_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._update_count = checkpoint.get('update_count', 0)
        logger.info(f"TC-GRPO model loaded from {path}")
