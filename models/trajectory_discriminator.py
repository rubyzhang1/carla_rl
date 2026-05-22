"""
轨迹判别器 (Trajectory Discriminator)
RL训练的评分网络，对候选轨迹打分
输入: 视觉特征 + 自车状态 + 候选轨迹 → (score, value)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TrajectoryDiscriminator(nn.Module):
    """
    轨迹判别器

    结构:
    - 条件编码器: visual_feat + ego_state → condition_vec (独立权重)
    - 轨迹编码器: traj(T_wp*2) → traj_embed
    - 打分头: concat(cond, traj_embed) → score
    - 价值头: concat(cond, traj_embed) → value (GAE用)
    """
    def __init__(self, config):
        super().__init__()

        visual_feat_dim = config.get('visual_feat_dim', 7168)
        ego_state_dim = config.get('ego_state_dim', 3)
        condition_dim = config.get('condition_dim', 256)
        self.traj_dim = config.get('traj_dim', 12)
        traj_embed_dim = config.get('traj_embed_dim', 128)
        hidden_dim = config.get('hidden_dim', 256)

        # 条件编码器（独立权重，不与Generator共享）
        self.condition_encoder = nn.Sequential(
            nn.Linear(visual_feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, condition_dim),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.condition_combine = nn.Sequential(
            nn.Linear(condition_dim + 32, condition_dim),
            nn.ReLU(),
        )

        # 轨迹编码器
        self.traj_encoder = nn.Sequential(
            nn.Linear(self.traj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, traj_embed_dim),
            nn.ReLU(),
        )

        # 共享隐藏层
        self.shared_fc = nn.Sequential(
            nn.Linear(condition_dim + traj_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )

        # 打分头
        self.score_head = nn.Linear(128, 1)

        # 价值头
        self.value_head = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 打分头小初始化
        nn.init.orthogonal_(self.score_head.weight, gain=0.01)
        nn.init.constant_(self.score_head.bias, 0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)

    def encode_condition(self, visual_feat, ego_state):
        """编码条件向量"""
        v = self.condition_encoder(visual_feat)
        e = self.ego_encoder(ego_state)
        return self.condition_combine(torch.cat([v, e], dim=-1))

    def forward(self, visual_feat, ego_state, trajectory):
        """
        前向传播：对一条轨迹打分

        参数:
            visual_feat: (B, visual_feat_dim)
            ego_state: (B, ego_state_dim)
            trajectory: (B, traj_dim) 或 (B, num_waypoints, 2)

        返回:
            score: (B,) 轨迹得分
            value: (B,) 状态价值
        """
        if trajectory.ndim == 3:
            trajectory = trajectory.reshape(trajectory.size(0), -1)

        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        if ego_state.ndim == 1:
            ego_state = ego_state.unsqueeze(0)

        condition = self.encode_condition(visual_feat, ego_state)
        traj_embed = self.traj_encoder(trajectory)

        combined = torch.cat([condition, traj_embed], dim=-1)
        shared = self.shared_fc(combined)

        score = self.score_head(shared).squeeze(-1)
        value = self.value_head(shared).squeeze(-1)

        return score, value

    def score_candidates(self, visual_feat, ego_state, candidates):
        """
        对N条候选轨迹批量打分

        参数:
            visual_feat: (B, visual_feat_dim)
            ego_state: (B, ego_state_dim)
            candidates: (B, N, num_waypoints, 2) 或 (B, N, traj_dim)

        返回:
            scores: (B, N)
            values: (B,) 状态价值（只算一次）
        """
        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        if ego_state.ndim == 1:
            ego_state = ego_state.unsqueeze(0)

        B, N = candidates.size(0), candidates.size(1)

        # 展平候选: (B*N, ...)
        traj_flat = candidates.reshape(B * N, -1)
        visual_feat_expanded = visual_feat.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        ego_state_expanded = ego_state.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        scores, _ = self.forward(visual_feat_expanded, ego_state_expanded, traj_flat)
        scores = scores.reshape(B, N)

        # 价值只算一次（不依赖轨迹）
        condition = self.encode_condition(visual_feat, ego_state)
        # 用零轨迹获取基础value
        dummy_traj = torch.zeros(B, self.traj_dim, device=visual_feat.device)
        traj_embed = self.traj_encoder(dummy_traj)
        combined = torch.cat([condition, traj_embed], dim=-1)
        shared = self.shared_fc(combined)
        values = self.value_head(shared).squeeze(-1)

        return scores, values
