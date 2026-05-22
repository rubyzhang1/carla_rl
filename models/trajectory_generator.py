"""
条件扩散轨迹生成器 (Conditional Diffusion Generator)
基于DDPM，用MLP去噪网络生成候选未来轨迹
输入: 视觉特征 + 自车状态 → N条候选waypoint轨迹
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码用于扩散时间步"""
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        """
        t: (B,) 整数时间步
        返回: (B, dim) 时间嵌入
        """
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class TrajectoryGenerator(nn.Module):
    """
    条件扩散轨迹生成器

    结构:
    - 条件编码器: visual_feat + ego_state → condition_vec
    - 去噪MLP: noisy_traj + timestep_embed + condition → predicted noise
    - DDPM采样生成N条候选轨迹
    """
    def __init__(self, config):
        super().__init__()

        visual_feat_dim = config.get('visual_feat_dim', 7168)
        ego_state_dim = config.get('ego_state_dim', 3)
        condition_dim = config.get('condition_dim', 256)
        self.traj_dim = config.get('traj_dim', 12)  # T_wp * 2
        self.num_waypoints = config.get('num_waypoints', 6)
        self.num_candidates = config.get('num_candidates', 5)
        self.diffusion_steps = config.get('diffusion_steps', 20)
        mlp_hidden_dim = config.get('mlp_hidden_dim', 512)
        num_res_blocks = config.get('num_res_blocks', 2)
        dropout = config.get('dropout', 0.1)

        # beta schedule
        beta_min = config.get('beta_min', 0.0001)
        beta_max = config.get('beta_max', 0.02)
        betas = torch.linspace(beta_min, beta_max, self.diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - self.alphas_cumprod))
        # 后验分布系数
        posterior_variance = betas * (1.0 - torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))
        posterior_mean_coef1 = betas * torch.cat([torch.ones(1), torch.sqrt(self.alphas_cumprod[:-1])]) / (1.0 - self.alphas_cumprod)
        posterior_mean_coef2 = (1.0 - torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

        # 条件编码器
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

        # 时间步嵌入
        self.time_embed = SinusoidalTimeEmbedding(dim=64)

        # 去噪MLP
        input_dim = self.traj_dim + 64 + condition_dim
        layers = [
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResBlock(mlp_hidden_dim))
        layers.extend([
            nn.Linear(mlp_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.traj_dim),
        ])
        self.denoise_net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 最后一层小初始化
        final_linear = self.denoise_net[-1]
        nn.init.orthogonal_(final_linear.weight, gain=0.01)
        nn.init.constant_(final_linear.bias, 0)

    def encode_condition(self, visual_feat, ego_state):
        """编码条件向量"""
        v = self.condition_encoder(visual_feat)
        e = self.ego_encoder(ego_state)
        return self.condition_combine(torch.cat([v, e], dim=-1))

    def forward(self, noisy_traj, t, condition):
        """
        去噪网络前向传播

        参数:
            noisy_traj: (B, traj_dim) 带噪轨迹
            t: (B,) 时间步
            condition: (B, condition_dim) 条件向量

        返回:
            (B, traj_dim) 预测噪声
        """
        t_embed = self.time_embed(t)
        x = torch.cat([noisy_traj, t_embed, condition], dim=-1)
        return self.denoise_net(x)

    def training_loss(self, visual_feat, ego_state, gt_trajectory):
        """
        DDPM训练损失

        参数:
            visual_feat: (B, visual_feat_dim)
            ego_state: (B, ego_state_dim)
            gt_trajectory: (B, traj_dim) 或 (B, num_waypoints, 2)

        返回:
            scalar loss
        """
        if gt_trajectory.ndim == 3:
            gt_trajectory = gt_trajectory.reshape(gt_trajectory.size(0), -1)

        B = gt_trajectory.size(0)
        device = gt_trajectory.device

        # 随机采样时间步
        t = torch.randint(0, self.diffusion_steps, (B,), device=device)

        # 采样噪声
        noise = torch.randn_like(gt_trajectory)

        # 加噪: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        noisy_traj = sqrt_alpha * gt_trajectory + sqrt_one_minus_alpha * noise

        # 预测噪声
        condition = self.encode_condition(visual_feat, ego_state)
        pred_noise = self.forward(noisy_traj, t, condition)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def generate(self, visual_feat, ego_state, N=None, deterministic=False):
        """
        DDPM采样生成N条候选轨迹

        参数:
            visual_feat: (B, visual_feat_dim) 或 (visual_feat_dim,)
            ego_state: (B, ego_state_dim) 或 (ego_state_dim,)
            N: 候选轨迹数量，默认用config
            deterministic: 是否确定性采样

        返回:
            (B, N, num_waypoints, 2) 候选轨迹
        """
        if N is None:
            N = self.num_candidates

        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        if ego_state.ndim == 1:
            ego_state = ego_state.unsqueeze(0)

        B = visual_feat.size(0)
        device = visual_feat.device
        condition = self.encode_condition(visual_feat, ego_state)

        # 扩展为N条候选: (B*N, ...)
        condition_expanded = condition.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        # 从纯噪声开始
        x = torch.randn(B * N, self.traj_dim, device=device)

        # 逆向去噪
        for t_idx in reversed(range(self.diffusion_steps)):
            t = torch.full((B * N,), t_idx, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t, condition_expanded)

            alpha_t = self.alphas_cumprod[t_idx]
            alpha_prev = self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)

            # 预测x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

            # 计算后验均值
            coef1 = self.posterior_mean_coef1[t_idx]
            coef2 = self.posterior_mean_coef2[t_idx]
            mean = coef1 * x0_pred + coef2 * x

            if t_idx > 0:
                log_var = self.posterior_log_variance[t_idx]
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * log_var) * noise
            else:
                x = mean

        # reshape: (B*N, traj_dim) -> (B, N, num_waypoints, 2)
        x = x.reshape(B, N, self.num_waypoints, 2)
        return x

    @torch.no_grad()
    def sample_trajectory(self, visual_feat, ego_state, N=1):
        """采样单条轨迹（评估用）"""
        trajs = self.generate(visual_feat, ego_state, N=N)
        return trajs[:, 0, :, :]  # 取第一条
