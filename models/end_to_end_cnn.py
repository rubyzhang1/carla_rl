"""
端到端自动驾驶CNN模型 - Actor-Critic架构
输入: RGB图像
输出: 均值(steer, throttle, brake) + 标准差 + 状态价值
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalize_image(img):
    """归一化图像，使用ImageNet统计"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    # 转为float并归一化到[0,1]
    img = img.float() / 255.0
    # ImageNet均值和标准差，标准化
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    if img.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    return (img - mean) / std


class CarlaEndToEndCNN(nn.Module):
    """
    端到端Actor-Critic CNN网络

    结构:
    - 多个卷积层提取图像特征
    - 全连接层映射到特征向量
    - Actor头输出动作均值和标准差
    - Critic头输出状态价值
    """

    def __init__(self, input_shape=(80, 160, 3), action_dim=3):
        super(CarlaEndToEndCNN, self).__init__()

        self.input_height, self.input_width, self.input_channels = input_shape
        self.action_dim = action_dim

        # 卷积层 - 加深网络 + 添加BatchNorm提升训练稳定性
        self.conv_layers = nn.Sequential(
            # 输入: 3x80x160
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 计算卷积输出尺寸
        conv_out_size = self._get_conv_out_size()

        # 特征提取
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Actor头: 输出动作均值
        self.actor_mean = nn.Linear(512, action_dim)
        # 可学习的动作标准差
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic头: 输出状态价值
        self.critic = nn.Linear(512, 1)

        # 初始化权重
        self._init_weights()

        # 改进初始化bias: 让初始更容易输出高油门、低刹车
        # steer初始偏置0，throttle初始偏置+0.5（输出更大值），brake初始偏置-0.5（输出更小值）
        with torch.no_grad():
            self.actor_mean.bias[0] = 0.0      # steer 居中
            self.actor_mean.bias[1] = 0.5      # throttle 偏向正 → 更大油门
            self.actor_mean.bias[2] = -0.5     # brake 偏向负 → 更小刹车

    def _get_conv_out_size(self):
        """计算卷积层输出尺寸"""
        dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        out = self.conv_layers(dummy_input)
        return int(np.prod(out.size()))

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        # 最后一层较小初始化
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: (batch_size, H, W, C) or (batch_size, C, H, W) 像素值0-255

        返回:
            action_mean: (batch_size, action_dim)
            action_std: (batch_size, action_dim)
            value: (batch_size, 1)
        """
        # 处理输入格式
        if x.shape[1:] != (self.input_channels, self.input_height, self.input_width):
            # 如果输入是 (B, H, W, C), 转换为 (B, C, H, W)
            if x.ndim == 4:
                x = x.permute(0, 3, 1, 2)

        # 归一化
        x = normalize_image(x)

        # CNN特征提取
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        # Actor输出
        action_mean = self.actor_mean(x)
        # 将动作范围压缩到[-1, 1]
        action_mean = torch.tanh(action_mean)
        # throttle和brake应该在[0,1], 需要调整吗？我们保持-1~1让PPO处理
        action_std = torch.exp(self.actor_log_std)
        action_std = action_std.expand_as(action_mean)

        # Critic输出
        value = self.critic(x)

        return action_mean, action_std, value.squeeze(-1)

    def get_action(self, x, deterministic=False):
        """
        获取动作用于与环境交互

        参数:
            x: 单张图像 (H, W, C) 或批量图像 (B, H, W, C)
            deterministic: 是否确定性动作(测试时用True)

        返回:
            action: 动作数组
            log_prob: 对数概率
            value: 状态价值
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)  # 添加batch维度

        with torch.no_grad():
            action_mean, action_std, value = self.forward(x)

            if deterministic:
                action = action_mean
            else:
                # 从正态分布采样
                normal = torch.distributions.Normal(action_mean, action_std)
                action = normal.sample()

            # 裁剪动作范围
            action = torch.clamp(action, -1.0, 1.0)

            if not deterministic:
                log_prob = normal.log_prob(action).sum(dim=-1)
            else:
                log_prob = None

        return action.squeeze(0).cpu().numpy(), log_prob, value.squeeze(-1).cpu().numpy()

    def evaluate_actions(self, x, action):
        """
        评估动作，用于训练

        参数:
            x: 批次观测
            action: 批次动作

        返回:
            log_prob: 对数概率
            entropy: 熵
            value: 状态价值
        """
        action_mean, action_std, value = self.forward(x)
        normal = torch.distributions.Normal(action_mean, action_std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)

        return log_prob, entropy, value
