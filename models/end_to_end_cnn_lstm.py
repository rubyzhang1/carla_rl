"""
端到端自动驾驶CNN+LSTM模型 - Actor-Critic架构
输入: 连续RGB图像序列
输出: 均值(steer, throttle, brake) + 标准差 + 状态价值
加入时序建模，利用历史信息做出更稳定决策
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


class CarlaEndToEndCNN_LSTM(nn.Module):
    """
    端到端Actor-Critic CNN + LSTM网络
    利用时序历史信息，决策更平滑稳定，减少不必要刹车

    结构:
    - CNN提取单帧图像特征
    - LSTM建模时序依赖
    - 全连接层映射到特征向量
    - Actor头输出动作均值和标准差
    - Critic头输出状态价值
    """

    def __init__(self, input_shape=(80, 160, 3), action_dim=3, hidden_dim=512, lstm_layers=1):
        super(CarlaEndToEndCNN_LSTM, self).__init__()

        self.input_height, self.input_width, self.input_channels = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # CNN骨干提取图像特征
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
        self.conv_out_dim = conv_out_size // self.input_height // self.input_width * 128
        # 实际conv输出通道就是128，空间形状压缩后flatten维度就是 128 * H * W
        dummy = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        out = self.conv_layers(dummy)
        self.feature_dim = int(np.prod(out.size()[1:]))  # C*H*W

        # LSTM时序建模：输入是CNN特征，输出隐状态
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=0.1 if self.lstm_layers > 1 else 0
        )

        # 特征提取
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
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

        # 保存LSTM隐状态，用于交互时递推
        self._lstm_hidden = None

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

        # LSTM初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        # 最后一层较小初始化
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        """
        前向传播 - 用于训练，输入是整个序列 (seq_len, ...) 或 (batch, seq_len, ...)

        参数:
            x: (batch_size, seq_len, H, W, C)  or  (seq_len, H, W, C)
                像素值0-255

        返回:
            action_mean: (batch_size, seq_len, action_dim)
            action_std: (batch_size, seq_len, action_dim)
            value: (batch_size, seq_len)
        """
        # 处理输入维度
        if x.ndim == 4:
            # (seq_len, H, W, C) -> (1, seq_len, C, H, W)
            x = x.unsqueeze(0)

        if x.shape[-1] == 3:
            # (B, T, H, W, C) -> (B*T, C, H, W)
            B, T, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)

        # CNN提取每帧特征
        x = normalize_image(x)
        features = self.conv_layers(x)  # (B*T, 128, Hc, Wc)
        features = features.reshape(B, T, -1)  # (B, T, feature_dim)

        # LSTM时序建模
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_dim)

        # 全连接层
        lstm_out = lstm_out.reshape(B*T, self.hidden_dim)
        x = self.fc(lstm_out)

        # Actor输出
        action_mean = self.actor_mean(x)  # (B*T, action_dim)
        action_mean = torch.tanh(action_mean)  # 范围[-1, 1]
        action_std = torch.exp(self.actor_log_std)
        action_std = action_std.expand_as(action_mean)

        # Critic输出
        value = self.critic(x)

        # reshape回去
        action_mean = action_mean.reshape(B, T, self.action_dim)
        action_std = action_std.reshape(B, T, self.action_dim)
        value = value.reshape(B, T).squeeze(-1)

        return action_mean, action_std, value

    def get_action(self, x, deterministic=False, reset_hidden=False):
        """
        获取动作，用于与环境交互，递推LSTM隐状态

        参数:
            x: 单张图像 (H, W, C) 像素值0-255
            deterministic: 是否确定性动作
            reset_hidden: 回合重置时重置隐状态

        返回:
            action: 动作数组
            log_prob: 对数概率
            value: 状态价值
        """
        if reset_hidden or self._lstm_hidden is None:
            # 重置LSTM隐状态，新回合开始
            self._lstm_hidden = (
                torch.zeros(self.lstm_layers, 1, self.hidden_dim, device=x.device if isinstance(x, torch.Tensor) else 'cuda'),
                torch.zeros(self.lstm_layers, 1, self.hidden_dim, device=x.device if isinstance(x, torch.Tensor) else 'cuda')
            )

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, C)

        with torch.no_grad():
            B, T, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)
            x = normalize_image(x)
            features = self.conv_layers(x)  # (B*T, 128, Hc, Wc)
            features = features.reshape(B, T, -1)  # (B, T, feature_dim)

            # LSTM递推
            lstm_out, self._lstm_hidden = self.lstm(features, self._lstm_hidden)  # (B, T, hidden_dim)
            lstm_out = lstm_out.reshape(B*T, self.hidden_dim)
            x = self.fc(lstm_out)  # (B*T, 512)

            # Actor输出
            action_mean = self.actor_mean(x)  # (B*T, action_dim)
            action_mean = torch.tanh(action_mean)  # 范围[-1, 1]
            action_std = torch.exp(self.actor_log_std)
            action_std = action_std.expand_as(action_mean)

            # Critic输出
            value = self.critic(x)

            # 取出结果
            action_mean = action_mean[0]
            action_std = action_std[0]
            value = value[0, 0]

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

        return action.cpu().numpy(), log_prob, value.cpu().numpy()

    def evaluate_actions(self, x, action):
        """
        评估动作，用于训练

        参数:
            x: 批次观测 (B, T, H, W, C)
            action: 批次动作 (B, T, action_dim)

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

    def reset_hidden(self):
        """重置LSTM隐状态，新回合调用"""
        self._lstm_hidden = None
