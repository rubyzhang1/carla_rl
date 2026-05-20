"""
共享CNN特征提取器
从RGB图像提取视觉特征，供Generator和Discriminator共用
"""
import torch
import torch.nn as nn
import numpy as np


def normalize_image(img):
    """归一化图像，使用ImageNet统计"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    img = img.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    if img.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    return (img - mean) / std


class SharedCNN(nn.Module):
    """
    共享CNN骨干网络
    输入: RGB图像 (B, 3, 80, 160) 或 (H, W, C) numpy
    输出: (flat_features, spatial_features)
        - flat_features: (B, 7168) 展平特征
        - spatial_features: (B, 128, 4, 14) 空间特征图
    """
    def __init__(self, input_shape=(80, 160, 3)):
        super(SharedCNN, self).__init__()
        self.input_height, self.input_width, self.input_channels = input_shape

        self.conv_layers = nn.Sequential(
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

        # 计算特征维度
        dummy = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        out = self.conv_layers(dummy)
        self.feature_dim = int(np.prod(out.size()[1:]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        """
        前向传播

        参数:
            img: (B, 3, 80, 160) tensor 或 (H, W, C) numpy → 内部转换

        返回:
            flat_features: (B, feature_dim)
            spatial_features: (B, 128, 4, 14)
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(next(self.parameters()).device)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.shape[-1] == 3 and img.ndim == 4:
            # (B, H, W, C) -> (B, C, H, W)
            img = img.permute(0, 3, 1, 2)

        img = normalize_image(img)
        spatial = self.conv_layers(img)
        flat = spatial.reshape(spatial.size(0), -1)
        return flat, spatial
