"""
默认配置文件
端到端强化学习在CARLA上的配置
"""

# CARLA环境配置
ENV_CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    'image_width': 160,
    'image_height': 80,
    'fov': 100,
    'target_distance': 500.0,  # 目标距离(米) - 增大后可以开更远
    # 奖励参数
# 拉大差距：生存奖励很高，碰撞/停车惩罚很重
# 让模型必须学会一直开才能得到高奖励，早停代价极大
# 惩罚不要太重，否则梯度太不稳定
    'collision_penalty': -200.0,    # 碰撞惩罚
    'out_road_penalty': -150.0,    # 出道路惩罚
    'distance_reward_weight': 2.0,
    'speed_reward_weight': 1.0,    # 速度奖励权重更大
}

# PPO算法配置
PPO_CONFIG = {
    'lr': 1e-4,  # 降低学习率，训练更稳定，减少震荡打转
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.05,  # 增大熵正则化，鼓励探索，防止过早收敛
    'max_grad_norm': 0.5,
    'update_epochs': 5,   # 减少更新轮数，防止过拟合当前批次，训练更稳定
    'batch_size': 64,
    'buffer_size': 2048,  # 每次更新收集多少样本
}

# 训练配置
TRAIN_CONFIG = {
    'total_episodes': 600,  # 增加到600episodes，更容易收敛
    'update_interval': 2048,  # 多少步更新一次
    'save_interval': 50,  # 多少个episode保存一次
    'log_interval': 10,
    'checkpoint_dir': '/mnt/d/checkpoints/',
    'log_dir': 'logs/',
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
}

# 模型配置
MODEL_CONFIG = {
    'input_shape': (80, 160, 3),
    'action_dim': 3,
}

# 完整配置
CONFIG = {
    'env': ENV_CONFIG,
    'ppo': PPO_CONFIG,
    'train': TRAIN_CONFIG,
    'model': MODEL_CONFIG,
}
