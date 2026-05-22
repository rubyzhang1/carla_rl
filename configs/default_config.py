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

# 扩散生成器配置
GENERATOR_CONFIG = {
    'visual_feat_dim': 7168,
    'ego_state_dim': 3,
    'condition_dim': 256,
    'traj_dim': 12,               # T_wp * 2 = 6 * 2
    'num_waypoints': 6,
    'waypoint_interval': 2.0,     # waypoint间距(米)
    'num_candidates': 5,          # 每步生成N条候选轨迹
    'diffusion_steps': 20,        # DDPM去噪步数
    'beta_min': 0.0001,
    'beta_max': 0.02,
    'mlp_hidden_dim': 512,
    'num_res_blocks': 2,
    'dropout': 0.1,
}

# 判别器配置
DISCRIMINATOR_CONFIG = {
    'visual_feat_dim': 7168,
    'ego_state_dim': 3,
    'condition_dim': 256,
    'traj_dim': 12,
    'traj_embed_dim': 128,
    'hidden_dim': 256,
}

# TC-GRPO算法配置
TC_GRPO_CONFIG = {
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.02,
    'max_grad_norm': 0.5,
    'update_epochs': 5,
    'batch_size': 64,
    'buffer_size': 2048,
    'num_candidates': 5,
    'tau': 1.0,                   # softmax温度
    'alpha': 0.7,                 # GAE优势权重 vs 组内相对优势
    'lambda_tc': 0.1,             # 时序一致性损失权重
    'similarity_threshold': 0.8,  # 余弦相似度阈值
}

# Pure Pursuit控制器配置
CONTROLLER_CONFIG = {
    'lookahead_distance': 3.0,
    'wheelbase': 2.5,
    'max_speed': 30.0,
    'min_speed': 3.0,
    'max_steering': 0.8,
    'speed_kp': 0.5,
    'min_throttle': 0.3,
}

# RAD-2训练流水线配置
RAD2_TRAIN_CONFIG = {
    # Phase 1: IL预训练Generator
    'il_episodes': 800,
    'il_lr': 1e-4,
    'il_batch_size': 32,
    'il_eval_interval': 50,
    # Phase 2: RL训练Discriminator
    'rl_episodes': 1500,
    'rl_update_interval': 2048,
    # Phase 3: OGO在线微调Generator
    'ogo_enabled': True,
    'ogo_interval': 5,            # 每5次Discriminator更新做1次Generator更新
    'ogo_lr': 1e-5,
    'ogo_lambda_div': 0.1,        # 多样性损失权重
    # 共用
    'save_interval': 50,
    'log_interval': 10,
    'checkpoint_dir': '/mnt/d/checkpoints/',
    'log_dir': 'logs/',
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
}

# 完整配置
CONFIG = {
    'env': ENV_CONFIG,
    'ppo': PPO_CONFIG,
    'train': TRAIN_CONFIG,
    'model': MODEL_CONFIG,
    'generator': GENERATOR_CONFIG,
    'discriminator': DISCRIMINATOR_CONFIG,
    'tc_grpo': TC_GRPO_CONFIG,
    'controller': CONTROLLER_CONFIG,
    'rad2_train': RAD2_TRAIN_CONFIG,
}
