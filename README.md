# CARLA 端到端强化学习 (无感知真值模式)

基于CARLA仿真的端到端自动驾驶强化学习框架，直接从原始相机图像学习转向/油门/刹车控制，不需要感知真值标注，通过奖励信号自动学习。

## 框架结构

```
carla_rl/
├── envs/
│   └── carla_env.py     # CARLA仿真环境Gym封装
├── models/
│   └── end_to_end_cnn.py # Actor-Critic CNN模型
├── algorithms/
│   └── ppo.py           # PPO强化学习算法实现
├── configs/
│   └── default_config.py # 默认配置
├── train.py             # 训练脚本
├── test.py              # 测试脚本
└── requirements.txt     # Python依赖
```

## 快速开始

### 1. 环境准备

确保已经正确安装CARLA:
```bash
cd /home/zyc/carla
# 检查PythonAPI
ls PythonAPI/carla/
```

安装Python依赖:
```bash
cd /home/zyc/carla_rl
pip install -r requirements.txt
# 安装CARLA Python API
easy_install /home/zyc/carla/PythonAPI/carla/dist/carla-*.egg
```

### 2. 启动CARLA服务器

在新终端启动CARLA:
```bash
cd /home/zyc/carla
# 无头模式(训练推荐)
./CarlaUE4.sh --quality-level=Low -RenderOffscreen

# 如果需要可视化，可以用:
# ./CarlaUE4.sh --quality-level=Low
```

### 3. 开始训练

```bash
cd /home/zyc/carla_rl
python train.py
```

模型会定期保存在 `checkpoints/` 目录下。

### 4. 测试训练好的模型

```bash
python test.py --checkpoint checkpoints/carla_ppo_final.pth --episodes 5
```

## 算法说明

### 网络结构
- **输入**: 80x160 RGB相机图像
- **中间**: 3层卷积提取特征 + 1层全连接
- **输出**:
  - Actor头: [steer(-1~1), throttle(-1~1), brake(-1~1)] → 后处理转throttle/brake到0~1
  - Critic头: 状态价值估计

### 强化学习设置
- 算法: PPO (Proximal Policy Optimization)
- 优势估计: GAE (Generalized Advantage Estimation)
- 奖励设计:
  - 距离奖励: 越接近目标奖励越大
  - 速度奖励: 鼓励保持合理速度 (~5m/s)
  - 碰撞惩罚: 碰撞直接结束并给大负奖励
  - 离道惩罚: 离开道路结束并给负奖励
  - 到达目标: 到达终点给+100奖励

### 配置说明

主要配置在 `configs/default_config.py`:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image_width` / `image_height` | 输入图像尺寸 | 160 x 80 |
| `target_distance` | 每次episode目标距离 | 50米 |
| `lr` | 学习率 | 3e-4 |
| `gamma` | 折扣因子 | 0.99 |
| `clip_epsilon` | PPO裁剪系数 | 0.2 |
| `update_epochs` | 每次更新迭代次数 | 10 |
| `batch_size` | 批次大小 | 64 |
| `buffer_size` | 每次更新收集多少样本 | 2048 |
| `total_episodes` | 总训练轮数 | 1000 |

## 训练技巧

1. **从小开始**: 先训练 `target_distance=20~30` 米，让网络快速学会走直线，再逐步增加距离

2. **奖励调参**: 如果车辆不爱动，增加 `speed_reward_weight`；如果开太快乱撞，减小这个值或者增加碰撞惩罚

3. **图形卡**: CNN训练需要CUDA加速，如果没有GPU，降低图像分辨率到 80x128 加快训练

4. **CARLA性能**: 训练时用 `--quality-level=Low -RenderOffscreen` 提升仿真速度

## 可能的改进方向

- **帧堆叠**: 堆叠连续4帧给网络，提供速度信息
- **图像增强**: 随机亮度/对比度增加泛化性
- **复杂场景**: 在多个城镇地图训练，增加障碍物
- **SAC/TD3**: 如果PPO训练不稳定，可以换用off-policy算法
- **更大网络**: 可以使用ResNet18作为backbone
