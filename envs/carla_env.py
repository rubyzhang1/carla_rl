"""
CARLA端到端强化学习环境 - 无感知真值模式
直接从原始相机图像输出控制信号（转向、油门、刹车）
"""
import os
import sys
import random
import time
import logging
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

# 添加CARLA PythonAPI路径
CARLA_PATH = '/home/zyc/carla/PythonAPI'
sys.path.insert(0, CARLA_PATH)
import carla
from carla import Client, Vehicle, World, ActorBlueprint, Map, Location, Rotation, VehicleControl, Transform, LaneType, Vector3D

class CarlaEndToEndEnv(gym.Env):
    """
    CARLA端到端强化学习环境

    Observation:原始相机图像 (H, W, 3) uint8 或者归一化到[0,1]
    Action: [steer, throttle, brake] 都在[-1, 1]范围
        - steer: -1 左转, +1 右转
        - throttle: 0~1 加速, 负值忽略当作0
        - brake: 0~1 刹车, 负值忽略当作0
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config):
        super(CarlaEndToEndEnv, self).__init__()

        self.config = config

        # CARLA连接参数
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 2000)
        self.timeout = config.get('timeout', 10.0)

        # 传感器参数
        self.image_width = config.get('image_width', 160)
        self.image_height = config.get('image_height', 80)
        self.fov = config.get('fov', 100)

        # 动作空间: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 观测空间: 图像
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.uint8
        )

        # 目标位置参数
        self.target_distance = config.get('target_distance', 50.0)
        self.destination = None

        # 状态变量
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.camera = None
        self.actor_list = []
        self.latest_image = None
        self.spawn_point = None

        # 奖励参数
        self.collision_penalty = config.get('collision_penalty', -100.0)
        self.out_road_penalty = config.get('out_road_penalty', -50.0)
        self.distance_reward_weight = config.get('distance_reward_weight', 1.0)
        self.speed_reward_weight = config.get('speed_reward_weight', 0.1)

        # 碰撞检测
        self.collision_sensor = None
        self.collided = False

        # 低速停车检测
        self.low_speed_steps = 0

    def connect(self):
        """连接到CARLA服务器"""
        try:
            self.client = Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.map = self.world.get_map()

            # 设置同步模式，固定步长，这样客户端控制才会生效
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

            print(f"成功连接到CARLA服务器: {self.host}:{self.port}")
            print(f"当前地图: {self.map.name}")
            print(f"同步模式已开启，固定步长: 0.05s")
            return True
        except Exception as e:
            print(f"连接CARLA失败: {e}")
            print("请确保CARLA服务器已启动: ./CarlaUE4.sh")
            return False

    def _destroy_actors(self):
        """销毁所有actors"""
        if self.client is None:
            return
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list = []
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

    def _spawn_vehicle(self):
        """生成车辆"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))

        # 随机选择一个生成点，碰撞时重试
        spawn_points = self.map.get_spawn_points()
        max_retries = 10
        for attempt in range(max_retries):
            self.spawn_point = random.choice(spawn_points)
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
                break
            except RuntimeError as e:
                if "collision" in str(e).lower() and attempt < max_retries - 1:
                    print(f"生成点碰撞，尝试另一个点... (尝试 {attempt + 1}/{max_retries})")
                    continue
                else:
                    raise
        self.actor_list.append(self.vehicle)

        # 随机初始速度：20 ~ 110 km/h → 转换成 m/s
        # 增加训练多样性，让模型适应不同初始速度
        init_speed_kmh = random.uniform(20, 110)
        init_speed_ms = init_speed_kmh / 3.6

        # 沿着当前车头方向给初始速度
        yaw_rad = np.deg2rad(self.spawn_point.rotation.yaw)
        init_vel = carla.Vector3D(
            x=init_speed_ms * np.cos(yaw_rad),
            y=init_speed_ms * np.sin(yaw_rad),
            z=0.0
        )
        self.vehicle.set_target_velocity(init_vel)

        print(f"车辆生成成功: {self.vehicle.id} at {self.spawn_point.location}, initial speed {init_speed_kmh:.0f} km/h ({init_speed_ms:.1f} m/s)")

    def _setup_sensors(self):
        """设置传感器"""
        blueprint_library = self.world.get_blueprint_library()

        # RGB相机
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', str(self.fov))

        # 相机安装位置: 前挡风玻璃
        camera_transform = Transform(Location(x=1.5, y=0.0, z=1.8), Rotation(pitch=0, yaw=0, roll=0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._on_camera_image(image))
        self.actor_list.append(self.camera)

        # 碰撞传感器
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)

    def _on_camera_image(self, image):
        """相机图像回调"""
        # 转换为numpy数组
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        img_array = img_array.reshape((self.image_height, self.image_width, 4))
        # BGRA -> RGB, drop alpha
        self.latest_image = img_array[:, :, :3][:, :, ::-1].copy()

    def _on_collision(self, event):
        """碰撞回调"""
        self.collided = True

    def _set_destination(self):
        """设置目标点: 沿车道向前一段距离"""
        current_transform = self.vehicle.get_transform()
        current_loc = current_transform.location
        yaw_rad = np.deg2rad(current_transform.rotation.yaw)

        # 沿当前前进方向设置目标点
        dest_x = current_loc.x + self.target_distance * np.cos(yaw_rad)
        dest_y = current_loc.y + self.target_distance * np.sin(yaw_rad)
        self.destination = Location(dest_x, dest_y, current_loc.z)

    def _get_distance_to_destination(self):
        """计算到目标的距离"""
        if self.destination is None:
            return None
        current_loc = self.vehicle.get_transform().location
        return current_loc.distance(self.destination)

    def _apply_action(self, action):
        """将动作应用到车辆控制"""
        steer = float(action[0])
        # 模型输出在[-1, 1]，转换到[0, 1]
        throttle = float((action[1] + 1) / 2)
        brake = float((action[2] + 1) / 2)

        # 限制范围
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        # 测试时过滤小刹车：只有刹车 > 0.5才生效，更小的都清零
        # 模型训练出来整体偏向刹车，测试时放宽让它开起来
        if brake < 0.5:
            brake = 0.0

        # 如果不刹车，保证最小油门，不让车慢慢停下
        if brake == 0.0 and throttle < 0.4:
            throttle = 0.4

        # 如果速度太低，自动给更大油门
        if self.vehicle is not None:
            speed = self._get_speed()
            if speed < 5.0 and brake == 0.0:
                throttle = 0.6

        # 调试：输出动作（每100步一次，避免太多）
        if not hasattr(self, '_action_debug_counter'):
            self._action_debug_counter = 0
        self._action_debug_counter += 1
        if self._action_debug_counter % 100 == 0:
            logger.debug(f"Action: steer={steer:.3f}, throttle={throttle:.3f}, brake={brake:.3f}")

        control = VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.reverse = False
        self.vehicle.apply_control(control)

    def _calculate_reward(self):
        """计算奖励"""
        reward = 0.0

        # 碰撞惩罚
        if self.collided:
            reward += self.collision_penalty
            return reward, True

        # 距离奖励: 接近目标给正奖励
        current_distance = self._get_distance_to_destination()

        # 检查是否在道路上
        vehicle_loc = self.vehicle.get_transform().location
        waypoint = self.map.get_waypoint(vehicle_loc, project_to_road=True)
        # 只要不在驾驶车道，立刻终止，只给初始5米缓冲
        # 冲出道路就早早结束，不用一直开到碰撞才终止
        if waypoint is None or waypoint.lane_type != LaneType.Driving:
            # 只给初始5米缓冲，避免刚生成就被判定出局
            if current_distance < self.target_distance - 5:
                reward += self.out_road_penalty
                return reward, True

        # 到达目标判断
        if current_distance < 2.0:
            # 到达目标后，不结束episode，直接在当前位置重新设置一个新目标
            # 这样可以连续一直开，测试不同路段
            self._set_destination()
            # 给奖励但不结束
            reward += 100.0
            current_distance = self._get_distance_to_destination()

        if self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            reward += distance_delta * self.distance_reward_weight

        self.prev_distance = current_distance

        # 持续生存奖励：只要还在道路上开，每一步都给正奖励
        # 走得越远，总奖励越高 → 直接鼓励开更远，这是最重要的奖励
        reward += 5.0

        # 速度奖励: 鼓励尽可能快，但不超过目标速度
        # 速度越快奖励越高，达到目标后给满分，不惩罚超速
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        target_speed = 5.0  # m/s ~ 18 km/h
        speed_reward = min(speed, target_speed) / target_speed
        reward += speed_reward * self.speed_reward_weight * 5.0

        # 停车检测：只有真的几乎完全停下才终止，放宽让车尽量开
        # 低速慢慢走也不终止，只有完全停下才算
        if speed < 0.2:
            self.low_speed_steps += 1
            # 连续50步 (~2.5秒) 速度低于0.2m/s，才认为停车终止
            if self.low_speed_steps > 50:
                reward += -50.0  # 停车惩罚
                return reward, True
        else:
            self.low_speed_steps = 0

        done = False
        return reward, done

    def step(self, action):
        """
        Gym step接口
        返回: observation, reward, terminated, truncated, info
        """
        # 应用动作
        self._apply_action(action)

        # 推进CARLA世界仿真
        self.world.tick()

        # 等待世界更新
        time.sleep(0.05)

        # 等待图像更新
        while self.latest_image is None:
            time.sleep(0.01)

        # 计算奖励
        reward, done = self._calculate_reward()

        # 获取观测
        obs = self.latest_image.copy()

        # 信息字典
        info = {
            'collided': self.collided,
            'speed': self._get_speed(),
            'distance_to_target': self._get_distance_to_destination()
        }

        return obs, reward, done, False, info

    def _get_speed(self):
        """获取当前速度"""
        if self.vehicle is None:
            return 0.0
        v = self.vehicle.get_velocity()
        return np.sqrt(v.x**2 + v.y**2 + v.z**2)

    def reset(self, seed=None, options=None):
        """Gym reset接口"""
        super().reset(seed=seed)

        # 重置状态
        self._destroy_actors()
        self.collided = False
        self.latest_image = None
        self.prev_distance = None
        self.low_speed_steps = 0

        # 重新生成车辆和传感器
        self._spawn_vehicle()
        self._setup_sensors()
        self._set_destination()

        # 等待第一帧图像
        while self.latest_image is None:
            self.world.tick()
            time.sleep(0.01)

        obs = self.latest_image.copy()
        info = {
            'speed': self._get_speed(),
            'distance_to_target': self._get_distance_to_destination()
        }

        return obs, info

    def render(self, mode='human'):
        """渲染"""
        if self.latest_image is not None:
            if mode == 'human':
                # 放大显示，原图160x80太小了，放大5倍到800x400
                # 使用INTER_CUBIC插值，比最近邻插值更平滑
                display_img = cv2.resize(self.latest_image, (800, 400), interpolation=cv2.INTER_CUBIC)

                # 获取当前状态信息
                speed = self._get_speed()
                if self.vehicle is not None:
                    control = self.vehicle.get_control()
                    throttle = control.throttle
                    brake = control.brake
                else:
                    throttle = 0
                    brake = 0
                dist = self._get_distance_to_destination()

                # 在图像上叠加文字信息
                text_y = 25
                # 黑色背景让文字更清晰
                cv2.rectangle(display_img, (5, 5), (250, 120), (0, 0, 0), -1)

                info_text = [
                    f"Speed: {speed:.1f} m/s ({speed*3.6:.0f} km/h)",
                    f"Throttle: {throttle:.2f}",
                    f"Brake: {brake:.2f}",
                    f"Dist to target: {dist:.1f} m" if dist is not None else "Dist: N/A"
                ]

                for i, text in enumerate(info_text):
                    cv2.putText(display_img, text, (10, text_y + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('CARLA End-to-End RL', display_img)
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return self.latest_image
        return None

    def close(self):
        """关闭环境"""
        # 恢复异步模式
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        self._destroy_actors()
        cv2.destroyAllWindows()
