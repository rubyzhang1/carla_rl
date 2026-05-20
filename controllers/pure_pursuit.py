"""
Pure Pursuit控制器
将waypoint轨迹转换为低层控制信号 [steer, throttle, brake]
"""
import numpy as np


class PurePursuitController:
    """
    Pure Pursuit横向控制 + P速度控制

    输入: ego-centric waypoints (T, 2) + 当前速度
    输出: [steer, throttle, brake]
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.lookahead_distance = config.get('lookahead_distance', 3.0)
        self.wheelbase = config.get('wheelbase', 2.5)
        self.target_speed = config.get('target_speed', 5.0)
        self.max_steering = config.get('max_steering', 0.8)
        self.speed_kp = config.get('speed_kp', 0.5)
        self.min_throttle = config.get('min_throttle', 0.3)

    def compute_control(self, waypoints_ego, current_speed):
        """
        计算控制信号

        参数:
            waypoints_ego: (T, 2) ego坐标系下的未来waypoints, x=前, y=左
            current_speed: float, 当前速度 m/s

        返回:
            [steer, throttle, brake]
        """
        if waypoints_ego is None or len(waypoints_ego) == 0:
            return [0.0, self.min_throttle, 0.0]

        waypoints_ego = np.asarray(waypoints_ego)

        # 1. 找前瞻点: 第一个距离 >= lookahead_distance 的waypoint
        lookahead_point = self._find_lookahead_point(waypoints_ego, current_speed)

        # 2. Pure Pursuit计算转向角
        steer = self._compute_steering(lookahead_point)

        # 3. P控制器计算油门/刹车
        throttle, brake = self._compute_speed_control(current_speed)

        return [steer, throttle, brake]

    def _find_lookahead_point(self, waypoints, current_speed):
        """找到前瞻点，速度越快前瞻距离越大"""
        # 动态前瞻距离: 速度越快看得越远
        dynamic_lookahead = max(
            self.lookahead_distance,
            current_speed * 0.6
        )

        for i in range(len(waypoints)):
            dist = np.sqrt(waypoints[i, 0]**2 + waypoints[i, 1]**2)
            if dist >= dynamic_lookahead:
                return waypoints[i]

        # 如果所有waypoint都比前瞻距离近，用最后一个
        return waypoints[-1]

    def _compute_steering(self, lookahead_point):
        """Pure Pursuit转向角计算"""
        lx, ly = lookahead_point[0], lookahead_point[1]
        dist = np.sqrt(lx**2 + ly**2)

        if dist < 0.1:
            return 0.0

        # Pure Pursuit公式: steer = atan(2 * L * sin(alpha) / L_d)
        alpha = np.arctan2(ly, lx)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), dist)

        # 裁剪到最大转向角
        steer = np.clip(steer, -self.max_steering, self.max_steering)
        return float(steer)

    def _compute_speed_control(self, current_speed):
        """P速度控制器"""
        error = self.target_speed - current_speed

        if error > 0:
            throttle = min(self.speed_kp * error + self.min_throttle, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self.speed_kp * abs(error), 1.0)

        return float(throttle), float(brake)
