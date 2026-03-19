import pdb
from collections import deque
import random


# ==========================================
# 这里是您提供的 RewardCalculator 类 (保持原样)
# ==========================================
import numpy as np
from collections import deque

import numpy as np

class RewardCalculator:
    def __init__(self, T_cabin_set, T_bat_set, T_motor_set):

        # ===== 目标温度 =====
        self.TARGET_CABIN_TEMP = T_cabin_set
        self.TARGET_BATTERY_TEMP = T_bat_set
        self.TARGET_MOTOR_TEMP = T_motor_set

        # ===== 归一化尺度（关键！）=====
        self.temp_scale = 10.0     # 最大温差 (°C)
        self.power_scale = 5.0     # 最大功率 (kW)

        # ===== 权重 =====
        self.cabin_weights = {'temp': 0.7, 'power': 0.3}
        self.coolant_weights = {'temp': 0.7, 'power': 0.3}
        self.refrigerant_weights = {'power': 1.0}

    # ==============================
    # 工具函数
    # ==============================
    def _norm_temp(self, error):
        # 平滑 + 限幅（防止梯度爆炸）
        return (error / self.temp_scale) ** 2

    def _norm_power(self, power):
        return power / self.power_scale

    # ==============================
    # Cabin Agent
    # ==============================
    def calculate_cabin_reward(self, cabin_temp, power):

        temp_error = abs(cabin_temp - self.TARGET_CABIN_TEMP)
        temp_cost = self._norm_temp(temp_error)

        power_cost = self._norm_power(power)

        reward = - (
            self.cabin_weights['temp'] * temp_cost +
            self.cabin_weights['power'] * power_cost
        )

        return reward

    # ==============================
    # Refrigerant Agent
    # ==============================
    def calculate_refrigerant_reward(self, power):

        power_cost = self._norm_power(power)

        reward = - self.refrigerant_weights['power'] * power_cost
        return reward

    # ==============================
    # Coolant Agent
    # ==============================
    def calculate_coolant_reward(self, battery_temp, motor_temp, battery_power, motor_power):

        temp_error = (
            0.5 * abs(motor_temp - self.TARGET_MOTOR_TEMP) +
            0.5 * abs(battery_temp - self.TARGET_BATTERY_TEMP)
        )

        temp_cost = self._norm_temp(temp_error)

        power_cost = self._norm_power(battery_power + motor_power)

        reward = - (
            self.coolant_weights['temp'] * temp_cost +
            self.coolant_weights['power'] * power_cost
        )

        return reward

    def reset(self):
        pass  # 已不需要历史状态
        # self.cabin_ema = None
        # self.battery_ema = None
        # self.motor_ema = None

# ==========================================
# 下面是为您编写的 __main__ 测试示例
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("RewardCalculator 模块测试 (模拟 FMU 数据交互)")
    print("=" * 60)

    # 1. 初始化配置
    # 设定目标温度：座舱 24°C, 电池 30°C, 电机 40°C
    reward_calc = RewardCalculator(T_cabin_set=24.0, T_bat_set=30.0, T_motor_set=40.0)

    # 2. 模拟训练/评估循环 (模拟 10 个时间步)
    total_steps = 10

    for step in range(total_steps):
        # --- 模拟 FMU 返回的数据 (info 字典) ---
        # 假设温度逐渐趋向目标值，功率随机波动
        cabin_temp = 30.0 - step * 0.5 + random.uniform(-0.5, 0.5)  # 从 30 度逐渐降到 25 度
        battery_temp = 35.0 - step * 0.3 + random.uniform(-0.2, 0.2)  # 从 35 度逐渐降到 32 度
        motor_temp = 45.0 - step * 0.4 + random.uniform(-0.3, 0.3)  # 从 45 度逐渐降到 41 度

        # 模拟功率 (单位：kW，随机波动)
        power_cabin = random.uniform(0.5, 1.5)
        power_refrigerant = random.uniform(1.0, 3.0)
        power_battery = random.uniform(0.2, 0.8)
        power_motor = random.uniform(0.3, 0.7)

        # --- 计算各智能体奖励 ---
        r_cabin = reward_calc.calculate_cabin_reward(cabin_temp, power_cabin)
        r_refrig = reward_calc.calculate_refrigerant_reward(power_refrigerant)
        r_coolant = reward_calc.calculate_coolant_reward(battery_temp, motor_temp, power_battery, power_motor)

        # --- 打印当前步信息 ---
        # 检查历史窗口长度 (验证滑动窗口是否生效)

        print(f"[Step {step:02d}] | "
              f"Temps(C:{cabin_temp:.1f}, B:{battery_temp:.1f}, M:{motor_temp:.1f}) | "
              f"Rewards(C:{r_cabin:.2f}, R:{r_refrig:.2f}, Cool:{r_coolant:.2f})")

    # 3. 测试 reset 功能
    print("\n" + "=" * 60)
    print("测试 reset() 功能...")
    print("=" * 60)
    reward_calc.reset()