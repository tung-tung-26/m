import pdb
import numpy as np
import torch
from collections import deque


import numpy as np


class DummyEnv:
    def __init__(self, cfg):
        self.cfg = cfg

        # ===== 1. 解析 observation =====
        self.obs_dict = cfg['obs_dict']
        self.obs_groups = []   # 每个 agent 的 obs key
        self.obs_consts = []   # 每个 agent 的常数

        for group in self.obs_dict:
            keys = []
            consts = {}
            last_const = None

            for item in group:
                if isinstance(item, (int, float)):
                    last_const = item
                else:
                    if last_const is not None:
                        consts[item] = last_const
                        last_const = None
                    keys.append(item)

            self.obs_groups.append(keys)
            self.obs_consts.append(consts)

        # ===== 2. 解析 action =====
        self.action_dict = cfg['action_dict']
        self.action_bounds = cfg['action_bounds']

        self.action_keys = []
        for group in self.action_dict:
            self.action_keys.extend(group)

        # ===== 3. reward =====
        self.reward_keys = cfg['reward_dict']

        # ===== 4. reset 初始化 =====
        self.reset_dict = cfg['env_reset_dict']

        # ===== 状态 =====
        self.state = {}
        self.current_step = 0
        self.max_steps = 1000

        self._reset_state()

    # =========================
    # 初始化状态
    # =========================
    def _reset_state(self):
        self.state = {}
        self.reward = {}
        # 先初始化所有观测变量
        all_obs_keys = set()
        for group in self.obs_groups:
            all_obs_keys.update(group)

        for key in all_obs_keys:
            if key == 'cabinVolume.summary.T':
                self.state[key] = self.cfg['env_reset_dict']['T_Cabin']
            elif key == 'battery.Batt_top[1].T':
                self.state[key] = self.cfg['env_reset_dict']['MY_battT0']
            elif key == 'machine.heatCapacitor.T':
                self.state[key] = self.cfg['env_reset_dict']['MY_motorT0']
            elif key == 'driverPerformance.controlBus.vehicleStatus.vehicle_velocity':
                self.state[key] = 60.0
            else:
                self.state[key] = np.random.randn() * 10



    # =========================
    # reset
    # =========================
    def reset(self):
        self.current_step = 0
        self._reset_state()
        return self._get_obs()

    # =========================
    # step
    # =========================
    def step(self, actions):
        self.current_step += 1
        # ===== 状态演化（简单随机模拟）=====
        for key in self.state:
            self.state[key] += np.random.randn() * 0.5
        trunc = (self.current_step >= self.max_steps)
        term = False
        # reward 相关变量
        for key in self.reward_keys:
            self.reward[key] = np.random.rand() * 1000
        return self._get_obs(), self.reward, term, trunc

    # =========================
    # 构造 observation（重点）
    # =========================
    def _get_obs(self): # value_str混合key列表 -> 纯数值
        obs_n = []
        for keys, consts in zip(self.obs_groups, self.obs_consts):
            obs = []
            for k in keys:
                # 如果有常数（比如 303.15）
                if k in consts:
                    obs.append(consts[k])
                # 状态变量
                obs.append(self.state[k])
            obs_n.append(np.array(obs, dtype=np.float64))
        return obs_n

    def render(self):
        pass

    def close(self):
        pass

# 用于测试的示例代码
if __name__ == "__main__":
    # 创建测试参数
    from config.config_maddpg import config
    from utils.utils_config import process_config
    from pprint import pp
    config["use_i2c"] = False
    cfg = process_config(config)
    pp(cfg)

    # 创建环境
    env = DummyEnv(cfg)

    # 测试重置
    obs = env.reset()

    # 测试动作
    actions = {
        "RPM_blower": 150,
        "RPM_comp": 1500,
        "RPM_batt": 1500,
        "RPM_motor": 1500,
        "V_three": 1,
        "V_four": 1
    }

    for i in range(3):
        obs, reward, term, trunc = env.step(actions)
        pp(f"step {i}: reward={reward}, obs={obs}, {type(obs[0])}")