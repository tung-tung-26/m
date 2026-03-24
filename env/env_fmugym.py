"""
基于 fmugym 库的 FMU 环境封装，适配 MADDPG 多智能体训练。

思路：
  - 继承 FMUGym 实现一个符合 gymnasium 标准的单环境 ITMSFMUEnv
  - 再在外面套一层 MultiAgentFMUEnv，把 obs/action 按 agent 拆分
"""
from tqdm import trange
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fmugym import FMUGym
from fmugym.fmugym_config import FMUGymConfig, VarSpace, State2Out, TargetValue


# =========================================================
# 1) 继承 FMUGym，实现所有抽象方法
# =========================================================
class ITMSFMUEnv(FMUGym):
    """
    整车热管理系统 FMU 的 gymnasium 环境。
    将你的 MyITMS FMU 包装成 FMUGym 子类。
    """

    def __init__(self, config, cfg_dict):
        """
        config: FMUGymConfig 对象
        cfg_dict: 你原来的 config dict（用于读取 reward_dict 等自定义字段）
        """
        self.cfg_dict = cfg_dict
        super().__init__(config)

    # --- 必须实现的抽象方法 ---

    def _get_info(self):
        return {"time": self.time}

    def _get_obs(self):
        """
        从 FMU 读取所有 output 变量，返回为 numpy array。
        """
        self._get_fmu_output()
        obs = np.array(list(self.observation.values()), dtype=np.float32).flatten()
        noisy_obs = obs + self._get_output_noise()
        return noisy_obs

    def _get_input_noise(self):
        return np.zeros(len(self.input_dict), dtype=np.float32)

    def _get_output_noise(self):
        return np.zeros(len(self.output_dict), dtype=np.float32)

    def _get_terminated(self):
        terminated = False
        truncated = self.time >= self.stop_time
        return terminated, truncated

    def _create_action_space(self, inputs):
        lows = []
        highs = []
        for name, space in inputs.items():
            lows.append(space.low[0])
            highs.append(space.high[0])
        return spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
        )

    def _create_observation_space(self, outputs):
        lows = []
        highs = []
        for name, space in outputs.items():
            lows.append(space.low[0])
            highs.append(space.high[0])
        return spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
        )

    def _noisy_init(self):
        """
        返回 {valueReference: value} 字典，用于初始化 FMU 参数。
        """
        init_states = {}
        for var_name, (vr, config_space) in self.random_vars_refs.items():
            # 从 config_space (Box) 中采样
            val = config_space.sample()[0]
            init_states[vr] = float(val)
        return init_states

    def _process_action(self, action):
        """
        动作预处理：clip + 可选 rounding 防止小数点过多。
        """
        processed = np.clip(action, self.action_space.low, self.action_space.high)
        # 可选：对 RPM 类动作 round 到整数
        # processed = np.round(processed, decimals=1)
        return processed.tolist()

    def setpoint_trajectory(self, y_start=None, y_stop=None, time=None):
        return np.array([])

    def _process_reward(self, obs, acts, info):
        # reward 在外层 MultiAgentFMUEnv 中计算
        return 0.0

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0.0

    def get_extra_vars(self):
        """
        读取 reward 所需的额外变量（如功率）。
        """
        extra = {}
        for name, fmu_key in self.cfg_dict.get("reward_dict", {}).items():
            if fmu_key in self.output_dict:
                if self.is_fmi3:
                    extra[name] = self.fmu.getFloat64([self.output_dict[fmu_key]])[0]
                else:
                    extra[name] = self.fmu.getReal([self.output_dict[fmu_key]])[0]
        return extra


# =========================================================
# 2) 构建 FMUGymConfig 的工厂函数
# =========================================================
def build_fmugym_config(cfg: dict) -> FMUGymConfig:
    """
    从你现有的 config dict 构建 FMUGymConfig。
    """
    # --- inputs ---
    inputs = VarSpace("inputs")
    for group in cfg["action_dict"]:
        for act_name in group:
            low, high = cfg["action_bounds"][act_name]
            inputs.add_var_box(act_name, float(low), float(high))

    # --- outputs（包括 obs 变量 + reward 变量）---
    outputs = VarSpace("outputs")
    # obs 变量
    for group in cfg["obs_dict"]:
        for alias, key in group:
            if isinstance(key, str):  # FMU 变量名（不是常数）
                outputs.add_var_box(key, -1e6, 1e6)
    # reward 额外变量
    for name, fmu_key in cfg.get("reward_dict", {}).items():
        if fmu_key not in outputs.variables:
            outputs.add_var_box(fmu_key, -1e6, 1e6)

    # --- random_vars（初始化参数）---
    random_vars = VarSpace("random_vars")
    for key, val in cfg.get("env_reset_dict", {}).items():
        if isinstance(val, (list, tuple)) and len(val) == 3:
            low, high, step = val
            random_vars.add_var_box(key, float(low), float(high))
        else:
            random_vars.add_var_box(key, float(val), float(val))

    sim_step_size = cfg.get("fmu_step_size", 1)
    action_step_size = cfg.get("fmu_step_size", 1)

    return FMUGymConfig(
        fmu_path=cfg["fmu_path"],
        start_time=0.0,
        stop_time=1800.0,
        sim_step_size=sim_step_size,
        action_step_size=action_step_size,
        inputs=inputs,
        outputs=outputs,
        random_vars=random_vars,
    )


# =========================================================
# 3) 多智能体适配层（给 MADDPG trainer 用）
# =========================================================
class MultiAgentFMUEnv:
    """
    将 ITMSFMUEnv（单环境）适配为你 trainer.py 需要的多智能体接口。
    保持和原来 FMUEnv 相同的接口：reset() → obs_list, step(action_dict) → obs_list, extra, term, trunc
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # 构建 fmugym config
        fmugym_config = build_fmugym_config(cfg)
        self.inner_env = ITMSFMUEnv(fmugym_config, cfg)

        # 构建 output_name → index 映射
        self.output_names = list(self.inner_env.output_dict.keys())
        self.output_index = {name: i for i, name in enumerate(self.output_names)}

        # 构建 input_name → index 映射
        self.input_names = list(self.inner_env.input_dict.keys())

        # obs 解析（和你原来的 obs_dict 格式一致）
        self.obs_dict = cfg["obs_dict"]
        self.obs_groups = []
        self.obs_consts = []
        for group in self.obs_dict:
            keys = []
            consts = {}
            for alias, key in group:
                keys.append(alias)
                if isinstance(key, (int, float)):
                    consts[alias] = key
            self.obs_groups.append(keys)
            self.obs_consts.append(consts)

        self.max_steps = 1800

    def reset(self):
        obs_raw, info = self.inner_env.reset()
        return self._parse_obs(obs_raw)

    def step(self, action_dict):
        """
        action_dict: {"RPM_blower": 100, "RPM_comp": 1500, ...} 或按 agent 分组后的 flat dict
        """
        # 按 input_names 顺序组装 action array
        action_array = np.array(
            [float(action_dict.get(name, 0.0)) for name in self.input_names],
            dtype=np.float32
        )

        obs_raw, reward, terminated, truncated, info = self.inner_env.step(action_array)
        extra = self.inner_env.get_extra_vars()

        obs_list = self._parse_obs(obs_raw)
        return obs_list, extra, terminated, truncated

    def _parse_obs(self, obs_raw):
        """
        将 FMU 的 flat obs array 按 config 中的 obs_dict 拆分为多智能体 obs list。
        每个元素是一个 dict: {alias: value}
        """
        obs_n = []
        for group in self.obs_dict:
            obs = {}
            for alias, key in group:
                if isinstance(key, (int, float)):
                    # 常数（如 setpoint）
                    obs[alias] = key
                elif key in self.output_index:
                    obs[alias] = float(obs_raw[self.output_index[key]])
                else:
                    obs[alias] = 0.0  # fallback
            obs_n.append(obs)
        return obs_n

    def close(self):
        self.inner_env.close()
# =========================================================
# 测试入口
# =========================================================
if __name__ == "__main__":
    import sys
    import traceback
    from pprint import pp
    from config.config_maddpg import config
    from utils.utils_config import process_config

    print("=" * 70)
    print("MultiAgentFMUEnv 集成测试")
    print("=" * 70)

    # ===== 1. 处理 config =====
    config["use_i2c"] = False
    cfg = process_config(config)
    cfg["fmu_path"] = '../fmu/MyITMS-dassl1.fmu'
    print("\n✅ [1/7] config 处理完成")
    print(f"   obs_dims  = {cfg['obs_dims']}")
    print(f"   action_dims = {cfg['action_dims']}")
    print(f"   obs_dict groups = {len(cfg['obs_dict'])}")
    print(f"   action_dict groups = {len(cfg['action_dict'])}")

    # ===== 2. 构建 FMUGymConfig =====
    try:
        fmugym_cfg = build_fmugym_config(cfg)
        print("\n✅ [2/7] FMUGymConfig 构建成功")
        print(f"   fmu_path       = {fmugym_cfg.fmu_path}")
        print(f"   sim_step_size  = {fmugym_cfg.sim_step_size}")
        print(f"   action_step_size = {fmugym_cfg.action_step_size}")
        print(f"   stop_time      = {fmugym_cfg.stop_time}")
        print(f"   inputs         = {list(fmugym_cfg.inputs.variables.keys())}")
        print(f"   outputs        = {list(fmugym_cfg.outputs.variables.keys())}")
        print(f"   random_vars    = {list(fmugym_cfg.random_vars.variables.keys())}")
    except Exception as e:
        print(f"\n❌ [2/7] FMUGymConfig 构建失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ===== 3. 创建环境 =====
    try:
        env = MultiAgentFMUEnv(cfg)
        print("\n✅ [3/7] MultiAgentFMUEnv 创建成功")
        print(f"   output_names   = {env.output_names}")
        print(f"   input_names    = {env.input_names}")
        print(f"   obs_groups     = {env.obs_groups}")
        print(f"   num agents     = {len(env.obs_groups)}")
    except Exception as e:
        print(f"\n❌ [3/7] MultiAgentFMUEnv 创建失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ===== 4. 测试 reset =====
    try:
        obs_list = env.reset()
        print(f"\n✅ [4/7] reset() 成功, 返回 {len(obs_list)} 个 agent 的观测")
        for i, obs in enumerate(obs_list):
            print(f"   agent_{i}: keys={list(obs.keys())}, "
                  f"values={[f'{v:.4f}' if isinstance(v, float) else v for v in obs.values()]}")
    except Exception as e:
        print(f"\n❌ [4/7] reset() 失败: {e}")
        traceback.print_exc()
        sys.exit(1)


    test_actions = {}
    for group in cfg["action_dict"]:
        for act_name in group:
            low, high = cfg["action_bounds"][act_name]
            test_actions[act_name] = float(low + high) / 2.0  # 取中间值
    for i in trange(1800):
        obs_list, extra, term, trunc = env.step(test_actions)



    # # ===== 5. 测试 step（使用 flat action_name dict，模拟 trainer._scale_action 的输出展开后） =====
    # try:
    #     # 模拟 trainer 给出的动作：先按 agent 分组，再展开为 {action_name: value}
    #     test_actions = {}
    #     for group in cfg["action_dict"]:
    #         for act_name in group:
    #             low, high = cfg["action_bounds"][act_name]
    #             test_actions[act_name] = float(low + high) / 2.0  # 取中间值
    #
    #     print(f"\n   测试动作: {test_actions}")
    #
    #     NUM_TEST_STEPS = 5
    #     print(f"\n✅ [5/7] 开始 step 测试 ({NUM_TEST_STEPS} 步)")
    #     for i in range(NUM_TEST_STEPS):
    #         obs_list, extra, term, trunc = env.step(test_actions)
    #         total_obs_vals = sum(
    #             len([v for v in obs.values() if isinstance(v, (int, float))])
    #             for obs in obs_list
    #         )
    #         print(f"   step {i}: "
    #               f"obs_agents={len(obs_list)}, "
    #               f"total_obs_values={total_obs_vals}, "
    #               f"extra_keys={list(extra.keys())}, "
    #               f"term={term}, trunc={trunc}")
    #         # 打印第一个 agent 的观测细节
    #         if i == 0:
    #             for j, obs in enumerate(obs_list):
    #                 print(f"      agent_{j}: {obs}")
    #             if extra:
    #                 print(f"      extra: {extra}")
    # except Exception as e:
    #     print(f"\n❌ [5/7] step() 失败: {e}")
    #     traceback.print_exc()
    #     sys.exit(1)
    #
    # # ===== 6. 测试连续 reset + step（模拟 episode 切换） =====
    # try:
    #     print(f"\n✅ [6/7] 测试 reset → step → reset 循环")
    #     for ep in range(3):
    #         obs_list = env.reset()
    #         ep_obs_count = 0
    #         for t in range(3):
    #             obs_list, extra, term, trunc = env.step(test_actions)
    #             ep_obs_count += 1
    #             if term or trunc:
    #                 break
    #         print(f"   episode {ep}: steps={ep_obs_count}, "
    #               f"agent_0 obs sample={list(obs_list[0].values())[:3]}")
    # except Exception as e:
    #     print(f"\n❌ [6/7] reset-step 循环失败: {e}")
    #     traceback.print_exc()
    #     sys.exit(1)
    #
    # # ===== 7. 测试 close =====
    # try:
    #     env.close()
    #     print(f"\n✅ [7/7] close() 成功")
    # except Exception as e:
    #     print(f"\n❌ [7/7] close() 失败: {e}")
    #     traceback.print_exc()
    #     sys.exit(1)
    #
    # # ===== 8. 与 Trainer 兼容性验证（仅结构检查，不真正训练）=====
    # print("\n" + "=" * 70)
    # print("Trainer 兼容性结构检查")
    # print("=" * 70)
    #
    # # 重新创建环境，检查 Trainer 需要的属性
    # env = MultiAgentFMUEnv(cfg)
    # checks = {
    #     "env.obs_groups 存在": hasattr(env, "obs_groups"),
    #     "env.obs_dict 存在": hasattr(env, "obs_dict"),
    #     "len(obs_groups) == len(obs_dict)":
    #         len(env.obs_groups) == len(env.obs_dict),
    #     "reset() 返回 list": isinstance(env.reset(), list),
    #     "step() 返回 4 元组":
    #         len(env.step(test_actions)) == 4,
    # }
    # all_pass = True
    # for desc, result in checks.items():
    #     status = "✅" if result else "❌"
    #     if not result:
    #         all_pass = False
    #     print(f"   {status} {desc}")
    #
    # env.close()
    #
    # print("\n" + "=" * 70)
    # if all_pass:
    #     print("🎉 所有测试通过！MultiAgentFMUEnv 可以接入 Trainer 使用。")
    # else:
    #     print("⚠️ 部分检查未通过，请根据上面的 ❌ 修复。")
    # print("=" * 70)
