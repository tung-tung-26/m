import copy
import random
import numpy as np
import torch


# =========================================================
# 基础工具函数
# =========================================================

def _sample_from_range(val):
    """
    支持：
    - 常数：直接返回
    - [low, high, step]：随机采样
    """
    if isinstance(val, (list, tuple)) and len(val) == 3:
        low, high, step = map(float, val)
        points = np.arange(low, high, step)
        return float(random.choice(points))
    return val


def _discretize_action(bound, n):
    """
    把连续动作离散化
    """
    low, high = bound

    if isinstance(low, torch.Tensor):
        low = low.item()
    if isinstance(high, torch.Tensor):
        high = high.item()

    return list(np.linspace(low, high, n).astype(np.float32))


# =========================================================
# 🔥 新增：自动提取 setpoint（支持 temp_*_set 命名）
# =========================================================

def _extract_setpoints(cfg):
    return {
        k: v for k, v in cfg.items()
        if k.startswith("temp_") and k.endswith("_set")
    }


# =========================================================
# 🔥 obs_dict 处理（支持 (alias, key) 结构）
# =========================================================

def _process_obs_dict(cfg):

    setpoint_map = _extract_setpoints(cfg)

    new_obs_dict = []

    for group in cfg.get("obs_dict", []):
        new_group = []

        for item in group:

            # 必须是 (alias, key)
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"obs_dict 必须为 (alias, key) 结构，错误项: {item}")

            alias, key = item

            # 如果是 setpoint → 替换为常数
            if isinstance(key, str) and key in setpoint_map:
                new_group.append((alias, setpoint_map[key]))
            else:
                new_group.append((alias, key))

        new_obs_dict.append(new_group)

    cfg["obs_dict"] = new_obs_dict


# =========================================================
# 🔥 reward config 校验（非常关键）
# =========================================================

def _validate_reward_config(cfg):

    reward_cfg = cfg.get("reward", {})

    if not isinstance(reward_cfg, dict):
        raise ValueError("reward 必须是 dict")

    for agent_id, rule in reward_cfg.items():

        if "type" not in rule:
            raise KeyError(f"{agent_id} 缺少 'type'")

        if "inputs" not in rule:
            raise KeyError(f"{agent_id} 缺少 'inputs'")

        if "params" not in rule:
            raise KeyError(f"{agent_id} 缺少 'params'")

        # 校验 inputs
        for name, spec in rule["inputs"].items():

            if not isinstance(spec, dict):
                raise ValueError(f"{agent_id}.{name} 必须是 dict")

            if "src" not in spec:
                raise KeyError(f"{agent_id}.{name} 缺少 'src'")

            if "key" not in spec:
                raise KeyError(f"{agent_id}.{name} 缺少 'key'")

            if spec["src"] == "obs" and "agent" not in spec:
                raise KeyError(f"{agent_id}.{name} 缺少 'agent'")


# =========================================================
# 🔥 action 离散化处理
# =========================================================

def _process_action(cfg):

    bounds = cfg.get("action_bounds", {})
    sep_nums = cfg.get("action_sep_num", {})

    discrete_map = {}

    for act_name, bound in bounds.items():

        if act_name not in sep_nums:
            raise KeyError(f"action_sep_num 中缺少动作 {act_name}")

        discrete_map[act_name] = _discretize_action(bound, sep_nums[act_name])

    new_action_dict = []

    for group in cfg.get("action_dict", []):
        new_group = []

        for act_name in group:
            if act_name not in discrete_map:
                raise KeyError(f"{act_name} 未在 action_bounds 中定义")

            new_group.append(discrete_map[act_name])

        new_action_dict.append(new_group)

    cfg["action_dict_sep"] = new_action_dict


# =========================================================
# 🔥 reset 参数采样
# =========================================================

def _process_reset(cfg):

    reset_dict = cfg.get("env_reset_dict", {})

    for key, val in reset_dict.items():
        cfg["env_reset_dict"][key] = _sample_from_range(val)


# =========================================================
# 🔥 维度计算
# =========================================================

def _compute_dims(cfg):

    cfg["obs_dims"] = [len(group) for group in cfg.get("obs_dict", [])]
    cfg["state_dims"] = cfg["obs_dims"]
    cfg["action_dims"] = [len(group) for group in cfg.get("action_dict", [])]


# =========================================================
# 🔥 主入口
# =========================================================

def process_config(config: dict) -> dict:

    cfg = copy.deepcopy(config)

    # 1️⃣ obs 处理（语义 mapping）
    _process_obs_dict(cfg)

    # 2️⃣ action 离散化
    _process_action(cfg)

    # 3️⃣ reset 随机采样
    _process_reset(cfg)

    # 4️⃣ 维度计算
    _compute_dims(cfg)

    # 5️⃣ reward 校验（🔥关键）
    _validate_reward_config(cfg)

    # 6️⃣ 可选：关闭 I2C 时清理字段
    if not cfg.get("use_i2c", False):
        cfg.pop("action_dict_sep", None)
        cfg.pop("action_sep_num", None)
    # ===== reward 字段统一 =====
    if "reward_mapping" in cfg:
        cfg["reward"] = cfg["reward_mapping"]

    # ===== 提取 setpoint 到顶层 =====
    for group in cfg["obs_dict"]:
        for alias, val in group:
            if alias.endswith("_set") and isinstance(val, (int, float)):
                cfg[alias] = val

    return cfg


# =========================================================
# 测试入口
# =========================================================

if __name__ == "__main__":
    from config.config_maddpg import config
    from pprint import pprint

    cfg = process_config(config)
    pprint(cfg)