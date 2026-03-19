from pprint import pp
import copy
import random
import numpy as np
import torch


def _sample_from_range(val):
    if isinstance(val, (list, tuple)) and len(val) == 3:
        low, high, step = val
        low, high, step = map(float, (low, high, step))
        points = np.arange(low, high, step)
        return round(random.choice(points), 2)
    return val


def _discretize_action(bound, n):
    low, high = bound


    if isinstance(low, torch.Tensor):
        low = low.item()
    if isinstance(high, torch.Tensor):
        high = high.item()
    return list(np.linspace(low, high, n).astype(np.float32))


def process_config(config: dict) -> dict:


    # 为了不意外改动调用方的原始 dict，先深拷贝一份
    cfg = copy.deepcopy(config)

    # --------------------------------------------------------------
    # 1️⃣ 处理 obs_dict 中的 *_set 变量
    # --------------------------------------------------------------

    phys_map = {
        "T_cabin_set": cfg.get("T_cabin_set"),
        "T_bat_set": cfg.get("T_bat_set"),
        "T_motor_set": cfg.get("T_motor_set"),
    }
    for agent_idx, obs_list in enumerate(cfg.get("obs_dict", [])):
        new_obs = []
    for name in obs_list:
        if name.endswith("_set"):
            new_obs.append(phys_map.get(name, name))
    else:
        new_obs.append(name)
    cfg["obs_dict"][agent_idx] = new_obs
    # --------------------------------------------------------------
    # 2️⃣ 把 action_bounds 按 action_sep_num 拆分为离散值列表
    # --------------------------------------------------------------
    bounds = cfg.get("action_bounds", {})
    sep_nums = cfg.get("action_sep_num", {})
    discrete_map = {}
    for act_name, bound in bounds.items():
        n = sep_nums.get(act_name)
    if n is None:
        raise KeyError(f"action_sep_num 中缺少对动作 '{act_name}' 的划分数")
    discrete_map[act_name] = _discretize_action(bound, n)
    new_action_dict = []
    for group in cfg.get("action_dict", []):
        new_group = []
    for act_name in group:
        if act_name not in discrete_map:
            raise KeyError(f"在 action_bounds 中找不到动作 '{act_name}'")
    new_group.append(discrete_map[act_name])
    new_action_dict.append(new_group)
    cfg["action_dict_sep"] = new_action_dict
    # --------------------------------------------------------------
    # 3️⃣ 处理 env_reset_dict —— 随机采样或直接使用
    # --------------------------------------------------------------
    reset_dict = cfg.get("env_reset_dict", {})
    for key, val in reset_dict.items():
        cfg["env_reset_dict"][key] = _sample_from_range(val)

    # ------------------------------------------------------------------
    # 4️⃣ 计算维度信息
    # ------------------------------------------------------------------
    obs_dims = [len(agent_obs) for agent_obs in cfg.get("obs_dict", [])]
    cfg["obs_dims"] = obs_dims
    cfg["state_dims"] = obs_dims
    action_dims = [len(agent_actions) for agent_actions in cfg.get("action_dict", [])]
    cfg["action_dims"] = action_dims

    # ------------------------------------------------------------------
    # 5 计算维度信息
    # ------------------------------------------------------------------
    if not cfg['use_i2c']:
        cfg.pop('action_dict_sep')
    cfg.pop('action_sep_num')
    return cfg

if __name__ == "__main__":
    from config.config_dummy import config
    pp(process_config(config))