import os
import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from MADDPG_Continous.MADDPG_agent import MADDPG
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
torch.set_default_dtype(torch.float32)

from utils.utils_reward import RewardCalculator
from utils.utils_misc import *
# =========================================
# 工具函数
# =========================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================
# Trainer
# =========================================
class Trainer:
    def __init__(self, env, cfg, save_dir="exps"):
        self.env = env
        self.cfg = cfg
        self.device = cfg["device"]

        current_timestamp = datetime.now().strftime('%m-%d_%H-%M')
        save_dir = os.path.join(save_dir, current_timestamp)
        ensure_dir(save_dir)
        self.save_dir = save_dir

        # ===== agent mapping =====
        self.agent_ids = [f"agent_{i}" for i in range(len(env.obs_groups))]

        # ===== dim info =====
        dim_info = {
            agent_id: (cfg["obs_dims"][i], cfg["action_dims"][i])
            for i, agent_id in enumerate(self.agent_ids)
        }

        self.action_bound = {
            agent_id: [
                np.array([-1.0] * cfg["action_dims"][i], dtype=np.float32),
                np.array([1.0] * cfg["action_dims"][i], dtype=np.float32)
            ]
            for i, agent_id in enumerate(self.agent_ids)
        }

        self.agent = MADDPG(
            dim_info,
            cfg["buffer_size"],
            cfg["batch_size"],
            cfg["actor_lr"],
            cfg["critic_lr"],
            self.action_bound,
            _chkpt_dir=save_dir,
            _device=self.device
        )
        self.best_reward = -np.inf
        self.reward_history = []
        self._build_agent_action_bounds()
        self.reward_calc = RewardCalculator(T_cabin_set=K_to_C(self.cfg['T_cabin_set']), T_bat_set=K_to_C(self.cfg['T_bat_set']), T_motor_set=K_to_C(self.cfg['T_motor_set']))


    # ========= env adapter =========
    def _obs_to_dict(self, obs_list):
        return {
            agent_id: obs_list[i]
            for i, agent_id in enumerate(self.agent_ids)
        }

    def _reward_to_dict(self, reward_dict):
        values = list(reward_dict.values())
        return {
            agent_id: values[i % len(values)]
            for i, agent_id in enumerate(self.agent_ids)
        }

    def train(self):
        step = 0
        for ep in range(self.cfg["num_episodes"]):
            obs = self._obs_to_dict(self.env.reset())
            done = {a: False for a in self.agent_ids}

            ep_reward = 0

            for t in range(self.cfg["episode_iter"]):
                step += 1

                if step < 1000:
                    raw_action = {
                        a: np.random.uniform(-1, 1, size=self.cfg["action_dims"][i])
                        for i, a in enumerate(self.agent_ids)
                    }
                else:
                    raw_action = self.agent.select_action(obs)
                action = self._scale_action(raw_action)

                next_obs, reward, term, trunc = self.env.step(action)
                next_obs = self._obs_to_dict(next_obs)
                reward = self._reward_to_dict(reward)
                done = {a: term or trunc for a in self.agent_ids}
                # 存 raw action！！  env用缩放后的值，网络用[-1,1]
                pdb.set_trace()
                self.agent.add(obs, raw_action, reward, next_obs, done)

                if step > 1000:
                    self.agent.learn(self.cfg["batch_size"], self.cfg["gamma"])
                    self.agent.update_target(self.cfg["tau"])

                obs = next_obs
                self.reward_history.append(sum(reward.values()))

                # ep_reward += sum(reward.values())
            # self.reward_history.append(ep_reward)

            print(f"[Episode {ep}] Reward: {ep_reward:.2f}")

            # ===== 保存 =====
            if ep % self.cfg["save_interval"] == 0:
                # actor\critic\target全保存
                self.agent.save_model()
                # 只保存actor，推理用
                torch.save(
                    {aid: ag.actor.state_dict()
                     for aid, ag in self.agent.agents.items()},
                    os.path.join(self.save_dir, f"ep{ep}.pt")
                )

            # ===== best =====
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                torch.save(
                    {aid: ag.actor.state_dict()
                     for aid, ag in self.agent.agents.items()},
                    os.path.join(self.save_dir, "best.pt")
                )
                print(f"now best: {ep}")

        self._plot_rewards()

    def _plot_rewards(self, window_size=50, alpha=0.1):
        def moving_average(data, window_size):
            if len(data) < window_size:
                return np.array([])
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode='valid')
        def exponential_moving_average(data, alpha):
            ema = np.zeros_like(data, dtype=np.float32)
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
            return ema
        def set_font():
            system_platform = platform.system()
            if system_platform == "Darwin":
                font = 'Arial Unicode MS'
            elif system_platform == "Windows":
                font = 'SimHei'
            else:
                font = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
        # ===== 数据准备 =====
        rewards = np.array(self.reward_history, dtype=np.float32)
        episodes = np.arange(len(rewards))
        if len(rewards) < 2:
            return  # 数据太少不画
        set_font()
        # ===== 平滑 =====
        ma = moving_average(rewards, window_size)
        ema = exponential_moving_average(rewards, alpha)
        # ===== 画图 =====
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        # 原始数据（淡色）
        ax.plot(episodes, rewards, color='lightgray', alpha=0.4, label='Raw Reward')
        # MA（如果长度足够）
        if len(ma) > 0:
            ax.plot(episodes[window_size - 1:], ma, linewidth=2, label=f'MA({window_size})')
        # EMA
        ax.plot(episodes, ema, linewidth=2, label=f'EMA(alpha={alpha})')
        # ===== 美化 =====
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title("Training Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        # ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        # ===== 保存 =====
        save_path = os.path.join(self.save_dir, "reward_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _build_agent_action_bounds(self):
        """
        根据 action_dict + action_bounds
        构建每个 agent 的 [low, high] 向量
        """
        self.agent_action_bounds = {}
        for i, agent_id in enumerate(self.agent_ids):
            action_names = self.cfg["action_dict"][i]
            low = []
            high = []
            for name in action_names:
                l, h = self.cfg["action_bounds"][name]
                low.append(l)
                high.append(h)
            self.agent_action_bounds[agent_id] = [
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32)
            ]

    def _scale_action(self, action_dict):
        """
        把 [-1,1] 的动作缩放到真实 action space
        """
        scaled_action = {}
        for agent_id, act in action_dict.items():
            low, high = self.agent_action_bounds[agent_id]
            # clip 防止越界（很重要）
            act = np.clip(act, -1.0, 1.0)
            # 线性映射
            scaled = low + (act + 1.0) * 0.5 * (high - low)
            scaled_action[agent_id] = scaled.astype(np.float32)
        return scaled_action

if __name__ == "__main__":
    from env.env_dummy import DummyEnv
    from config.config_maddpg import config
    from utils.utils_config import process_config
    from pprint import pp
    config["use_i2c"] = False
    config = process_config(config)
    pp(config)
    env = DummyEnv(config)
    # pp(config)
    trainer = Trainer(env, config)
    trainer.train()