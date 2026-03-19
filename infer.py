import os
import pdb
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from MADDPG_Continous.MADDPG_agent import MADDPG


class Inferencer:
    def __init__(self, env, cfg, ckpt_path):
        self.env = env
        self.cfg = cfg
        self.device = cfg["device"]
        self.ckpt_path = ckpt_path

        self.agent_ids = [f"agent_{i}" for i in range(len(env.obs_groups))]
        print(env.obs_groups)

        self.dim_info = {
            aid: (cfg["obs_dims"][i], cfg["action_dims"][i])
            for i, aid in enumerate(self.agent_ids)
        }

        # ===== action bound =====
        self.action_bound = {}
        for i, aid in enumerate(self.agent_ids):
            dim = cfg["action_dims"][i]
            self.action_bound[aid] = [
                np.array([-1.0] * dim, dtype=np.float32),
                np.array([1.0] * dim, dtype=np.float32)
            ]

        self.agent = MADDPG(
            self.dim_info,
            capacity=1,
            batch_size=1,
            actor_lr=1e-4,
            critic_lr=1e-3,
            action_bound=self.action_bound,
            _chkpt_dir=None,
            _device=self.device
        )

        self._load_checkpoint()

    def _load_checkpoint(self):
        data = torch.load(self.ckpt_path, map_location=self.device)
        for aid in self.agent_ids:
            self.agent.agents[aid].actor.load_state_dict(data[aid])
        print(f"[Inferencer] Loaded {self.ckpt_path}")
        from pprint import pp
        pp(data)

    def _obs_to_dict(self, obs_list):
        return {aid: obs_list[i] for i, aid in enumerate(self.agent_ids)}

    # =========================================
    # 推理
    # =========================================
    def run(self, episodes=1, max_steps=200):
        action_series = {}
        driving_data = {
            "acc": [],
            "brake": [],
            "vel": [],
            "soc": []
        }

        for ep in range(episodes):
            obs = self.env.reset()
            obs = self._obs_to_dict(obs)

            for t in range(max_steps):
                action = self.agent.select_action(obs)

                # ===== 记录 action =====
                for aid, a in action.items():
                    for i, val in enumerate(a):
                        key = f"{aid}_dim{i}"
                        action_series.setdefault(key, []).append(val)

                # ===== step =====
                next_obs, reward, term, trunc = self.env.step(action)

                # ===== 记录 driving cycle（从 env.state）=====
                state = self.env.state

                driving_data["acc"].append(
                    state.get("driverPerformance.controlBus.driverBus._acc_pedal_travel", 0)
                )
                driving_data["brake"].append(
                    state.get("driverPerformance.controlBus.driverBus._brake_pedal_travel", 0)
                )
                driving_data["vel"].append(
                    state.get("driverPerformance.controlBus.vehicleStatus.vehicle_velocity", 0)
                )
                driving_data["soc"].append(
                    state.get("battery.controlBus.batteryBus.battery_SOC[1]", 0)
                )

                obs = self._obs_to_dict(next_obs)

                if trunc or term:
                    break

        self._plot(action_series, driving_data)

    # =========================================
    # 画图（含 driving cycle）
    # =========================================
    def _plot(self, action_series, driving_data):

        # ===== action 名 =====
        action_names = []
        for group in self.cfg["action_dict"]:
            action_names.extend(group)

        # ===== 单位接口 =====
        unit_dict = {
            "RPM_blower": "rpm",
            "RPM_comp": "rpm",
            "RPM_batt": "rpm",
            "RPM_motor": "rpm",
            "V_three": "0~1",
            "V_four": "0~1"
        }

        keys = list(action_series.keys())
        n_action = len(keys)

        # +1 for driving cycle
        fig, axes = plt.subplots(n_action + 1, 1,
                                 figsize=(10, 1.6 * (n_action + 1)),
                                 sharex=True)

        # ===== action plots =====
        for idx in range(n_action):
            ax = axes[idx]
            data = action_series[keys[idx]]

            name = action_names[idx] if idx < len(action_names) else keys[idx]
            unit = unit_dict.get(name, "")

            ax.plot(data, linewidth=1.2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            label = name if unit == "" else f"{name}\n({unit})"
            ax.set_ylabel(label, rotation=90, fontsize=9)
            ax.yaxis.set_label_coords(-0.08, 0.5)

            ax.tick_params(axis='y', labelsize=8)
            ax.grid(False)

        # =========================================
        # Driving Cycle（双y轴）
        # =========================================
        ax = axes[-1]

        t = np.arange(len(driving_data["vel"]))

        # 左轴：velocity
        ax.plot(t, driving_data["vel"], label="Velocity", linestyle="-")
        ax.set_ylabel("Velocity\n(km/h)", fontsize=9)
        ax.yaxis.set_label_coords(-0.08, 0.5)

        # 右轴：acc / brake / soc
        ax2 = ax.twinx()

        ax2.plot(t, driving_data["acc"], label="Acc")
        ax2.plot(t, driving_data["brake"], label="Brake")
        ax2.plot(t, driving_data["soc"], label="SOC")
        ax2.set_ylabel("Pedal / SOC\n(0~1)", fontsize=9)

        # 去掉多余边框
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax.grid(False)

        # 图例（合并）
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc="upper right")

        # ===== x轴 =====
        axes[-1].set_xlabel("Time step", fontsize=10)

        plt.subplots_adjust(hspace=0.15)

        save_path = "actions_with_driving_cycle.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Inferencer] Saved to {save_path}")

if __name__ == "__main__":
    from env.env_dummy import DummyEnv
    from config.config_maddpg import config
    from utils.utils_config import process_config
    from pprint import pp

    config = process_config(config)
    env = DummyEnv(config)
    # pp(config)
    inferer = Inferencer(env, config, "exps/best.pt")
    inferer.run(episodes=3)