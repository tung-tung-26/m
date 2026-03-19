import torch
from utils.utils_misc import C_to_K

config = {
    # ===== 环境 =====
    "fmu_path": "fmu/MyITMS-dassl01.fmu",
    "fmu_step_size": 5,
    "drivecycle": "WLTC.txt",
    # ===== 设备 & 训练 =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_folder": "runs/",
    "num_episodes": 100,
    "episode_iter": 20,
    "buffer_size": 10000,
    "hidden_dim": 1024,
    "num_layers": 6,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.95,
    "tau": 1e-2,
    "batch_size": 12,
    "eval_interval":1, # 每n个episode
    "save_interval":1, # 每n个episode

    "temp_cabin_set": C_to_K(25),
    "temp_battery_set": C_to_K(30),
    "temp_motor_set": C_to_K(60),
    # ===== 观测（每个 agent 一个 list）=====
    "obs_dict": [
        [
            ("temp_cabin_set", "temp_cabin_set"),
            ("temp_cabin", "cabinVolume.summary.T"),
            ("acc", "driverPerformance.controlBus.driverBus._acc_pedal_travel"),
            ("brake", "driverPerformance.controlBus.driverBus._brake_pedal_travel"),
            ("speed", "driverPerformance.controlBus.vehicleStatus.vehicle_velocity"),
        ],
        [
            ("superheat", "superHeatingSensor.outPort"),
            ("subcool", "superCoolingSensor.outPort"),
        ],
        [
            ("temp_battery_set", "temp_battery_set"),
            ("temp_battery", "battery.Batt_top[1].T"),
            ("temp_motor_set", "temp_motor_set"),
            ("temp_motor", "machine.heatCapacitor.T"),
        ],
    ],
    # ===== 动作 =====
    "action_dict": [
        ["RPM_blower"],
        ["RPM_comp"],
        ["RPM_batt", "RPM_motor", "V_three", "V_four"],
    ],
    # ===== 奖励 =====
    "reward_dict": {"power_compressor": "TableDC.Pe",
                    "power_battery": "TableDC1.Pe",
                    "power_motor": "TableDC2.Pe",
                    "power_cabin": "TableDC3.Pe",
                    # "temp_cabin": "cabinVolume.summary.T",
                    # "temp_battery": "battery.Batt_top[1].T",
                    # "temp_motor": "machine.heatCapacitor.T",
                    },
    "reward_mapping": {
        "agent_0": {
            "type": "cabin",
            "inputs": {
                "temp_cabin": {"src": "obs", "agent": "agent_0", "key": "temp_cabin"},
                "power_cabin": {"src": "extra", "agent": "agent_0", "key": "power_cabin"}
            },
            "params": {
                "temp_cabin_set": "temp_cabin_set",
                "weights": {"temp": 0.7, "power": 0.3}
            }
        },
        "agent_1": {
            "type": "refrigerant",
            "inputs": {
                "power_compressor": {"src": "extra", "agent": "agent_1", "key": "power_compressor"}
            },
            "params": {
                "weights": {"power": 1.0}
            }
        },
        "agent_2": {
            "type": "coolant",
            "inputs": {
                "temp_battery": {"src": "obs", "agent": "agent_2", "key": "temp_battery"},
                "temp_motor": {"src": "obs", "agent": "agent_2", "key": "temp_motor"},
                "power_battery": {"src": "extra", "agent": "agent_2", "key": "power_battery"},
                "power_motor": {"src": "extra", "agent": "agent_2", "key": "power_motor"}
            },
            "params": {
                "temp_battery_set": "temp_battery_set",
                "temp_motor_set": "temp_motor_set",
                "weights": {"temp": 0.7, "power": 0.3}
            }
        }
    },
    # ===== 动作约束 =====
    "action_bounds": {
        "RPM_blower": [10, 150],
        "RPM_comp": [100, 3000],
        "RPM_batt": [100, 2000],
        "RPM_motor": [100, 2000],
        "V_three": [0, 1],
        "V_four": [0, 1],
    },

    # ==== 初始化 ====
    "env_reset_dict": {"MY_socinit": [0.1, 1.0, 0.05],
                       "T_Cabin": [C_to_K(20), C_to_K(40), 5],
                       "MY_battT0": [C_to_K(20), C_to_K(40), 5],
                       "MY_motorT0": [C_to_K(40), C_to_K(80), 5],
                       "RPM_blower": 100,
                       "RPM_comp": 1500,
                       "RPM_batt": 1000,
                       "RPM_motor": 1000,
                       "V_three": 1,
                       "V_four": 1,
                    },



    # ===== I2C =====
    "use_i2c": True,
    "lambda_temp": 10.0,
    "i2c_hidden_dim": 256,
    "prior_buffer_size": 100,
    "prior_buffer_percentile": 80,
    "message_feature_dim": 16,
    "i2c_num_layers": 6,
    "prior_lr": 1e-3,
    "prior_train_iter": 3,
    "prior_train_batch_size": 24,
    "prior_update_frequency": 5,
    # ===== 动作离散化 =====
    "action_sep_num": {
        "RPM_blower": 16,
        "RPM_comp": 31,
        "RPM_batt": 21,
        "RPM_motor": 21,
        "V_three": 11,
        "V_four": 11,
    },
}