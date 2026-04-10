import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pdb


def plot_multi_csv_advanced(
    csv_paths,
    var_groups,
    shared_vars_last=None,
    exp_labels=None,              # 每个CSV名字
    subplot_labels=None,          # 每个subplot的ylabel
    var_rename_dict=None,         # 普通变量重命名
    shared_var_labels=None,       # 最后subplot变量命名
    save_path="plot.png"
):
    dfs = [pd.read_csv(p) for p in csv_paths]

    n_exp = len(csv_paths)

    # ===== 默认实验名 =====
    if exp_labels is None:
        exp_labels = [f"exp_{i}" for i in range(n_exp)]

    # ===== subplot数量 =====
    n_subplots = len(var_groups)
    if shared_vars_last is not None:
        n_subplots += 1

    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=(10, 2.5 * n_subplots),
        sharex=True
    )

    if n_subplots == 1:
        axes = [axes]

    # =========================================
    # 1️⃣ 普通subplot
    # =========================================
    for i, var_list in enumerate(var_groups):
        ax = axes[i]

        for var in var_list:
            display_name = var
            if var_rename_dict and var in var_rename_dict:
                display_name = var_rename_dict[var]

            for df, exp_name in zip(dfs, exp_labels):
                if var not in df.columns:
                    continue

                data = df[var].values
                t = np.arange(len(data)) * 3.6   # 👈 在这里乘！
                ax.plot(
                    t,
                    data,
                    linewidth=1.2,
                    label=f"{exp_name}" #HACK
                )

        # ===== 风格 =====
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ylabel
        if subplot_labels and i < len(subplot_labels):
            ax.set_ylabel(subplot_labels[i], fontsize=16)
        else:
            ax.set_ylabel(", ".join(var_list), fontsize=16)

        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(False)

        ax.legend(fontsize=16, loc="best")


    # =========================================
    # 2️⃣ 最后一个subplot（双y轴）
    # =========================================
    if shared_vars_last is not None:
        ax = axes[-1]
        ax2 = ax.twinx()

        df0 = dfs[-1]
        # 👉 明确指定变量（推荐）
        vel_key = "driverPerformance.controlBus.vehicleStatus.vehicle_velocity(m/s)"
        acc_key = "driverPerformance.controlBus.driverBus._acc_pedal_travel"
        brake_key = "driverPerformance.controlBus.driverBus._brake_pedal_travel"

        for var in shared_vars_last:
            if var not in df0.columns:
                continue

            label = shared_var_labels.get(var, var) if shared_var_labels else var
            data = df0[var].values
            t = np.arange(len(data)) * 3.6   # 👈 在这里乘！

            # ===== 速度 → 右轴 =====
            if var == vel_key:
                data = data * 3.6  # m/s → km/h
                ax2.plot(
                    t,
                    data,
                    linewidth=1.5,
                    linestyle="-",
                    label=label
                )

            # ===== 踏板 → 左轴 =====
            else:
                ax.plot(
                    t,
                    data,
                    linewidth=1.5,
                    linestyle="--",
                    label=label
                )

        # ✅ 核心：锁定左轴范围（防止被压缩）
        ax.set_ylim(0, 1)

        # ===== 左轴 =====
        ax.set_ylabel("驾驶工况", fontsize=16)
        ax.yaxis.set_label_coords(-0.08, 0.5)

        # ===== 右轴 =====
        ax2.set_ylabel("（km/h）", fontsize=16)

        # ===== 风格 =====
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax.tick_params(axis='y', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

        ax.grid(False)

        # ===== legend 合并 =====
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]

        ax.legend(lines, labels, fontsize=16, loc="best")

        # =========================================
        # x轴
        # =========================================
        axes[-1].set_xlabel("时间（s）", fontsize=16)

        plt.subplots_adjust(hspace=0.15)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved to {save_path}")





plot_multi_csv_advanced(
    csv_paths=[
        "A.csv",
        "B.csv",
        "C.csv",
        "D.csv",
    ],

    var_groups = [
        ["cabinVolume.summary.T(degC)"],
        ["battery.Batt_top[1].T(degC)",],
        ["machine.heatCapacitor.T(degC)"],
        ["battery.controlBus.batteryBus.battery_SOC[1]"],
    ],

    shared_vars_last=[
        "driverPerformance.controlBus.driverBus._acc_pedal_travel", 
        "driverPerformance.controlBus.driverBus._brake_pedal_travel", 
        "driverPerformance.controlBus.vehicleStatus.vehicle_velocity(m/s)",
    ],

    # 实验名字
    exp_labels=["实验A", "实验B", "实验C", "实验D"],

    # subplot左侧标签
    subplot_labels=[
        "座舱温度（°C）",
        "电池温度（°C）",
        "电机温度（°C）",
        "SOC",
        "驾驶工况"
    ],

    # # 普通变量改名（可选）
    # var_rename_dict={
    #     "cabinVolume.summary.T(degC)": "",
    #     "battery.Batt_top[1].T(degC)": "",
    #     "machine.heatCapacitor.T(degC)": ""
    # },

    # 最后subplot变量命名
    shared_var_labels={
        "driverPerformance.controlBus.driverBus._brake_pedal_travel": "刹车踏板",
        "driverPerformance.controlBus.driverBus._acc_pedal_travel": "油门踏板",
        "driverPerformance.controlBus.vehicleStatus.vehicle_velocity(m/s)": "行车速度"
    },

    save_path="cn-abcd.png"
)