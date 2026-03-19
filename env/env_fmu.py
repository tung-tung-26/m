import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
import pathlib


class FMUEnv:
    def __init__(self, cfg):
        self.cfg = cfg

        self.fmu_path = cfg["fmu_path"]
        self.step_size = cfg["fmu_step_size"]
        self.max_steps = 1800

        # ===== obs 解析（完全复用 DummyEnv 逻辑）=====
        self.obs_dict = cfg['obs_dict']
        self.obs_groups = []
        self.obs_consts = []

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

        # ===== action =====
        self.action_dict = cfg['action_dict']

        # ===== FMU 初始化 =====
        self._init_fmu()

        self.current_step = 0

    def _init_fmu(self):
        self.unzipdir = extract(self.fmu_path)
        self.md = read_model_description(self.fmu_path, validate=False)

        self.vrs = {}
        for var in self.md.modelVariables:
            self.vrs[var.name] = (var.valueReference, var.type)

        self.fmu = FMU2Slave(
            guid=self.md.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.md.coSimulation.modelIdentifier,
            instanceName='instance'
        )

        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()

        for k, v in self.cfg["env_reset_dict"].items():
            if k in self.vrs:
                self.fmu.setReal([self.vrs[k][0]], [float(v)])

        self.fmu.exitInitializationMode()

    def reset(self):
        try:
            self.fmu.terminate()
        except:
            pass

        self._init_fmu()
        self.current_step = 0

        return self._get_obs()

    def step(self, action_dict):
        self.current_step += 1

        # ===== 写入 action =====
        for k, v in action_dict.items():
            if k in self.vrs:
                self.fmu.setReal([self.vrs[k][0]], [float(v)])

        # ===== 前进一步 =====
        self.fmu.doStep(self.current_step * self.step_size, self.step_size)

        obs = self._get_obs()
        extra = self._get_extra_vars()
        term = False
        trunc = self.current_step >= self.max_steps

        return obs, extra, term, trunc

    def _read_var(self, name):
        vr, typ = self.vrs[name]
        return self.fmu.getReal([vr])[0]

    def _get_obs(self):
        obs_n = []
        for group in self.obs_dict:
            obs = {}
            for alias, key in group:
                if isinstance(key, (int, float)):
                    obs[alias] = key
                else:
                    obs[alias] = self._read_var(key)
            obs_n.append(obs)
        return obs_n

    def _get_extra_vars(self):
        extra = {}
        for name, fmu_key in self.cfg.get("reward_dict", {}).items():
            if fmu_key in self.vrs:
                extra[name] = self._read_var(fmu_key)
        return extra

    def close(self):
        try:
            self.fmu.terminate()
        except:
            pass