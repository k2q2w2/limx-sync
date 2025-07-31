from typing import Tuple
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class VecGymWrapper:
    def __init__(self, env: LeggedRobot):
        self._env = env
        self._extras = self._env.extras
        self._extras["observations"] = {}
        self._extras["observations"]["actor"] = self._env.proprioceptive_obs_buf
        self._extras["obs_history"] = {}
        self._extras["obs_history"]["actor"] = self._env.obs_history
        if self._env.privileged_obs_buf is not None:
            self._extras["observations"]["critic"] = self._env.privileged_obs_buf
            self._extras["obs_history"]["critic"] = self._env.obs_history
        self.reset()

    def get_observations(self) -> torch.Tensor:
        return self._env.proprioceptive_obs_buf

    def reset(self) -> Tuple[torch.Tensor, dict]:
        self._env.reset()
        self._extras["observations"]["actor"] = self._env.proprioceptive_obs_buf
        self._extras["obs_history"]["actor"] = self._env.obs_history
        self._extras["commands"] = self._env.commands[:, :3] * self._env.commands_scale
        if self._env.privileged_obs_buf is not None:
            self._extras["observations"]["critic"] = self._env.privileged_obs_buf
            self._extras["obs_history"]["critic"] = self._env.obs_history
        return self._env.proprioceptive_obs_buf, self._extras

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._env.step(actions)
        self._extras["observations"]["actor"] = self._env.proprioceptive_obs_buf
        self._extras["obs_history"]["actor"] = self._env.obs_history
        self._extras["commands"] = self._env.commands[:, :3] * self._env.commands_scale
        if self._env.privileged_obs_buf is not None:
            self._extras["observations"]["critic"] = self._env.privileged_obs_buf
            self._extras["obs_history"]["critic"] = self._env.obs_history
        return self._env.proprioceptive_obs_buf, self._env.privileged_obs_buf,self._env.rew_buf, self._env.reset_buf, self._env.extras

    def set_camera(self, position, lookat):
        self._env.set_camera(position=position, lookat=lookat)

    @property
    def cfg(self):
        return self._env.cfg

    @cfg.setter
    def cfg(self, value):
        self._env.cfg = value

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._env.episode_length_buf = value

    @property
    def reset_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self._env.reset_buf

    @property
    def extras(self):
        return self._extras

    @property
    def envs(self):
        return self._env.envs

    @property
    def sim(self):
        return self._env.sim

    @property
    def gym(self):
        return self._env.gym

    @property
    def viewer(self):
        return self._env.viewer

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def max_episode_length(self):
        return self._env.max_episode_length

    @property
    def num_actions(self):
        return self._env.num_actions

    @property
    def feet_indices(self):
        return self._env.feet_indices

    @property
    def num_obs(self):
        return self._env.num_obs

    @property
    def num_privileged_obs(self):
        return self._env.num_privileged_obs

    @property
    def obs_history_length(self):
        return self._env.obs_history_length

    @property
    def privileged_obs_history_length(self):
        return self._env.obs_history_length

    @property
    def dt(self):
        return self._env.dt

    @property
    def dof_pos(self):
        return self._env.dof_pos

    @property
    def dof_vel(self):
        return self._env.dof_vel

    @property
    def torques(self):
        return self._env.torques

    @property
    def commands(self):
        return self._env.commands

    @property
    def num_commands(self):
        return self._env.cfg.commands.num_commands

    @property
    def base_lin_vel(self):
        return self._env.base_lin_vel

    @property
    def base_ang_vel(self):
        return self._env.base_lin_vel

    @property
    def contact_forces(self):
        return self._env.contact_forces

    def get_privileged_observations(self):
        return self._env.get_privileged_observations()