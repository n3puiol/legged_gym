from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR


import torch

from legged_gym.envs.atlas_description.atlas_description_config import AtlasDescriptionRoughCfg


class AtlasDescription(LeggedRobot):
    cfg: AtlasDescriptionRoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Load actuator network if applicable
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionally empty actuator network hidden states
        if hasattr(self, 'sea_hidden_state_per_env'):
            self.sea_hidden_state_per_env[:, env_ids] = 0.
            self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors if the network is used
        if self.cfg.control.use_actuator_network:
            self.sea_input = torch.zeros(self.num_envs * self.num_actions, 1, 2, device=self.device,
                                         requires_grad=False)
            self.sea_hidden_state = torch.zeros(2, self.num_envs * self.num_actions, 8, device=self.device,
                                                requires_grad=False)
            self.sea_cell_state = torch.zeros(2, self.num_envs * self.num_actions, 8, device=self.device,
                                              requires_grad=False)
            self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
            self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        # Choose between PD controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (
                            actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (
                self.sea_hidden_state, self.sea_cell_state))
                return torques.view(-1, self.num_actions)  # Ensure torques are the correct shape
        else:
            return super()._compute_torques(actions)
        #     # Use the default PD controller
