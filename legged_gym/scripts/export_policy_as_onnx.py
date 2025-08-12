from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path, class_to_dict
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

import numpy as np
import torch
import copy

def export_policy_as_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    loaded_dict = torch.load(resume_path)
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    actor_critic = actor_critic_class(
        env_cfg.env.num_propriceptive_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    # export policy as an onnx file
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()
    class rnn_export(torch.nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor = copy.deepcopy(actor_critic.actor)
            self.is_recurrent = actor_critic.is_recurrent
            self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
            self.memory.cpu()
            self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        #self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

        def forward(self, x_in, h_in):
            x, h = self.memory(x_in.unsqueeze(0), h_in)
            x = x.squeeze(0)
            return self.actor(x), h
    
        @torch.jit.export
        def reset_memory(self):
            self.hidden_state[:] = 0.
 
        def export(self, path):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'policy_rnn_1.pt')
            self.to('cpu')
            traced_script_module = torch.jit.script(self)
            traced_script_module.save(path)

    model2 = rnn_export(actor_critic)
    model2.eval()
    model2.to("cpu")
    dummy_input = torch.randn(env_cfg.env.num_propriceptive_obs)
    dummy_input2 = torch.randn(1,512)
    input_names = ["nn_input0","nn_input1"]
    output_names = ["nn_output0","nn_output1"]

    torch.onnx.export(
        model2,
        (dummy_input,dummy_input2),
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=11,
    )
    print("Exported policy as onnx script to: ", path)


if __name__ == '__main__':
    args = get_args()
    export_policy_as_onnx(args)
