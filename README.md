# Pointfoot-Gym-Tron-Camp

This repository provides a template for training point-foot legged robots, built upon Isaac Gym and LeggedGym framework.

This code package contains the following components:

- `legged_gym`: Trainin code for robots.
- `pointfootMujoco/rl_controller.py`: Deployment code for RL policy.
- `pointfootMujoco/simulator.py`: Mujoco simulation code for the robot.

The following instructions are for the Tron-Camp participants who are not using the GradMotion platform.
If you are using the GradMotion platform, please refer to the instructions in the platform.
---

### Installation

1. Activate conda env
   - `conda activate pointfoot_legged_gym`
2. Install Isaac Gym
   - `mkdir -p ~/Tron-Camp`
   - `cd ~/Tron-Camp/isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
3. Install rsl_rl (PPO implementation)
   -  `cd ~/Tron-Camp/rsl_rl && git checkout v1.0.2 && pip install -e .` 
4. Install pointfoot-Gym
    - Clone this repository in `~/Tron-Camp`
   - `cd ~/Tron-Camp/pointfootGym && pip install -e .`

### Project Structure
1. Environment Configuration
   - `pointfoot.py`: Point-foot environment definition
   - `pointfoot_rough_config.py`: Contains two main configuration classes
     - `PointFootRoughCfg`: Environment parameters
     - `PointFootRoughCfgPPO`: Training parameters

2. Reward System
   - Each non-zero reward scale in `cfg` adds a corresponding function to the total reward calculation

3. Task Registration
   - Tasks are registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`
   - Registration can be done in `envs/__init__.py` or externally

### How to Train and Play

1. Choose Robot model: ```export ROBOT_TYPE=PF_TRON1A```
1. Train:  
  ```python legged_gym/scripts/train.py --task=pointfoot_rough```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python legged_gym/scripts/play.py --task=pointfoot_rough --load_run <run_name> --checkpoint <checkpoint>```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
3. Monitor Your Training:

During training, you can visualize various metrics like rewards and loss using TensorBoard. To launch it, run the following command in the project's root directory:

```bash
tensorboard --logdir=./
```

### Mujoco Sim-to-Sim

1. Replace the policy
   - Policies are exported to `logs/<experiment_name>/export/policies` **when you play your poicy**.
   - Replace MuJoCo policy at `ponitfootMujoco/policy/PF_TRON1A/policy/policy.onnx`

2. Environment Setup
```bash
conda create -n rl_deploy python=3.9
conda activate rl_deploy
pip install mujoco
```

3. Install Low-level Controller
```bash
cd ~/Tron-Camp/pointfootMujoco
pip install pointfoot-sdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl
```

4. Run Simulation

```bash
export ROBOT_TYPE=PF_TRON1A
python simulator.py
```

5. Run RL Controller

```bash
python rl_controller.py
```

6. Run Robot-joystick

If your policy has been successfully loaded in MuJoCo, launch the joystick to send commands.

```bash
git clone https://github.com/limxdynamics/robot-joystick.git
./robot-joystick/robot-joystick
```
