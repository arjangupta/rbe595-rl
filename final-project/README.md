# RL Research Project

## Dependencies
Create conda env from rlgpu
```
conda create --name rl-final-project --clone rlgpu
```
Make sure you already executed
```pip install -e .``` in the aeriel_gym_simulator folder
so that the imports work.

Install the bezier library
```
pip install bezier
```

Use the aerial gym fork
```
git clone git@github.com:arjangupta/aerial_gym_simulator.git
```

Modify the helpers.py file in aerial gym with the following parameters:
```
def get_args(additional_parameters=[]):
    custom_parameters = [
        {"name": "--train", "action": "store_true", "default": False, "help": "Used by main()"},
        {"name": "--num_episodes", "type": int, "default": "10", "help": "Number of episodes for training to run"},
        {"name": "--task", "type": str, "default": "tier1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": "1", "help": "Number of environments to create. Overrides config file if provided."},
    ]
```