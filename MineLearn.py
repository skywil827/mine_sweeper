from stable_baselines3 import PPO
import os
import time
import wandb
from MineEnv import *
from stable_baselines3.common.callbacks import BaseCallback
from natsort import natsorted


class WandbLoggingCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        # This method will be called by the model after each call to `env.step()`.
        info = self.model.env.envs[0].get_last_game_info()
        if info is not None:
            wandb.log(info)
        return True


# Initialize callback
callback = WandbLoggingCallback()

# Initialize wandb
wandb.init(

    project="minesweeper-project",

    config={
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "timesteps": 100000,
        "env": "MineEnv"
    }
)

model_root_dir = 'models'
log_root_dir = 'logs'

current_time = str(int(time.time()))

models_dir = os.path.join(model_root_dir, current_time)
log_dir = os.path.join(log_root_dir, current_time)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = MinesweeperEnv()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device='cuda')

# Use natural sort to find the latest model.
list_folders = natsorted(os.listdir(model_root_dir), reverse=True)
for folder in list_folders:
    list_models = natsorted(os.listdir(os.path.join(model_root_dir, folder)), reverse=True)
    if len(list_models) > 0:
        model_name = os.path.join(model_root_dir, folder, list_models[0])
        print('Loading model from', model_name)
        model = PPO.load(path=model_name, env=env, device='cuda', tensorboard_log=log_dir)
        break

TIMESTEPS = 100000
iters = 0

while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=callback)

    model.save(os.path.join(models_dir, str(iters)))
    print(f"Saved model to {models_dir}/{iters}")

# [Optional] Finish the wandb run
wandb.finish()
