from stable_baselines3.common.env_checker import check_env
from MineEnv import *

env = MinesweeperEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

episodes = 50
for i in range(episodes):
    print('!'*35, 'Game', str(i), '!'*35, '\n')
    done = False
    obs = env.reset()
    while not done:
        env.render()
        random_action = env.action_space.sample()
        print('action:', random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        print('reward:', reward, '\n')
