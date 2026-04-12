import gymnasium as gym
import numpy
from stable_baselines3 import PPO
import time
def train():
    env=gym.make('LunarLander-v3',continuous=False)
    # model=PPO("MlpPolicy", env, verbose=1)
    model=PPO.load("lunar_nc/lunar_model_master",env=env)
    # model.load_replay_buffer("lunar_256/lunar_buffer_bb_50k")
    model.learn(total_timesteps=300000)
    model.save("lunar_nc/lunar_model_master")
    # model.save_replay_buffer("lunar_256/lunar_nc_buffer_master")
    env.close()
def run():
    env=gym.make('LunarLander-v3',continuous=False,render_mode='human')
    model=PPO.load("lunar_nc/lunar_model_master")
    episodes=5
    for _ in range(episodes):
        done=False
        obs, info=env.reset()
        env.render()
        re=0
        while not done:
            action, _states=model.predict(obs, deterministic=True)
            new_obs, reward, te, tr, info=env.step(action)
            done=tr or te 
            obs=new_obs
            re+=reward
        time.sleep(1)
        print(reward)
# train()
run()

#360k timesteps master  