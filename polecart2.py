import gymnasium as gym
from stable_baselines3 import PPO
# import gym
import numpy
import time
#obs=x,x-,theta,theta-
# training
def train():
    env=gym.make('CartPole-v1')
    model=PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save("to_file")
    env.close()
def run():
    env=gym.make('CartPole-v1',render_mode='human')
    model=PPO.load("to_file")
    episodes=10
    for _ in range(episodes):
        done=False
        obs, info=env.reset()
        env.render()
        while not done:
            action, _states=model.predict(obs, deterministic=True)
            new_obs, reward, te, tr, info=env.step(action)
            done=tr or te 
            obs=new_obs
        time.sleep(1)
# train()
run()