import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import time

class ChaosPlanetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        random_gravity = np.random.uniform(-0.01, -11.99)
        random_wind = np.random.uniform(-30.0, 30.0)
        random_turbulence = np.random.uniform(0.0, 2.0)
        self.env.unwrapped.gravity = random_gravity
        self.env.unwrapped.wind_power = random_wind
        self.env.unwrapped.turbulence_power = random_turbulence
        return self.env.reset(**kwargs)




def train():
    base_env=gym.make('LunarLander-v3',gravity=-10, continuous=True, enable_wind=True)# , wind_power=30.0, turbulence_power=2)
    env = ChaosPlanetWrapper(base_env)
    policy_kwargs = dict(net_arch=[256, 256]) #big brain filter og: [64,64]
    model=SAC("MlpPolicy", env,policy_kwargs=policy_kwargs, verbose=1)
    # model=SAC.load("lunar_256/lunar_model_big_brain_50k.zip",env=env)
    # model.load_replay_buffer("lunar_256/lunar_buffer_bb_50k")
    model.learn(total_timesteps=1500000)
    model.save("lunar/lunar_256/lunar_master_random")
    model.save_replay_buffer("lunar/lunar_256/lunar_master_random_buffer")
    env.close()
def run(n=0):
    models=["lunar/lunar_256/lunar_master_random"]
    env=gym.make('LunarLander-v3',render_mode='human',gravity=-10, continuous=True,
               enable_wind=True, wind_power=30.0, turbulence_power=2)
    model=SAC.load(models[n])
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
run()

