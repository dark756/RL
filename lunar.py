import gymnasium as gym
import numpy
from stable_baselines3 import SAC
import time
def train():
    env=gym.make('LunarLanderContinuous-v3')
    policy_kwargs = dict(net_arch=[256, 256]) #big brain filter og: [64,64]
    # model=SAC("MlpPolicy", env,policy_kwargs=policy_kwargs, verbose=1)
    model=SAC.load("lunar_256/lunar_model_big_brain_50k.zip",env=env)
    model.load_replay_buffer("lunar_256/lunar_buffer_bb_50k")
    model.learn(total_timesteps=600000)
    model.save("lunar_256/lunar_model_big_brain_600k")
    model.save_replay_buffer("lunar_256/lunar_buffer_bb_600k")
    env.close()
def run(n):
    models=["lunar/lunar_256/lunar_model_big_brain_50k","lunar/lunar_model_100k","lunar/lunar_256/lunar_master","lunar/lunar_256/lunar_master_random"]
    env=gym.make('LunarLander-v3',render_mode='human',gravity=-10, continuous=True,
               enable_wind=True, wind_power=20.0, turbulence_power=2)
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
run(3)
# 0 is bad because training size was too small (50k w 256 bits underfitting)
# 1 is good since training size matches policy (100k timesteps)
# 2 is great since it has large policy and training size (600k+ timesteps 1 hr + training) 256 bit policy (4x)
#3 master 1.5 mil random training