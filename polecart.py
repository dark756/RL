import gymnasium as gym
# import gym
import numpy
import time
#obs=x,x-,theta,theta-
env=gym.make('CartPole-v1',render_mode='human')
# training
def train(env):
    #
    episodes=1#000
    for _ in range(episodes):
        done=False
        env.reset()
        env.render()
        while not done:
            #
            a,b,c,d,e=env.step(env.action_space.sample())
            print(a,b,c,d,e,sep='\n')
            done=True#terminated or truncated
train(env)