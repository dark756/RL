import gymnasium as gym
from collections import defaultdict
import numpy as np
import time

q = None

def run(epsilon=1.0, episodes=1000):
    global q
    k=0
    training = epsilon > 0
    
    if not training:
        env = gym.make('FrozenLake-v1', is_slippery=True)#, render_mode='human')
    else:
        env = gym.make('FrozenLake-v1', is_slippery=True)
        if q is None:
            q = defaultdict(lambda: np.zeros(env.action_space.n))
            
    alpha = 0.1
    gamma = 0.99
    
    for _ in range(episodes):
        if training:
            epsilon = max(0.01, epsilon*0.999)
            
        obs, info = env.reset()
        done = False
        
        # if not training:
        #     env.render()
            
        while True:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q[obs]))
                
            next_obs, reward, te, tr, info = env.step(action)
            
            if training:
                q[obs][action] += alpha * (reward + int(not te) * np.max(q[next_obs]) * gamma - q[obs][action])
                
            obs = next_obs
            done = te or tr
            
            if done:
                if not training:
                    k+=reward
                    # time.sleep(2)
                break
                
            # if not training:
            #     time.sleep(0.3)
    print(k)
run(epsilon=1.0, episodes=15000)
# print(q)
run(epsilon=0.0, episodes=300)