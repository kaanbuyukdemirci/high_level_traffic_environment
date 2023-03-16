import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class FitObsToUnitInterval(gym.ObservationWrapper):
    """Say you have:
    low = np.array([-1, 0])
    high = np.array([2, 4])
    self.observation_space = Box(low, high)
    
    passing through this filter, you will have:
    low = np.array([-1/3, 0])
    high = np.array([2/3, 1])
    self.observation_space = Box(low, high)
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.env = env
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.difference = self.high - self.low
        
        low = self.low / self.difference
        high = self.high / self.difference
        self.observation_space = Box(low, high)
    
    def observation(self, obs):
        obs = obs/self.difference
        return obs

if __name__ == "__main__":

    class TestEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            low = np.array([-1, 0], dtype=np.float32)
            high = np.array([2, 4], dtype=np.float32)
            self.observation_space = Box(low, high, dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            obs = options
            info = {}
            return obs, info
    
    test_env = TestEnv()
    test_env = FitObsToUnitInterval(test_env)
    
    print(test_env.observation_space)    
    for i in np.linspace(start=test_env.low[0], stop=test_env.high[0], num=6):
        for j in np.linspace(start=test_env.low[1], stop=test_env.high[1], num=3):
            options = np.array([i, j], dtype=np.float32)
            print(f"Orig. obs: {options} \t Unit obs: {test_env.reset(options=options)[0]}")