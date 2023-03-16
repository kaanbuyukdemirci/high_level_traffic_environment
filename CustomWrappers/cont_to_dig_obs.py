import gymnasium as gym
import numpy as np
if __name__ == "__main__":
    from gymnasium.spaces import Discrete, Box

class ContToDigObs(gym.ObservationWrapper):
    """Say you have:
    low = np.array([0, 0])
    high = np.array([2, 3])
    self.observation_space = Box(low, high)
    
    passing through this filter with box_counts = np.array([2, 3]), you will have:
    self.observation_space = Box(low, high)
    but, for the first dimension:
        * 0 to 1 will be mapped to 0
        * 1 to 2 will be mapped to 2
    and, for the second dimension:
        * 0    to 0.75 will be mapped to 0
        * 0.75 to 2.25 will be mapped to 1.5
        * 2.25 to 3    will be mapped to 3
    for example:
        Cont. obs: [0. 0.]       Disc. ovs: [0. 0.]
        Cont. obs: [0.  1.2]     Disc. ovs: [0.  1.5]
        Cont. obs: [0.  2.4]     Disc. ovs: [0. 3.]
        Cont. obs: [1.2 0. ]     Disc. ovs: [2. 0.]
        Cont. obs: [1.2 1.2]     Disc. ovs: [2.  1.5]
        Cont. obs: [1.2 2.4]     Disc. ovs: [2. 3.]
    """
    def __init__(self, env, box_counts:np.ndarray):
        super().__init__(env)
        if np.any(box_counts<2): raise ValueError(f"Minimum box count must be 2. Given box counts:{box_counts}")
        
        self.env = env
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        
        self.box_counts = box_counts
        self.observation_space = env.observation_space
        
        # some variables to remember so that the computations are faster
        self.intervals = (self.high-self.low) / (self.box_counts-1)
        self.margin = self.low - self.intervals/2
    
    def observation(self, obs):
        orders = (obs-self.margin) // self.intervals # centers the approximations
        approximations = orders * self.intervals + self.low
        return approximations

if __name__ == "__main__":

    class TestEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            low = np.array([-5, 8], dtype=np.float32)
            high = np.array([-3, 19], dtype=np.float32)
            self.observation_space = Box(low, high, dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            obs = options
            info = {}
            return obs, info
    
    box_counts = np.array([3, 2])
    
    test_env = TestEnv()
    test_env = ContToDigObs(test_env, box_counts)
    
    print(test_env.observation_space)    
    for i in np.linspace(start=test_env.low[0], stop=test_env.high[0], num=6):
        for j in np.linspace(start=test_env.low[1], stop=test_env.high[1], num=3):
            options = np.array([i, j], dtype=np.float32)
            print(f"Cont. obs: {options} \t Dig. obs: {test_env.reset(options=options)[0]}")