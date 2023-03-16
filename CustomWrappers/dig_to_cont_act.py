import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
if __name__ == "__main__":
    from gymnasium.spaces import Box

class DigToContAct(gym.ActionWrapper):
    """Say you have:
    low = np.array([0, 0])
    high = np.array([2, 3])
    self.action_space = Box(low, high)
    
    passing through this filter with box_counts = np.array([2, 3]), you will have:
    self.action_space = Discrete(2*3)
    where:
        Dig. act: 0      Cont. act: [0. 0.]
        Dig. act: 1      Cont. act: [0.  1.5]
        Dig. act: 2      Cont. act: [0. 3.]
        Dig. act: 3      Cont. act: [2. 0.]
        Dig. act: 4      Cont. act: [2.  1.5]
        Dig. act: 5      Cont. act: [2. 3.]
    
    """
    def __init__(self, env, box_counts:np.ndarray, dtype=np.float32):
        super().__init__(env)
        
        self.env = env
        self.dtype = dtype
        self.low = env.action_space.low
        self.high = env.action_space.high
        
        self.box_counts = box_counts
        self.total_action_count = np.prod(box_counts)
        self.action_space = Discrete(int(self.total_action_count))
        
        # some variables to remember so that the computations are faster
        self.intervals = (self.high-self.low) / (self.box_counts-1)
        self.intervals = np.nan_to_num(self.intervals) # or just pass boc count > 2
        
        # %
        self.reminder_divisors = box_counts.copy()
        for i in range(box_counts.shape[0]-2, -1, -1):
            self.reminder_divisors[i] *= self.reminder_divisors[i+1]
        
        # //
        self.quotient_divisors = np.roll(box_counts, -1)
        self.quotient_divisors[-1] = 1
        for i in range(box_counts.shape[0]-3, -1, -1):
            self.quotient_divisors[i] *= self.quotient_divisors[i+1]
    
    def action(self, act):
        digital_actions = (act % self.reminder_divisors) // self.quotient_divisors
        continuos_actions = digital_actions*self.intervals + self.low
        return continuos_actions.astype(self.dtype)

if __name__ == "__main__":

    class TestEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            low = np.array([0, -1, 1], dtype=np.float32)
            high = np.array([2, 3, 1], dtype=np.float32)
            self.action_space = Box(low, high, dtype=np.float32)
        
        def step(self, act):
            obs = None
            reward = None
            terminated = None
            truncated = None
            info = act
            return obs, reward, terminated, truncated, info 
    
    box_counts = np.array([5, 2, 1])
    
    test_env = TestEnv()
    test_env = DigToContAct(test_env, box_counts)
    
    print(test_env.action_space)
    for act in np.arange(0, test_env.total_action_count):
        print(f"Dig. act: {act} \t Cont. act: {test_env.step(act)[-1]}")