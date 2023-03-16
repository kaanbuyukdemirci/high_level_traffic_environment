import gymnasium
from gym.spaces import Discrete, Box

class NewToOldVer(gymnasium.Wrapper):
    """ You can pass an environment written in a newer version of gymnasium, and make it
    compatible for older versions, where:
        - 'done' instead of 'terminated' and 'truncated'
        - reset only returns 'obs', not "obs, info"
        - "def render(self, mode): ..." instead of 'self.render_mode'
    """
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_space = Discrete(self.action_space.n)
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.observation_space = Box(self.observation_space.low, self.observation_space.high, dtype=self.observation_space.dtype)
    
    def step(self, action):
        obs, reward, terminated, truncated, info  = super().step(action)
        return obs, reward, terminated or truncated, info
    
    def reset(self):
        obs, info = super().reset()
        return obs
    
    #def render(self, mode):
    #    self.render_mode = mode
    #    return super().render()