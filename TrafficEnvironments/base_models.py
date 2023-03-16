from collections.abc import Callable
from typing_extensions import Literal
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium.spaces.box import Box
import cv2 as cv


class Lane(object):
    """To represent the lanes. Mostly for rendering purposes. Doesn't really restrict the cars' movement.
    So, they are practically of infinite length.
    """
    def __init__(self, start_location:int, end_location:int, color:tuple[int,int,int]) -> None:
        self.start_location = start_location
        self.end_location = end_location
        self.color = color
    
    def draw(self, image:np.ndarray, lane_index:int, total_number_of_lanes:int, min_start_of_lanes:float, max_end_of_lanes:float) -> None:
        # x
        max_interval = max_end_of_lanes - min_start_of_lanes
        size = image.shape[1]
        x_start = int( (self.start_location - min_start_of_lanes)/max_interval * size )
        x_end = int( (self.end_location - min_start_of_lanes)/max_interval * size )
        
        # y
        size = image.shape[0]
        y1 = int( lane_index * size/total_number_of_lanes )
        y2 = int( (lane_index+1) * size/total_number_of_lanes )
        
        # draw
        thickness = 3
        cv.line(image, pt1=[x_start, y1], pt2=[x_end, y1], color=self.color, thickness=thickness)
        cv.line(image, pt1=[x_start, y2], pt2=[x_end, y2], color=self.color, thickness=thickness)

class Barrier(object):
    """Obstacles between 2 consecutive lanes. Restricts movement. Going through them means collision.
    """
    def __init__(self, start_location:int, end_location:int, left_lane_index:int, color:tuple[int,int,int]) -> None:
        # between [-1, len(lanes)-1]
        # barriers are to the right of the left lane
        
        self.start_location = start_location
        self.end_location = end_location
        self.left_lane_index = left_lane_index
        self.color = color
    
    def draw(self, image:np.ndarray, total_number_of_lanes:int, min_start_of_lanes:float, max_end_of_lanes:float) -> None:
        # x
        max_interval = max_end_of_lanes - min_start_of_lanes
        size = image.shape[1]
        x_start = int( (self.start_location - min_start_of_lanes)/max_interval * size )
        x_end = int( (self.end_location - min_start_of_lanes)/max_interval * size )
        
        # y
        size = image.shape[0]
        y = int( (self.left_lane_index+1) * size/total_number_of_lanes )
        
        # draw
        thickness = 3
        cv.line(image, pt1=[x_start, y], pt2=[x_end, y], color=self.color, thickness=thickness)

class RoadBlock(object):
    """Obstacles on lanes. Restricts movement. They are observed the same way the cars are to keep the overall observation space smaller.
    """
    def __init__(self, lane_index:int, location:float, color:tuple[int,int,int]) -> None:
        self.lane_index = lane_index
        self.location = location
        self.color = color
    
    def draw(self, image:np.ndarray, total_number_of_lanes:int, min_start_of_lanes:float, max_end_of_lanes:float) -> None:
        # assumes road blocks stays on lanes
        # x
        max_interval = max_end_of_lanes - min_start_of_lanes
        size = image.shape[1]
        x = int( (self.location - min_start_of_lanes)/max_interval * size )
        
        # y
        size = image.shape[0]
        y_start = int( (self.lane_index) * size/total_number_of_lanes )
        y_end = int( (self.lane_index+1) * size/total_number_of_lanes )
        
        # draw
        thickness = 3
        cv.line(image, pt1=[x, y_start], pt2=[x, y_end], color=self.color, thickness=thickness)
    
class Car(object):
    """ For both the agent and the other cars in the environment. Brain takes observation and outputs an action. Others are
    self explanatory. Brain is not used if the car is the agent.
    """
    def __init__(self, lane_index:int, location:float, width:float, speed:float, min_speed:float, max_speed:float,
                 min_acceleration:float, max_acceleration:float, brain:Callable, color:tuple[int,int,int],
                 finish_lane_index:int, finish_location:float) -> None:
        self.lane_index = lane_index
        self.location = location # for the midpoint
        self.width = width
        self.__speed = speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_acceleration = min_acceleration
        self.max_acceleration = max_acceleration
        self.brain = brain # not used if car is the agent.
        self.color = color
        self.finish_lane_index = finish_lane_index # where the car wants to go
        self.finish_location = finish_location # where the car wants to go
    
    @property
    def speed(self):
        return self.__speed
    @speed.setter
    def speed(self, new_speed):
        self.__speed = max(min(new_speed, self.max_speed), self.min_speed)
    
    @property
    def nose_location(self):
        return self.location+self.width/2
    
    @property
    def tail_location(self):
        return self.location-self.width/2
    
    def run(self, time_interval:float, acceleration:float, lane_index_change:int) -> None:
        acceleration = max(min(self.max_acceleration, acceleration), self.min_acceleration)
        old_speed = self.speed
        self.speed = self.speed+acceleration*time_interval
        self.lane_index += lane_index_change
        self.location = self.location + time_interval*(self.speed+old_speed)/2
        
    def draw(self, image:np.ndarray, total_number_of_lanes:int, min_start_of_lanes:float, max_end_of_lanes:float) -> None:
        # car size
        car_to_lane_size_ratio = 0.7
        
        # x
        max_interval = max_end_of_lanes - min_start_of_lanes
        size = image.shape[1]
        x = int( (self.location - min_start_of_lanes)/max_interval * size )
        x_tail = int( (self.tail_location - min_start_of_lanes)/max_interval * size )
        x_nose = int( (self.nose_location - min_start_of_lanes)/max_interval * size )
        
        # y
        size = image.shape[0]
        y_start = int( (self.lane_index) * size/total_number_of_lanes + (1-car_to_lane_size_ratio)*size/total_number_of_lanes )
        y_end = int( (self.lane_index+1) * size/total_number_of_lanes - (1-car_to_lane_size_ratio)*size/total_number_of_lanes )
        
        # draw
        thickness = -1
        cv.rectangle(image, [x_tail, y_start], [x_nose, y_end], self.color, thickness)

class BaseSpawner(object):
    """ Spawner. Used to spawn cars, lanes, road blocks, and barriers. Overwrite this to set the environment as you want.
    There is no despawning until env.reset(), because that would affect the observations in a unreal way.
    
    Spawner is meant to be overwritten by the user based on the environment they want to construct. It might be also necessary to overwrite the 
    BaseTrafficEnvironment to pass the arguments you want to pass to these methods. See:
        BaseTrafficEnvironment.run_initialize_cars          (run at every env.reset() to form the environment)
        BaseTrafficEnvironment.run_initialize_agent         (run at every env.reset() to form the environment)
        BaseTrafficEnvironment.run_initialize_lanes         (run at every env.reset() to form the environment)
        BaseTrafficEnvironment.run_initialize_barriers      (run at every env.reset() to form the environment)
        BaseTrafficEnvironment.run_initialize_roadblocks    (run at every env.reset() to form the environment)
        BaseTrafficEnvironment.run_spawn_cars               (run at every env.step() to spawn new cars)
        
    You can also overwrite BaseTrafficEnvironment.reset method to change the order these functions are called.
    """
    def __init__(self, min_acceleration:float, max_acceleration:float, min_speed:float, max_speed:float, 
                 min_car_width:float, max_car_width:float, min_lane_count:int, max_lane_count:int, min_lane_start_location:float, 
                 max_lane_end_location:float, environment_car_color:tuple[int,int,int], agent_color:tuple[int,int,int], seed=None) -> None:
        """All of these values are for determining the boundaries of the state space. So, more correctly, they are 'min_possible_...' and
        'max_possible_...'
        """
        self.min_acceleration = min_acceleration
        self.max_acceleration = max_acceleration
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_car_width = min_car_width
        self.max_car_width = max_car_width
        self.min_lane_count = min_lane_count
        self.max_lane_count = max_lane_count
        self.min_lane_start_location = min_lane_start_location
        self.max_lane_end_location = max_lane_end_location
        
        self.environment_car_color = environment_car_color
        self.agent_color = agent_color
    
    def initialize_cars(self) -> list[Car]:
        raise NotImplementedError
    
    def initialize_agent(self) -> Car:
        raise NotImplementedError
    
    def initialize_lanes(self) -> list[Lane]:
        raise NotImplementedError
    
    def initialize_barriers(self) -> list[Barrier]:
        raise NotImplementedError
    
    def initialize_roadblocks(self) -> list[RoadBlock]:
        raise NotImplementedError
    
    def spawn_cars(self) -> list[Car]:
        raise NotImplementedError

class BaseTrafficEnvironment(gym.Env):
    """Base class for the traffic environments.
    """
    def __init__(self, spawner:BaseSpawner, time_interval:float, max_observation_distance:float, max_step_count:int, 
                 env_name:str, render_resolution:tuple[int,int]=(540, 1080), canvas_color:tuple[np.uint8,np.uint8,np.uint8]=(0,0,0), 
                 render_mode:Literal['human', 'rgb_array', 'save_as_vid']='rgb_array', speed_up_rendering_by_x_times=1, 
                 default_seed:int=None, dtype=np.float32) -> None:
        super().__init__()
        # given
        self.spawner = spawner
        self.time_interval = time_interval # in seconds
        self.max_observation_distance = max_observation_distance # the max distance cars can see. from nose to tail and tail to nose
        self.max_step_count = max_step_count # the max number of steps the environment will be run.
        self.env_name = env_name
        self.render_resolution = render_resolution
        self.render_mode = render_mode
        self.speed_up_rendering_by_x_times = speed_up_rendering_by_x_times
        self.canvas_color = canvas_color
        self.dtype = dtype
        
        # from the spawner
        self.min_acceleration = spawner.min_acceleration
        self.max_acceleration = spawner.max_acceleration
        self.min_speed = spawner.min_speed
        self.max_speed = spawner.max_speed
        self.min_car_width = spawner.min_car_width
        self.max_car_width = spawner.max_car_width
        self.min_lane_count = spawner.min_lane_count
        self.max_lane_count = spawner.max_lane_count
        self.min_lane_start_location = spawner.min_lane_start_location
        self.max_lane_end_location = spawner.max_lane_end_location 
        
        # set at reset
        self.step_counter = None
        self.default_seed = default_seed
        
        # video
        if render_mode == 'save_as_vid':
            self.video_writer = cv.VideoWriter(f"{self.env_name}.mp4", cv.VideoWriter_fourcc(*'mp4v'), 
                                               1/self.time_interval*self.speed_up_rendering_by_x_times, 
                                               (self.render_resolution[1], self.render_resolution[0]), isColor=True)
        else:
            self.video_writer = None

        # set at reset by the spawner
        self.lanes = None # index 0 is the left most lane, len(self.lanes)-1 is the right most lane
        self.barriers = None
        self.road_blocks = None
        self.cars = None
        self.agent = None
        
        # spaces
        self.action_space = None
        self.observation_space = None
        self.__set_spaces()
    
    
    
    def __set_spaces(self) -> None:
        # assume:
        #   finish location is in a lane
        #   cars starts in a lane
        #   min speed is greater than 0
        # then, assume the worst.
        lows = np.array([#0, 0, 
                            self.min_speed, self.min_car_width, 
                            -self.max_observation_distance, self.min_speed-self.max_speed, -self.max_car_width, self.min_speed-self.max_speed, 
                            -self.max_observation_distance, self.min_speed-self.max_speed, -self.max_car_width, self.min_speed-self.max_speed,
                            -self.max_observation_distance, self.min_speed-self.max_speed, -self.max_car_width, self.min_speed-self.max_speed,
                            0, 0, 1-self.max_lane_count, -(self.max_speed*self.time_interval+self.max_car_width/2)], dtype=self.dtype)
        highs = np.array([#self.max_speed*self.max_step_count+self.max_lane_end_location, self.max_lane_count-0-1, 
                            self.max_speed, self.max_car_width,
                            self.max_car_width, self.max_speed-self.min_speed, self.max_observation_distance, self.max_speed-self.min_speed,
                            self.max_car_width, self.max_speed-self.min_speed, self.max_observation_distance, self.max_speed-self.min_speed,
                            self.max_car_width, self.max_speed-self.min_speed, self.max_observation_distance, self.max_speed-self.min_speed,
                            1, 1, self.max_lane_count-1, self.max_lane_end_location-self.min_lane_start_location], dtype=self.dtype)
        self.observation_space = Box(low=lows, high=highs, dtype=self.dtype)
        
        lows = np.array([self.min_acceleration, -1], dtype=self.dtype)
        highs = np.array([self.max_acceleration, +1], dtype=self.dtype)
        self.action_space = Box(low=lows, high=highs, dtype=self.dtype)
    
    def __calculate_lane_observation(self, car:Car, lane_literal:Literal['left', 'same', 'right']) -> list[float]:
        # return 
        # [back car nose relative location, 
        # back car relative speed, 
        # front car tail relative location, 
        # front car relative speed]
        # (car or road block)
        
        if lane_literal == 'right':
            lane_index = car.lane_index+1
        elif lane_literal == 'same':
            lane_index = car.lane_index
        elif lane_literal == 'left':
            lane_index = car.lane_index-1
        else:
            raise ValueError
        
        if (lane_index >= len(self.lanes)) or (lane_index < 0): # if no lane
            return [-self.max_observation_distance, 0, self.max_observation_distance, 0]
        else:
            # cars
            backcar_nose_loc = -self.max_observation_distance
            frontcar_tail_loc = +self.max_observation_distance
            backcar_speed = 0
            frontcar_speed = 0
            for other_car in self.cars:
                if other_car != car:
                    if other_car.lane_index == lane_index:
                        # back other_car
                        if (other_car.location < car.location) and (other_car.nose_location-car.tail_location > backcar_nose_loc):
                            backcar_nose_loc = other_car.nose_location - car.tail_location
                            backcar_speed = other_car.speed - car.speed
                        
                        # front car
                        elif (other_car.location >= car.location) and (other_car.tail_location-car.nose_location < frontcar_tail_loc):
                            frontcar_tail_loc = other_car.tail_location - car.nose_location
                            frontcar_speed = other_car.speed - car.speed
            
            # road_blocks
            for road_block in self.road_blocks:
                if road_block.lane_index == lane_index:
                    # back
                    if (road_block.location < car.location) and (road_block.location-car.tail_location > backcar_nose_loc):
                        backcar_nose_loc = road_block.location-car.tail_location
                        backcar_speed = 0 - car.speed
                    # front
                    elif (road_block.location >= car.location) and (road_block.location-car.nose_location < frontcar_tail_loc):
                        frontcar_tail_loc = road_block.location-car.nose_location
                        frontcar_speed = 0 - car.speed
            
            return [backcar_nose_loc, backcar_speed, frontcar_tail_loc, frontcar_speed]
    
    def __calculate_barrier_observation(self, car:Car, lane_literal:Literal['left', 'right']) -> int:
        lane_index = car.lane_index
        if lane_literal == 'left':
            left_lane_index = lane_index-1
        elif lane_literal == 'right':
            left_lane_index = lane_index
        else:
            raise ValueError
        
        bar = 0 # 0 means no barrier, 1 means there is barrier
        for barrier in self.barriers:
            if barrier.left_lane_index == left_lane_index:
                # barrier includes car
                if ((car.tail_location > barrier.start_location) and (car.tail_location < barrier.end_location) or
                    (car.nose_location > barrier.start_location) and (car.tail_location < barrier.end_location)):
                    bar = 1
                # car includes barrier
                elif ((barrier.start_location > car.tail_location) and (barrier.start_location < car.nose_location) or
                      (barrier.end_location > car.tail_location) and (barrier.end_location < car.nose_location)):
                    bar = 1
                
        return bar 
    
    def calculate_car_observation(self, car:Car) -> np.ndarray:
        """
        '0  location' # not included in the final observation. used for collision detection.
        '1  lane_index' # not included in the final observation. used for collision detection.
        '2  speed'
        '3  width'
        '4  leftlane_backcar_relative_nose_location'
        '5  leftlane_backcar_relative_speed'
        '6  leftlane_frontcar_relative_tail_location'
        '7  leftlane_frontcar_relative_speed'
        '8  samelane_backcar_relative_nose_location'
        '9  samelane_backcar_relative_speed'
        '10 samelane_frontcar_relative_tail_location'
        '11 samelane_frontcar_relative_speed'
        '12 rightlane_backcar_relative_nose_location'
        '13 rightlane_backcar_relative_speed'
        '14 rightlane_frontcar_relative_tail_location'
        '15 rightlane_frontcar_relative_speed'
        '16 leftlane_barrier'
        '17 rightlane_barrier'
        '18 finishpoint_relative_lane'
        '19 finishpoint_relative_location'
        """
        # essentials
        location = car.location
        lane_index = car.lane_index
        speed = car.speed
        width = car.width
        
        # left lane
        left_lane_obs = self.__calculate_lane_observation(car, 'left')
        # same lane
        same_lane_obs = self.__calculate_lane_observation(car, 'same')
        # right lane
        right_lane_obs = self.__calculate_lane_observation(car, 'right')
        
        # left barriers
        left_barrier_obs = self.__calculate_barrier_observation(car, 'left')
        # right barriers
        right_barrier_obs = self.__calculate_barrier_observation(car, 'right')
        
        # relative finish lane and location
        relative_finish_lane_index = car.finish_lane_index - car.lane_index
        relative_finish_location = car.finish_location - car.location
        
        # obs
        obs = np.array([location, lane_index, speed, width, *left_lane_obs, *same_lane_obs, *right_lane_obs, left_barrier_obs, right_barrier_obs, relative_finish_lane_index, relative_finish_location])
        
        return obs
    
    def calculate_action(self, car:Car, obs:np.ndarray) -> np.ndarray:
        act = car.brain(obs)
        return act
    
    def execute_action(self, car:Car, action:np.ndarray) -> None:
        if not(self.action_space.contains(action)):
            raise ValueError(f"The given action:\n{action},dtype={action.dtype}\nis not in the action space:\n{self.action_space}\nMake sure that dtype={self.action_space.dtype}.")
        else:
            acceleration, lane_index_change = action
            car.run(self.time_interval, acceleration, int(round(lane_index_change)))
    
    def __collision_check(self, old_observation:np.ndarray, new_observation:np.ndarray) -> bool:
        # checks collision with barrier, cars, and road blocks
        new_samelane_backcar_relative_nose_location = new_observation[8]
        new_samelane_frontcar_relative_tail_location = new_observation[10]
        new_lane_index = new_observation[1]
        old_lane_index = old_observation[1]
        old_leftlane_barrier = old_observation[16]
        old_rightlane_barrier = old_observation[17]
            
        # with cars and road blocks
        if (new_samelane_frontcar_relative_tail_location < 0) or (new_samelane_backcar_relative_nose_location>0):
            return True
        # with barriers
        elif new_lane_index != old_lane_index:
            if new_lane_index == old_lane_index+1: # right
                return True if old_rightlane_barrier==1 else False
            else: # left
                return True if old_leftlane_barrier==1 else False
        return False
    
    def __finish_check(self):
        if (self.agent.lane_index == self.agent.finish_lane_index) and (self.agent.tail_location < self.agent.finish_location) and (self.agent.nose_location > self.agent.finish_location):
            return True
        elif self.agent.tail_location > self.agent.finish_location:
            return True
        else:
            return False
    
    def calculate_reward(self, old_observation:np.ndarray, new_observation:np.ndarray) -> float:
        default_reward = -1
        collision_reward = self.max_step_count*default_reward
        finish_reward = -self.max_step_count*default_reward
        
        if (self.agent.lane_index == self.agent.finish_lane_index) and (self.agent.tail_location < self.agent.finish_location) and (self.agent.nose_location > self.agent.finish_location):
            print("Finish reward!") # 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
            return finish_reward
        elif self.__collision_check(old_observation, new_observation):
            return collision_reward
        else:
            return default_reward
    
    def calculate_done(self, old_observation:np.ndarray, new_observation:np.ndarray) -> tuple[bool, bool]:
        # return terminated, truncated
        if self.step_counter > self.max_step_count:
            return False, True
        elif self.__collision_check(old_observation, new_observation) or self.__finish_check():
            return True, False
        else:
            return False, False
        
    def step(self, action:np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # old observation, necessary for collision check
        old_observation = self.calculate_car_observation(self.agent)
        
        # observe (for cars)
        observations = []
        for car in self.cars:
            observations.append(self.calculate_car_observation(car)[2:])
        
        # decide on the actions and execute them (for cars)
        for index, car in enumerate(self.cars):
            car_action = self.calculate_action(car, observations[index])
            self.execute_action(car, car_action)
        
        # execute action (for agent)
        self.execute_action(self.agent, action)

        # observe (for agent)
        new_observation = self.calculate_car_observation(self.agent)
        
        # calculate if the episode is done, and do the observation for the agent while doing so.
        terminated, truncated = self.calculate_done(old_observation, new_observation)
        
        # calculate reward based on new state (observation) (for agent)
        reward = self.calculate_reward(old_observation, new_observation)
        
        # any info to return
        info = {}
        
        # spawn cars
        self.cars += self.run_spawn_cars()
        
        # increment step counter
        self.step_counter += 1
        
        return new_observation[2:], reward, terminated, truncated, info 
    
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # reset step counter
        self.step_counter = 0
        
        # reset car list
        if seed != None:
            np.random.seed(seed)
        elif self.default_seed != None:
            np.random.seed(self.default_seed)
        self.lanes = self.run_initialize_lanes()
        self.barriers = self.run_initialize_barriers()
        self.road_blocks = self.run_initialize_roadblocks()
        self.cars = self.run_initialize_cars()
        self.agent = self.run_initialize_agent()
        
        # obs and info
        obs = self.calculate_car_observation(self.agent)
        info = {}
        
        return obs[2:], info
    
    def render(self) -> None:
        # canvas
        image = np.zeros(shape=(*self.render_resolution, 3), dtype=np.uint8) + np.array(list(self.canvas_color), dtype=np.uint8).reshape(1,1,3)
        # draw
        for lane_index, lane in enumerate(self.lanes):
            lane.draw(image, lane_index, len(self.lanes), self.min_lane_start_location, self.max_lane_end_location)
        for barrier in self.barriers:
            barrier.draw(image, len(self.lanes), self.min_lane_start_location, self.max_lane_end_location)
        for road_block in self.road_blocks:
            road_block.draw(image, len(self.lanes), self.min_lane_start_location, self.max_lane_end_location)
        for car in self.cars:
            car.draw(image, len(self.lanes), self.min_lane_start_location, self.max_lane_end_location)
        self.agent.draw(image, len(self.lanes), self.min_lane_start_location, self.max_lane_end_location)
        
        if self.render_mode == 'human':
            cv.imshow(self.env_name, image)
            cv.waitKey(int(self.time_interval*1000/self.speed_up_rendering_by_x_times))
        elif self.render_mode == 'rgb_array':
            pass
        elif self.render_mode == 'save_as_vid':
            self.video_writer.write(image)
        else:
            raise ValueError(f"No such render mode as {self.render_mode}")
        
        return image

    def close(self) -> None:
        if self.video_writer != None:
                self.video_writer.release()
        cv.destroyAllWindows()
        
    
    def run_initialize_cars(self) -> list[Car]: 
        """This function is run at every reset. This function determines the cars the environment starts with.
        """
        return self.spawner.initialize_cars()
    def run_initialize_agent(self) -> Car:
        """This function is run at every reset. This function determines the agent the environment starts with.
        """
        return self.spawner.initialize_agent()
    def run_initialize_lanes(self) -> list[Lane]:
        """This function is run at every reset. This function determines the lanes the environment starts with.
        """
        return self.spawner.initialize_lanes()
    def run_initialize_barriers(self) -> list[Barrier]:
        """This function is run at every reset. This function determines the barriers the environment starts with.
        """
        return self.spawner.initialize_barriers()
    def run_initialize_roadblocks(self) -> list[RoadBlock]:
        """This function is run at every reset. This function determines the road blocks the environment starts with.
        """
        return self.spawner.initialize_roadblocks()
    def run_spawn_cars(self) -> list[Car]:
        """This function is run at the end of every step to add more cars to the environment. The list of cars this function returns is added
        to the overall car population.
        """
        return self.spawner.spawn_cars()
