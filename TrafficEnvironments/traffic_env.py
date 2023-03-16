import numpy as np

from .base_models import Lane, Barrier, RoadBlock, Car, BaseSpawner, BaseTrafficEnvironment

# first, subclass the BaseSpawner so that you can set rules about how the cars spawn, the environment, etc.
class Spawner1(BaseSpawner):
    def __init__(self, min_acceleration: float, max_acceleration: float, min_speed: float, max_speed: float, min_car_width: float, max_car_width: float, min_lane_count: int, max_lane_count: int, min_lane_start_location: float, max_lane_end_location: float, environment_car_color: tuple[int, int, int], agent_color: tuple[int, int, int]) -> None:
        super().__init__(min_acceleration, max_acceleration, min_speed, max_speed, min_car_width, max_car_width, min_lane_count, max_lane_count, min_lane_start_location, max_lane_end_location, environment_car_color, agent_color)
        self.exact_lane_count = None
        self.agent = None
        
        self.mean_distance_between_cars = 30
        self.std_distance_between_cars = 5
        self.mean_speed_of_cars = 5.9
        self.std_speed_of_cars = 0.5
        
        self.spawn_check_period = 2 # seconds. how often the environment will be checked to spawn new cars.
        
        # see the order the below functions are called in BaseTrafficEnvironment.reset. It can be changed as needed.
        # right now:
        # 1) initialize_lanes
        # 2) initialize_barriers
        # 3) initialize_roadblocks
        # 4) initialize_cars
        # 5) initialize_agent
        # 6) spawn_cars
    
    def initialize_lanes(self) -> list[Lane]:
        # set the lanes
        color = (255, 0, 0)
        lane0 = Lane(0, self.max_lane_end_location, color)
        lane1 = Lane(0, self.max_lane_end_location, color)
        lane2 = Lane(0, self.max_lane_end_location, color)
        lane3 = Lane(0, self.max_lane_end_location, color)
        lane4 = Lane(0, self.max_lane_end_location, color)
        lane5 = Lane(0, self.max_lane_end_location, color)
        lanes = [lane0, lane1, lane2, lane3, lane4, lane5]
        self.exact_lane_count = len(lanes)
        return lanes
    
    def initialize_barriers(self) -> list[Barrier]:
        # set the barriers
        color = (255, 255, 0)
        barrier0 = Barrier(-10000, 10000, -1, color) # keep the barrier long enough or cars will go through them after the lane ends
        barrier1 = Barrier(-10000, 10000, 5, color)
        barrier2 = Barrier(-10000, 150, 4, color)
        barriers = [barrier0, barrier1, barrier2]
        return barriers
    
    def initialize_roadblocks(self) -> list[RoadBlock]:
        # no road blocks for this environment, but an example is provided in comment.
        #color = (255, 255, 0)
        #road_block0 = RoadBlock(3, 100, color)
        #road_blocks = [road_block0]
        road_blocks = []
        return road_blocks
    
    def initialize_cars(self) -> list[Car]:
        # initialize cars to fill the whole road, and choose one of them randomly 
        # (while keeping in mind some rules) to be the agent.
        width = 4.5
        min_speed = self.min_speed
        max_speed = self.max_speed
        min_acceleration = self.min_acceleration
        max_acceleration = self.max_acceleration
        brain = lambda *args, **kwargs: np.array([0,0], dtype=np.float32)
        color = self.environment_car_color
        
        lane_index = 0
        location = np.random.normal(0, self.std_distance_between_cars)
        cars = []
        while lane_index != self.exact_lane_count:
            cars.append(Car(lane_index, location, width, np.random.normal(self.mean_speed_of_cars, self.std_speed_of_cars), 
                            min_speed, max_speed, min_acceleration, max_acceleration, brain, color, lane_index, self.max_lane_end_location))
            location += np.random.normal(self.mean_distance_between_cars, self.std_distance_between_cars)
            
            if location > self.max_lane_end_location:
                lane_index += 1
                location = np.random.normal(0, self.std_distance_between_cars)
        
        # Choose the agent randomly from set of cars. Remove it from the car list.
        random_indexes_to_try = np.arange(len(cars))
        np.random.shuffle(random_indexes_to_try)
        for index in random_indexes_to_try:
            if (cars[index].location < (3/5)*self.max_lane_end_location) and (cars[index].location >= 0):
                self.agent = cars.pop(index)
                return cars
    
    def initialize_agent(self) -> Car:
        # the agent is already chosen in self.initialize_cars, just change its color
        self.agent.color = self.agent_color
        self.agent.finish_lane_index, self.agent.finish_location = [np.random.randint(0, self.exact_lane_count), self.max_lane_end_location]
        return self.agent
    
    def spawn_cars(self, cars:list[Car], step_counter, time_interval) -> list[Car]:
        # spawn cars if the other cars are far enough from the start. Check this every self.spawn_check_period seconds.
        spawned_cars = []
        if (step_counter*time_interval) % self.spawn_check_period <= time_interval:
            width = 4.5
            min_speed = self.min_speed
            max_speed = self.max_speed
            min_acceleration = self.min_acceleration
            max_acceleration = self.max_acceleration
            brain = lambda *args, **kwargs: np.array([0,0], dtype=np.float32)
            color = self.environment_car_color
            
            for lane_index in range(0, self.exact_lane_count):
                min_location = 500.0
                for car in cars:
                    if (car.lane_index == lane_index):
                        min_location = min(car.location, min_location)
                if min_location > self.mean_distance_between_cars:
                    spawned_cars.append(Car(lane_index, np.random.normal(0, self.std_distance_between_cars), 
                                            width, np.random.normal(self.mean_speed_of_cars, self.std_speed_of_cars), 
                                            min_speed, max_speed, min_acceleration, max_acceleration, brain, color, 
                                            lane_index, self.max_lane_end_location))
                    
        return spawned_cars

class Spawner2(Spawner1):
    def initialize_lanes(self) -> list[Lane]:
        # set the lanes
        color = (255, 0, 0)
        lane0 = Lane(0, self.max_lane_end_location, color)
        lane1 = Lane(0, self.max_lane_end_location, color)
        lanes = [lane0, lane1]
        self.exact_lane_count = len(lanes)
        return lanes
    
    def initialize_barriers(self) -> list[Barrier]:
        # set the barriers
        color = (255, 255, 0)
        barrier0 = Barrier(-10000, 10000, -1, color) # keep the barrier long enough or cars will go through them after the lane ends
        barrier1 = Barrier(-10000, 10000, 1, color)
        barrier2 = Barrier(-10000, 150, 0, color)
        barriers = [barrier0, barrier1, barrier2]
        return barriers

# overwrite some run functions so that you can pass any info you want to the car spawner
class TrafficEnvironment(BaseTrafficEnvironment):
    def run_spawn_cars(self) -> list[Car]:
        return self.spawner.spawn_cars(self.cars, self.step_counter, self.time_interval)

# traffic_env_1
# Why do we need these variables?
min_acceleration = -5.0 # for the action-space boundaries
max_acceleration = +5.0 # for the action-space boundaries
min_speed = 0.0 # for the observation-space boundaries and the cars in the environment
max_speed = 10.0 # for the observation-space boundaries and the cars in the environment
min_car_width = 1 # for the observation-space boundaries
max_car_width = 30 # for the observation-space boundaries
min_lane_count = 5 # for the observation-space boundaries
max_lane_count = 5 # for the observation-space boundaries
min_lane_start_location = 0 # for the observation-space boundaries
max_lane_end_location = 300 # for the observation-space boundaries, the rendering, the finish line, and the cars in the environment
environment_car_color = (255, 255, 255)
agent_color = (0, 0, 255)
spawner = Spawner1(min_acceleration, max_acceleration, min_speed, max_speed, min_car_width, max_car_width, 
                   min_lane_count, max_lane_count, min_lane_start_location, max_lane_end_location, environment_car_color, agent_color)

time_interval = 0.1
max_observation_distance = 500
max_step_count = 1000
env_name = 'traffic_env_1'
render_resolution = (250, 1600)
canvas_color = (0,0,0)
render_mode = 'save_as_vid'
traffic_env_1 = TrafficEnvironment(spawner, time_interval, max_observation_distance, max_step_count, env_name, 
                                   render_resolution, canvas_color, render_mode, speed_up_rendering_by_x_times=10)

# traffic_env_2
# Why do we need these variables?
min_acceleration = -5.0 # for the action-space boundaries
max_acceleration = +5.0 # for the action-space boundaries
min_speed = 0.0 # for the observation-space boundaries and the cars in the environment
max_speed = 20.0 # for the observation-space boundaries and the cars in the environment
min_car_width = 1 # for the observation-space boundaries
max_car_width = 30 # for the observation-space boundaries
min_lane_count = 2 # for the observation-space boundaries
max_lane_count = 2 # for the observation-space boundaries
min_lane_start_location = 0 # for the observation-space boundaries
max_lane_end_location = 300 # for the observation-space boundaries, the rendering, the finish line, and the cars in the environment
environment_car_color = (255, 255, 255)
agent_color = (0, 0, 255)
spawner = Spawner2(min_acceleration, max_acceleration, min_speed, max_speed, min_car_width, max_car_width, 
                   min_lane_count, max_lane_count, min_lane_start_location, max_lane_end_location, environment_car_color, agent_color)

time_interval = 0.1
max_observation_distance = 500
max_step_count = 500
env_name = 'traffic_env_2'
render_resolution = (int(250/3), 1600)
canvas_color = (0,0,0)
render_mode = 'human'
traffic_env_2 = TrafficEnvironment(spawner, time_interval, max_observation_distance, max_step_count, env_name, 
                                   render_resolution, canvas_color, render_mode)