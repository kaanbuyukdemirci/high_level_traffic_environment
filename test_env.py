from TrafficEnvironments import traffic_env_1, traffic_env_2

def log(obs, reward, terminated, truncated, info, step):
    print(f"=========================== obs {step}:")
    print(f"speed:                                          {obs[0]}")
    print(f"width:                                          {obs[1]}")
    print(f"leftlane_backcar_relative_nose_location:        {obs[2]}")
    print(f"leftlane_backcar_relative_speed:                {obs[3]}")
    print(f"leftlane_frontcar_relative_tail_location:       {obs[4]}")
    print(f"leftlane_frontcar_relative_speed:               {obs[5]}")
    print(f"samelane_backcar_relative_nose_location:        {obs[6]}")
    print(f"samelane_backcar_relative_speed:                {obs[7]}")
    print(f"samelane_frontcar_relative_tail_location:       {obs[8]}")
    print(f"samelane_frontcar_relative_speed:               {obs[9]}")
    print(f"rightlane_backcar_relative_nose_location:       {obs[10]}")
    print(f"rightlane_backcar_relative_speed:               {obs[11]}")
    print(f"rightlane_frontcar_relative_tail_location:      {obs[12]}")
    print(f"rightlane_frontcar_relative_speed:              {obs[13]}")
    print(f"leftlane_barrier:                               {obs[14]}")
    print(f"rightlane_barrier:                              {obs[15]}")
    print(f"finishpoint_relative_lane:                      {obs[16]}")
    print(f"finishpoint_relative_location:                  {obs[17]}")
    print(f"obs shape:                                      {obs.shape}")
    print(f"obs dtype:                                      {obs.dtype}")
    print(f"reward:                                         {reward}")

if __name__=='__main__':
    # run
    env = traffic_env_1
    env.speed_up_rendering_by_x_times = 50
    
    obs, info = env.reset()
    for step_i in range(env.max_step_count):
        #action = traffic_env.action_space.sample()
        action = env.agent.brain(obs)
        env.render()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step_i % 100 == 0:
            log(obs, reward, terminated, truncated, info, step_i)
        if terminated or truncated:
            log(obs, reward, terminated, truncated, info, step_i)
            break
    env.close()
