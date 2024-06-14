#!/usr/bin/env python3
import sys
print("THE PATH IS", sys.path)

from robobo_interface import SimulationRobobo
from learning_machines import HardwareRobobo, task1
# from learning_machines import move_till_obstacle, plot_sensor, plot_avg_sensor

if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo(1) #0 for robobo on the plain arena, 1 for robobo on the obstacle arena, 2 for robobo on the maze arena
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")


    num_episodes = 100
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    num_states = 50
    num_actions = 4

    # env = RoboboEnv(rob)
    # Q_table = train(env, num_episodes, alpha, gamma, epsilon, num_states, num_actions)
    task1(rob)
    # run_all_actions(rob)
    # move_till_obstacle(rob)
    # plot_maker(rob)
    # front_sensor_data = move_till_obstacle(rob)
    # plot_sensor(front_sensor_data)
    # plot_avg_sensor(front_sensor_data)