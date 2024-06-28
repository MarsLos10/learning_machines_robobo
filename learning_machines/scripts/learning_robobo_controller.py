#!/usr/bin/env python3
import sys
print("THE PATH IS", sys.path)

from robobo_interface import SimulationRobobo
from learning_machines import (HardwareRobobo, train_model, plots,test_sensors, 
                               adjust_pan_and_tilt, run_robot, 
                               avg_plots, run_multiple_trainings)

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
        rob = SimulationRobobo() #0 for robobo on the plain arena, 1 for robobo on the obstacle arena, 2 for robobo on the maze arena
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")


    
    adjust_pan_and_tilt(rob) #robot performs well when it looks down so better to run this
    # test_sensors(rob) 
    reward_list, q_table, times = train_model(rob, 50, 50) #run this to train the robot
    plots(reward_list, q_table, times) #plots one training session
    all_rewards, all_times = run_multiple_trainings(rob) #multiple training sessions 
                                        #change the number of training sessions in the function found in test_actions.py
    avg_plots(all_rewards, all_times) #plots multiple training sessions

    run_robot(rob) #run this function to let robobo perform based on its knowledge from the training

 