from .test_actions import (HardwareRobobo, SimulationRobobo,
                           train_model, plots, adjust_pan_and_tilt, test_sensors, 
                           run_robot, run_multiple_trainings, avg_plots)

__all__ = ("HardwareRobobo", "SimulationRobobo",
           train_model, plots, run_robot, adjust_pan_and_tilt, 
           test_sensors, run_multiple_trainings, avg_plots)
# __all__ = ("run_all_actions", "HardwareRobobo", "move_till_obstacle", "plot_sensor", "plot_avg_sensor")