import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)

def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(0.2)
    print("after reset: ", rob.read_wheels())

def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())

def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())

def test_sim(rob: SimulationRobobo): 
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.get_position())

def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo): 
        rob.play_simulation() 
    # test_emotions(rob)
    # test_sensors(rob)
    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob) 

    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)

    test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

"""
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
Here is our part
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
import random
def turn_right(rob: IRobobo):
    rob.move_blocking(120, -120, 450)
    rob.sleep(0.2)

def mean_sensor_data(rob: IRobobo):
    ir_data = rob.read_irs()
    for i in range(len(ir_data)):
        if ir_data[i] > 10000 or ir_data[i] == float('inf'): # deal with inf values
            ir_data[i] = 10000
    return sum(ir_data) / len(ir_data)

def must_stop(ir_data, threshold=1800):
    front_sensors = ir_data[2:6] + [ir_data[7]]
    for i in front_sensors:
        if i >= threshold or i == float('inf'):
            return True
    return False

def move_till_obstacle(rob: IRobobo):
    front_sensor_data = {2: [], 3: [], 4: [], 5:[], 7: []}

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        print("Initial sensor data:", rob.read_irs(), "\n")
        move_counts = 0
        while True:
            ir_data = rob.read_irs()
            for i in [2, 3, 4, 5, 7]:
                if ir_data[i] > 10000 or ir_data[i] == float('inf'):
                    ir_data[i] = 10000
                front_sensor_data[i].append(ir_data[i])
            print("Sensor data:", ir_data)

            if must_stop(ir_data):
                print("Whoops, I think I see an obstacle! Here is the sensors' data: \n", ir_data)
                move_counts = 0
                turn_right(rob)
            else:
                rob.move_blocking(50, 50, 200)
                move_counts += 1
                print("move counts \n", move_counts)
            rob.sleep(0.2)
            
            if move_counts > 20: # if nothing happens for a while, stop
                print("I got boDDDred,SKATA I will stop now.")
                rob.stop_simulation()
                break
    print("results: \n", front_sensor_data)
    return front_sensor_data

def plot_sensor(front_sensor_data):
    sensor_labels = {
        2: "Left",
        3: "Right",
        4: "Centered",
        5: "Far Right",
        7: "Far Left",
    }
    plt.figure(figsize=(10, 6))
    for sensor_id, data in front_sensor_data.items():
        label = sensor_labels.get(sensor_id, f'Front Sensor {sensor_id}')
        plt.plot(data, label=label)
    plt.xlabel('Time Step')
    plt.ylabel('Sensor Value')
    plt.title('Front Sensors Data Over Time')
    plt.legend()
    plt.savefig(FIGRURES_DIR / "front_sensors_plot.png")
    plt.show()

def plot_avg_sensor(front_sensor_data):
    avg_sensor_values = [sum(values) / len(values) for values in zip(*front_sensor_data.values())]
    plt.figure(figsize=(10, 6))
    plt.plot(avg_sensor_values, label='Average Sensor Value')
    plt.xlabel('Time Step')
    plt.ylabel('Average Sensor Value')
    plt.title('Average Sensor Values Over Time')
    plt.legend()
    plt.savefig(FIGRURES_DIR / "average_sensor_values.png")
    plt.show()

