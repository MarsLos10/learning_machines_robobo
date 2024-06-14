import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
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
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    # cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    # print("Phone pan: ", rob.read_phone_pan())
    # print("Phone tilt: ", rob.read_phone_tilt())
    # print("Current acceleration: ", rob.read_accel())
    # print("Current orientation: ", rob.read_orientation())


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

def run_all_actions(rob):

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)

    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)

    test_phone_movement(rob)


    """
    ##########################
    Here is our code for task_1:
    ##########################
    """

def task1(rob):

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

#     test_move_and_wheel_reset(rob)
#     if isinstance(rob, SimulationRobobo):
#         test_sim(rob)

#     if isinstance(rob, HardwareRobobo):
#         test_hardware(rob)

#     test_phone_movement(rob)

    ######parameter initialization#######
    num_episodes = 30 
    max_steps_per_episode = 30 

    learning_rate = 0.1 
    discount_rate = 0.99  
    exploration_rate = 1  
    max_exploration_rate = 1 #min and max ε limit epsilon's values to the desired interval
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    action_space_size = 3  # forward, Left, Right
    state_space_size = 256  # 2^8 = 256, 8 is the amount of the IR sensors on Robobo and 2 is bc we are gonna use binary encoding

    q_table = np.zeros((state_space_size, action_space_size))  #initialize Q-table

    #store the data
    episode_rewards = []  
    exploration_rates = []  
    ir_readings_over_time = []
    accel_readings_over_time = []

    def get_state(rob):
        irs = rob.read_irs()
        return int(''.join(['1' if ir and ir < 0.5 else '0' for ir in irs]), 2) #convert the ir sensor data to binary to "encode" the state

    def take_action(rob, action):
        if action == 0:  #forward
            rob.move_blocking(100, 100, 500)
        elif action == 1:  #left
            rob.move_blocking(-50, 50, 500)
        elif action == 2:  #right
            rob.move_blocking(50, -50, 500)

    def choose_action(state):
        if state & (1 << 0):  # obstacle detected in front
            return random.choice([1, 2])  #choose between left and right
        else:
            #applying the ε-greedy policy
            exploration_rate_threshold = random.uniform(0, 1) 
            if exploration_rate_threshold > exploration_rate: 
                return np.argmax(q_table[state]) #get the learned values
            else:
                return random.choice(range(action_space_size)) #randomly choose an action

    for episode in range(num_episodes):
        #initialize the environment for the new episode
        state = get_state(rob)  
        total_reward = 0  
        
        for step in range(max_steps_per_episode):
            action = choose_action(state)
            take_action(rob, action)
            new_state = get_state(rob)

            ir_data, acceleration = rob.read_irs(), rob.read_accel()
            ir_readings_over_time.append(ir_data)
            accel_readings_over_time.append(acceleration)

            if new_state == state:  #no change in state means obstacle encountered or ineffective movement
                reward = -10  #remaining in front of the same obstacle is too bad, very negative reward
            elif new_state & (1 << 0):  #obstacle detected in front
                reward = -1  #we don't want to approach the obstacle -> negative reward
            else:
                reward = 1  #positive reward for moving forward

            #save the new data in the Q-table
            q_table[state][action] = (q_table[state][action] * (1 - learning_rate) +
                                      learning_rate * (reward + discount_rate * np.max(q_table[new_state])))

            state = new_state
            total_reward += reward

        episode_rewards.append(total_reward)
        exploration_rates.append(exploration_rate)
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation() #stop the simulation after the number of episodes is reached (only way to terminate,
                             #Robobo doesn't have any goal point to reach, just wander around without hitting obstacles

    print("Training finished.\n")

    ############### Plotting #################
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')

    plt.subplot(2, 3, 2)
    plt.plot(exploration_rates)
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate')
    plt.title('Exploration Rate per Episode')

    plt.subplot(2, 3, 3)
    for ir_readings in ir_readings_over_time:
        plt.plot(ir_readings)
    plt.xlabel('Time Step')
    plt.ylabel('IR Sensor Readings')
    plt.title('IR Sensor Readings Over Time')

    plt.tight_layout()

    plot_path = FIGRURES_DIR / "training_data_plot.png"
    plt.savefig(plot_path)
    plt.close()

 
    
    
    
    