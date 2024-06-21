import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os
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


# def test_emotions(rob: IRobobo):
#     rob.set_emotion(Emotion.HAPPY)
#     rob.talk("Hello")
#     rob.play_emotion_sound(SoundEmotion.PURR)
#     rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


# def test_move_and_wheel_reset(rob: IRobobo): 
#     rob.move_blocking(100, 100, 1000)
#     print("before reset: ", rob.read_wheels())
#     rob.reset_wheels()
#     rob.sleep(1)
#     print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    # print("Phone pan: ", rob.read_phone_pan())
    # print("Phone tilt: ", rob.read_phone_tilt())
    # print("Current acceleration: ", rob.read_accel())
    # print("Current orientation: ", rob.read_orientation())


# def test_phone_movement(rob: IRobobo):
#     rob.set_phone_pan_blocking(180, 100)
#     print("Phone pan after move to 20: ", rob.read_phone_pan())
#     rob.set_phone_tilt_blocking(90, 100)
#     print("Phone tilt after move to 50: ", rob.read_phone_tilt())


# def test_sim(rob: SimulationRobobo):
#     print(rob.get_sim_time())
#     print(rob.is_running())
#     rob.stop_simulation()
#     print(rob.get_sim_time())
#     print(rob.is_running())
#     rob.play_simulation()
#     print(rob.get_sim_time())
#     print(rob.get_position())


# def test_hardware(rob: HardwareRobobo):
#     print("Phone battery level: ", rob.read_phone_battery())
#     print("Robot battery level: ", rob.read_robot_battery())



"""
################################################
task2
################################################
"""
def adjust_pan_and_tilt(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        rob.set_phone_pan_blocking(180, 10) #make the camera used to face the front of the robot
        rob.set_phone_tilt_blocking(110, 10) #tilt the phone slightly to the ground for wider view of the arena

def get_image(rob: IRobobo):
    image = rob.get_image_front()
    image = cv2.flip(image, 0) #the image was upside down before 
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    return image

def approach_green_object(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    prev_image = get_image(rob)
    hsv_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2HSV)
    #shades of green
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_taken = "none"

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(prev_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x = x + w // 2 #center of the green object
        image_center_x = prev_image.shape[1] // 2 #center of what the camera generally sees

        #this is how each action is triggered:
        if center_x < image_center_x - 200: #if the object is to the left of the center...
            rob.move_blocking(0, 70, 700)
            action_taken = 'left'
        elif center_x > image_center_x + 200: #same for the right
            rob.move_blocking(70, 0, 700)
            action_taken = 'right'
        else: #if the object is in the center
            rob.move_blocking(100, 100, 900) 
            action_taken = 'forward'
    else: #if there is no green object in sight, randomly choose to search right or step back
        if random.choice([True, False]): 
            rob.move_blocking(50, -10, 500)
            action_taken = "search_right"
        else:
            rob.move_blocking(-50, -50, 500)
            action_taken = "step_back"

    current_image = get_image(rob)
    return action_taken, prev_image, current_image

def calculate_reward(rob, action_result, previous_image, current_image, food_counter):
    reward = 0
    if action_result in ["right", "left", "forward"]:
        if action_result == "forward" and is_closer_to_green_object(previous_image, current_image):
            reward += 10 
        elif touched_green_object(current_image):
            reward += 30 + 2**food_counter #the more food eaten, the bigger the reward
    if hit_wall(rob):
        reward -= 20
    # reward -= 1
    return reward

def is_closer_to_green_object(previous_image, current_image):
    prev_distance = find_green_object_distance(previous_image)
    curr_distance = find_green_object_distance(current_image)
    return curr_distance < prev_distance

def find_green_object_distance(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, np.array([40, 40, 40]), np.array([80, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest_contour)
        return 1 / (w * h)  #if area is bigger, distance is smaller, so we use the inverse
    return float('inf')

def touched_green_object(image):
    threshold_area = 1000  
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, np.array([40, 40, 40]), np.array([80, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest_contour)
        return w * h > threshold_area #to make sure the object is being touched using the 100 threshold number
                                        #that looks like it works well
    return False


def hit_wall(rob):
    ir_data = rob.read_irs()  # type(ir_data) => list
    front_sensor_data = {2: [], 3: [], 4: [], 5: [], 7: []} #old code from task0
    for i in [2, 3, 4, 5, 7]:
        if ir_data[i] > 10000 or ir_data[i] == float('inf'):
            ir_data[i] = 10000
        front_sensor_data[i].append(ir_data[i])
    front_ir_values = [value for sublist in front_sensor_data.values() for value in sublist] 
    action_taken, _, _ = approach_green_object(rob)
    if action_taken in ["search_right", "search_left"] and any(sensor > 1000 for sensor in front_ir_values): #
        return True
    return False  

def determine_state(image):
    # 0: left, 1: center, 2: right, 3: none
    #choose the state based on the position of the green object in the image
    position = find_green_object_position(image)
    state_mapping = {'left': 0, 'center': 1, 'right': 2, 'none': 3}
    return state_mapping[position]

def find_green_object_position(image): #based on the horizontal position of the food 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        image_center_x = image.shape[1] // 2
        if center_x < image_center_x / 3:
            return 'left'
        elif center_x > 2 * image_center_x / 3:
            return 'right'
        else:
            return 'center'
    return 'none'

def train_model(rob, episodes, steps_per_episode):
    # Q-learning parameters
    alpha = 0.1  #α is the learning rate
    gamma = 0.9  #γ is the discount factor
    epsilon = 0.4  #ε is the exploration rate, 0.4 is the initial value

    num_states = 4  #left, center, right, none
    num_actions = 5  #move_left, move_right, move_forward, search_right, step_back
    Q_table = np.zeros((num_states, num_actions))

    # if isinstance(rob, SimulationRobobo):
    #     rob.play_simulation()
    time_list = []
    for episode in range(episodes):
        adjust_pan_and_tilt(rob)
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        
        current_image = get_image(rob)
        current_state_index = determine_state(current_image)
        total_reward = 0
        reward_list = [] #only used for plotting 
        food_counter = 0


        for step in range(steps_per_episode):
            if np.random.rand() < epsilon: 
                action_index = np.random.randint(num_actions)  #chances of choosing a random action are getting lower as 
                                                                #the robot slowly learns
                # print("random action: ", action_index)
            else:
                action_index = np.argmax(Q_table[current_state_index])  #and chances of choosing the best action
                                                                        #are getting higher as robot learns
                # print("optimal action: ", action_index)

            action_result, previous_image, current_image = approach_green_object(rob) #action
            next_state_index = determine_state(current_image) #state
            reward = calculate_reward(rob, action_result, previous_image, current_image, food_counter) #reward
            if reward > 11: #getting the exact == value of reward didn't work for some reason but this works
                food_counter += 1
                if food_counter > 6: #this was SOMETIMES working (??) in my computer
                    print("All food found!")
                    print("Time needed to complete the task:", rob.get_sim_time()) 
                    time_list.append(rob.get_sim_time())
                    break

            #update q_table
            old_value = Q_table[current_state_index, action_index]
            future_optimal_value = np.max(Q_table[next_state_index])
            # print("old value: ", old_value)
            # print("future optimal value: ", future_optimal_value)
            
            #formula is: Q(s,a) = Q(s,a) + α(r + γmaxQ(s',a') - Q(s,a)) 
            new_q_value = old_value + alpha * (reward + gamma * future_optimal_value - old_value)
            # print("new q value: ", new_q_value)
            Q_table[current_state_index, action_index] = new_q_value 
            # print("Q_table: \n", Q_table)

            current_state_index = next_state_index
            total_reward += reward
            reward_list.append(total_reward)


        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        print("Q_table of whole episode: \n", Q_table)
        epsilon *= 0.97 #because 97's kids are cool and 0.99 was too slow
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
    # if isinstance(rob, SimulationRobobo):
    #     rob.stop_simulation()

    print("Training completed.")
    #save the Q_table
    Q_table_array = np.array(Q_table)
    with open(FIGRURES_DIR / "Q_table.pkl", "wb") as f: #figures dir was familiar so q-table was saved there
        pickle.dump(Q_table_array, f)
    print("Q_table saved.")
    print("time_list: ", time_list)
    return reward_list, Q_table, time_list
    


def run_robot(rob): #Q_table is the trained Q_table from the training function, for robobo to perform what it learnt
    print("Now running the run robot function")
    
    try:
        with open(FIGRURES_DIR / "Q_table.pkl", "rb") as f:
            Q_table = pickle.load(f)
        print("Q_table loaded.")
    except FileNotFoundError:
        print("Q_table not found.")
        return
    except Exception as e:
        print("An error occurred while loading the Q_table:")
        print(e)
        return
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    adjust_pan_and_tilt(rob)
    current_image = get_image(rob)
    current_state_index = determine_state(current_image)
    total_reward = 0
    food_counter = 0
    while True:
        action_index = np.argmax(Q_table[current_state_index]) #choose best action
        action_result, previous_image, current_image = approach_green_object(rob)
        next_state_index = determine_state(current_image)
        reward = calculate_reward(rob, action_result, previous_image, current_image, food_counter)
        if reward > 11: #again, the exact == value did not really work
            food_counter += 1
            print(f"Food found! Total food found: {food_counter}")
        total_reward += reward
        current_state_index = next_state_index

        if total_reward < -10 or food_counter > 6: #stop if robot doesn't perform well or if it has found all the food
                                                #but still sometimes the robot doesn't stop I dunno why
            print("Ate it all!")
            # print("Time needed to complete the task:", rob.get_sim_time(), "seconds")
            break
        
        print(f"Total Reward: {total_reward}")
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
            
        

#############Plots#################


def plot_learning_curve(reward_list):
    # reward_list, _ = train_model(rob, episodes, steps_per_episode)
    plt.plot(reward_list)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plot_path = FIGRURES_DIR / "learning_curve.png"
    plt.savefig(plot_path)
    plt.close()
    

#q-values heatmap
def plot_q_values(Q_table):
    # _, Q_table = train_model(rob, episodes, steps_per_episode)
    plt.imshow(Q_table, cmap='hot', interpolation='nearest')
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.title("Q-Value Evolution")
    plot_path = FIGRURES_DIR / "Q_values_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    
#actions' bar chart
def plot_action_distribution(Q_table):
    # _, Q_table = train_model(rob, episodes, steps_per_episode)
    action_distribution = np.sum(Q_table, axis=0)
    plt.bar(range(5), action_distribution)
    plt.xticks(range(5), ['move_left', 'move_right', 'move_forward', 'search_right', 'step_back'])
    plt.ylabel("Total Q-Value")
    plt.title("Action Distribution")
    plot_path = FIGRURES_DIR / "action_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    
    
def time_plot(times):
    plt.plot(times)
    plt.xlabel("Episodes")
    plt.ylabel("Time")
    plt.title("Time to complete the task")
    plot_path = FIGRURES_DIR / "time_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
#you only have to call plots() in controller file to get plots    
def plots(reward_list, Q_table, time_list):
    plot_learning_curve(reward_list)
    plot_q_values(Q_table)
    plot_action_distribution(Q_table)
    time_plot(time_list)
    print("Plots saved.")

    













