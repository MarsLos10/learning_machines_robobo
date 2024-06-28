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

robobo_sim = SimulationRobobo()
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
    cv2.imwrite(str(FIGRURES_DIR / "testphoto.png"), image)
    print("test image saved.")

"""
################################################
task3 with 2nd attempt of training
################################################
"""
    
def get_image(rob: IRobobo):
    image = rob.get_image_front()
    # image = cv2.flip(image, 0) #the image was upside down before 
    cv2.imwrite(str(FIGRURES_DIR / "get_photo.png"), image)
    # print("Image saved.")
    return image

def adjust_pan_and_tilt(rob: IRobobo): #from task2
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        rob.set_phone_pan_blocking(180, 100) #make the camera face the front of the robot
        rob.set_phone_tilt_blocking(109, 100) # 109 is the maximum tilt angle but we wanted more
        get_image(rob)
        
#Robobo checks if it sees red in front of it:
def check_red(rob: IRobobo): #returns boolean
    image = get_image(rob)
    #apparently red is split in the two ends of the HSV color space 
    #so to catch most shades we need the left and right mask and combine them
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
    if contours: 
        print("Red detected.")
        return True
    return False


def check_green(rob: IRobobo): #returns boolean and green pixel count
    image = get_image(rob) #from task1
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        green_pixel_count = cv2.countNonZero(mask)
        #we count the green pixels to use them in the reward function
        return True, green_pixel_count
    return False, 0

def determine_state(rob: IRobobo): 
    sees_green, _ = check_green(rob)
    if check_red(rob) and sees_green:
        return 0 #sees green and holds food
    elif check_red(rob) and not sees_green:
        return 1 #sees food but is not near green
    elif not check_red(rob) and sees_green:
        return 2 #bad state, close to goal with no food
    else:
        return 3 #no food, no green


def calculate_reward(rob: IRobobo, current_image):
    #trying to keep reward function simple 
    reward = 0
    if not check_red(rob):
        reward += 0
    print("No green or red in sight: ", reward)
    # if check_red(rob):
    #     reward = 30
        # print("Red detected.")
    green_boolean, green_pixels = check_green(rob)
    if green_boolean:
        reward += (green_pixels*(10**4))/(current_image.shape[0]*current_image.shape[1])
        print("Green detected.")
        # reward -= robobo_sim._base_food_distance() #doesn't work
        # print(f"Will subtract {robobo_sim._base_food_distance()} distance from the reward.")

    if robobo_sim.base_detects_food():#if goal is reached
        print(f"Will subtract {rob.get_sim_time()} secs from the reward.")
        reward += 500 #bravo
        reward -= rob.get_sim_time() #the faster the robot completes the task, the bigger the reward
    # else: #looks like this doesn't help at the end
    #     print("Goal not reached. Penalty -70secs added.")
    #     reward -= 70 #it takes around 60secs to complete the 
    #                 #if the goal is not achieved.
    print("Reward: ",reward, "\n")
    return reward


################actions########################
action_list = ["move_left", "move_right", "move_forward"]
def move_forward(rob: IRobobo):
    rob.move_blocking(100, 100, 500)
    print("Moving forward")
def move_left(rob: IRobobo):
    rob.move_blocking(0, 100, 400)
    print("Moving left") 
def move_right(rob: IRobobo):
    print("Moving right")
    rob.move_blocking(100, 0, 400)
    
def take_action(rob: IRobobo, action):
    if action == 'move_left':
        move_left(rob)
    elif action == 'move_right':
        move_right(rob)
    elif action == 'move_forward':
        move_forward(rob)
    else:
        print("Invalid action")
        rob.stop_simulation()


def approach_green_area(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    prev_image = get_image(rob)
    hsv_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_taken = random.choice(action_list) #randomly choose an action
    return action_taken, prev_image, get_image(rob)
        
        
def train_model(rob: IRobobo, episodes, steps_per_episode):
    #q-learning parameters
    alpha = 0.1  #α is the learning rate
    gamma = 0.9  #γ is the discount factor
    epsilon = 0.4  #ε is the exploration rate, 0.4 is the initial value
    num_states = 4  
    num_actions = len(action_list)  
    Q_table = np.zeros((num_states, num_actions))

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    time_list = [] 

    for episode in range(episodes):
        reward_list = [] #only used for plotting 
        adjust_pan_and_tilt(rob)
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        
        rob.move_blocking(100, 100, 1000) #start
        
        current_image = get_image(rob)
        current_state_index = determine_state(rob)
        total_reward = 0

        for step in range(steps_per_episode):
            if np.random.rand() < epsilon: 
                print("Random action")
                action_index = np.random.randint(num_actions)  
                #Chances of choosing a random action are getting lower as 
                #as the robot slowly learns
                #and chances of choosing the best action...
            else:
                print("Optimal action")
                action_index = np.argmax(Q_table[current_state_index])  
                #...are getting higher as robot learns.

            action_taken = take_action(rob, action_list[action_index]) #action
            next_state_index = determine_state(rob) #state
            reward = calculate_reward(rob, current_image) #reward
            if robobo_sim.base_detects_food():
                # print("we are in the if statement")
                break
            #update q_table
            old_value = Q_table[current_state_index, action_index]
            future_optimal_value = np.max(Q_table[next_state_index])
            new_q_value = old_value + alpha * (reward + gamma * future_optimal_value - old_value)
            Q_table[current_state_index, action_index] = new_q_value 
            current_state_index = next_state_index
            total_reward += reward
            reward_list.append(total_reward)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        print("Q_table of whole episode: \n", Q_table)
        epsilon *= 0.95
        
        if isinstance(rob, SimulationRobobo):
            time_list.append(rob.get_sim_time())
            print("Time needed to finish episode:", rob.get_sim_time())
            rob.stop_simulation()
    print("Training completed.")
        #save the Q_table in pickle file
    Q_table_array = np.array(Q_table)
    with open(FIGRURES_DIR / "Q_table3.pkl", "wb") as f:
        pickle.dump(Q_table_array, f)
    print("Q_table saved.")
    print("time_list: ", time_list)
    print("reward_list: ", reward_list)
    return reward_list, Q_table, time_list

##################MULTIPLE RUNS####################
def fixate_len_of_lists(list_of_lists): #because some episodes terminate sooner so lengths are different
    max_length = max(len(l) for l in list_of_lists)
    return [l + [0] * (max_length - len(l)) for l in list_of_lists] #remaining values are filled with 0s

def run_multiple_trainings(rob: IRobobo, num_runs=3):
    all_rewards = []
    all_timelines = []
    for i in range(num_runs):
        print(f"Now running the run multiple trainings function, number {i}")
        rewards, _, times = train_model(rob, 50, 50) 
        all_rewards.append(rewards)
        all_timelines.append(times)
        
    fixed_rewards = fixate_len_of_lists(all_rewards)
    fixed_timelines = fixate_len_of_lists(all_timelines)
    return np.array(fixed_rewards), np.array(fixed_timelines)

def run_robot(rob): #Q_table is the trained Q_table for robobo to perform
    print("Now running the run robot function")
    #reload the Q_table
    try:
        with open(FIGRURES_DIR / "Q_table3.pkl", "rb") as f: #for some reason this doesn't work(?)
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
    current_state_index = determine_state(rob)
    total_reward = 0
    while True:
        take_action = np.argmax(Q_table[current_state_index])
        action_result, previous_image, current_image = approach_green_area(rob)
        next_state_index = determine_state(rob)
        reward = calculate_reward(rob, current_image)

        total_reward += reward
        current_state_index = next_state_index

        print("Time needed to complete the task:", rob.get_sim_time(), "seconds")
            
        print(f"Total Reward: {total_reward}")
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
            
#########################PLOTS####################

#learning curve 
def plot_learning_curve(reward_list):
    # reward_list, _ = train_model(rob, episodes, steps_per_episode)
    plt.plot(reward_list)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plot_path = FIGRURES_DIR / "3_learning_curve.png"
    plt.savefig(plot_path)
    plt.close()
 
#time plot    
def time_plot(times):
    plt.plot(times)
    plt.xlabel("Episodes")
    plt.ylabel("Time")
    plt.title("Time to complete the task")
    plot_path = FIGRURES_DIR / "3time_plot.png"
    plt.savefig(plot_path)
    plt.close()

#Q-value heatmap
def plot_q_values(Q_table):
    plt.imshow(Q_table, cmap='hot', interpolation='nearest')
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.title("Q-Value Evolution")
    action_labels = ['move_left', 'move_right', 'move_forward']
    plt.xticks(range(len(action_labels)), action_labels)
    # Define the state labels based on the comments in determine_state
    state_labels = ['green_and_food', 'food_not_green', 'no_food_green', 'no_food_no_green']
    plt.yticks(range(len(state_labels)), state_labels)
    plot_path = FIGRURES_DIR / "3_Q_values_heatmap.png"
    plt.savefig(plot_path)
    plt.close()

#action distribution barchart
def plot_action_distribution(Q_table):
    # _, Q_table = train_model(rob, episodes, steps_per_episode)
    action_distribution = np.sum(Q_table, axis=0)
    plt.bar(range(3), action_distribution)
    plt.xticks(range(3), ['move_left', 'move_right', 'move_forward'])
    plt.ylabel("Total Q-Value")
    plt.title("Action Distribution")
    plot_path = FIGRURES_DIR / "3_action_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    
def avg_reward_plot(reward_list):
    avg_reward = []
    for i in range(1, len(reward_list)+1):
        avg_reward.append(sum(reward_list[:i])/i)
    plt.plot(avg_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Episode")
    plot_path = FIGRURES_DIR / "3_avg_reward_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
def avg_learning_curve(all_rewards):
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    episodes = range(1, len(avg_rewards) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_rewards, label='Average Reward')
    plt.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Average Learning Curve')
    plt.legend()
    plot_path = FIGRURES_DIR / "3_avg_learning_curve.png"
    plt.savefig(plot_path)
    plt.close()
    
def avg_times(all_timelines):
    avg_timeline = np.mean(all_timelines, axis=0)
    std_timeline = np.std(all_timelines, axis=0)
    time_steps = range(len(avg_timeline))
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, avg_timeline, label='Average Value')
    plt.fill_between(time_steps, avg_timeline - std_timeline, avg_timeline + std_timeline, alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Average Timeline')
    plt.legend()
    plot_path = FIGRURES_DIR / "3_avg_timeline.png"
    plt.savefig(plot_path)
    plt.close()

    
def plots(reward_list, Q_table, time_list): #just call this in the controller.py
    plot_learning_curve(reward_list)
    print("learning curve saved \n")
    plot_q_values(Q_table)
    print("q values saved \n")
    plot_action_distribution(Q_table)
    print("action distribution saved \n")
    time_plot(time_list)
    print("time plot saved \n")
    
    
    
def avg_plots(all_rewards, all_timelines): #and just call this for the avg plots
    avg_reward_plot(all_rewards)
    print("average reward plot saved \n")
    avg_learning_curve(run_multiple_trainings(robobo_sim))
    print("average learning curve saved \n")
    avg_times(all_timelines)
    print("average times saved \n")
    
    










# """
# ################################################
# task2
# ################################################
# """
# # def adjust_pan_and_tilt(rob: IRobobo):
# #     if isinstance(rob, SimulationRobobo):
# #         rob.play_simulation()
# #         rob.set_phone_pan_blocking(180, 10) #make the camera used to face the front of the robot
# #         rob.set_phone_tilt_blocking(110, 10) #tilt the phone slightly to the ground for wider view of the arena


    
# def get_image(rob: IRobobo):
#     image = rob.get_image_front()
#     image = cv2.flip(image, 0) #the image was upside down before 
#     cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
#     return image

# def approach_green_object(rob):
#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()
#     prev_image = get_image(rob)
#     hsv_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2HSV)
#     #shades of green
#     lower_green = np.array([40, 40, 40])
#     upper_green = np.array([80, 255, 255])
#     mask = cv2.inRange(hsv_image, lower_green, upper_green)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     action_taken = "none"


#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         cv2.rectangle(prev_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         center_x = x + w // 2 #center of the green object
#         image_center_x = prev_image.shape[1] // 2 #center of what the camera generally sees

#         #this is how each action is triggered:
#         if center_x < image_center_x - 200: #if the object is to the left of the center...
#             rob.move_blocking(0, 70, 700)
#             action_taken = 'left'
#         elif center_x > image_center_x + 200: #same for the right
#             rob.move_blocking(70, 0, 700)
#             action_taken = 'right'
#         else: #if the object is in the center
#             rob.move_blocking(100, 100, 900) 
#             action_taken = 'forward'
#     else: #if there is no green object in sight, randomly choose to search right or step back
#         if random.choice([True, False]): 
#             rob.move_blocking(50, -10, 500)
#             action_taken = "search_right"
#         else:
#             rob.move_blocking(-50, -50, 500)
#             action_taken = "step_back"

#     current_image = get_image(rob)
#     return action_taken, prev_image, current_image

# def calculate_reward(rob, action_result, previous_image, current_image, food_counter):
#     reward = 0
#     if action_result in ["right", "left", "forward"]:
#         if action_result == "forward" and is_closer_to_green_object(previous_image, current_image):
#             reward += 10 
#         elif touched_green_object(current_image):
#             reward += 30 + 2**food_counter #the more food eaten, the bigger the reward
#     if hit_wall(rob):
#         reward -= 20
#     # reward -= 1
#     return reward

# # def is_closer_to_green_object(previous_image, current_image):
# #     prev_distance = find_green_object_distance(previous_image)
# #     curr_distance = find_green_object_distance(current_image)
# #     return curr_distance < prev_distance

# # def find_green_object_distance(image):
# #     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #     mask = cv2.inRange(hsv_image, np.array([40, 40, 40]), np.array([80, 255, 255]))
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if contours:
# #         largest_contour = max(contours, key=cv2.contourArea)
# #         _, _, w, h = cv2.boundingRect(largest_contour)
# #         return 1 / (w * h)  #if area is bigger, distance is smaller, so we use the inverse
# #     return float('inf')

# def touched_green_object(image):
#     threshold_area = 1000  
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_image, np.array([40, 40, 40]), np.array([80, 255, 255]))
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         _, _, w, h = cv2.boundingRect(largest_contour)
#         return w * h > threshold_area #to make sure the object is being touched using the 100 threshold number
#                                         #that looks like it works well
#     return False


# def hit_wall(rob):
#     ir_data = rob.read_irs()  # type(ir_data) => list
#     front_sensor_data = {2: [], 3: [], 4: [], 5: [], 7: []} #old code from task0
#     for i in [2, 3, 4, 5, 7]:
#         if ir_data[i] > 10000 or ir_data[i] == float('inf'):
#             ir_data[i] = 10000
#         front_sensor_data[i].append(ir_data[i])
#     front_ir_values = [value for sublist in front_sensor_data.values() for value in sublist] 
#     action_taken, _, _ = approach_green_object(rob)
#     if action_taken in ["search_right", "search_left"] and any(sensor > 1000 for sensor in front_ir_values): #
#         return True
#     return False  

# def determine_state(image):
#     # 0: left, 1: center, 2: right, 3: none
#     #choose the state based on the position of the green object in the image
#     position = find_green_object_position(image)
#     state_mapping = {'left': 0, 'center': 1, 'right': 2, 'none': 3}
#     return state_mapping[position]

# def find_green_object_position(image): #based on the horizontal position of the food 
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_green = np.array([40, 40, 40])
#     upper_green = np.array([80, 255, 255])
#     mask = cv2.inRange(hsv_image, lower_green, upper_green)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         center_x = x + w // 2
#         image_center_x = image.shape[1] // 2
#         if center_x < image_center_x / 3:
#             return 'left'
#         elif center_x > 2 * image_center_x / 3:
#             return 'right'
#         else:
#             return 'center'
#     return 'none'

# def train_model(rob, episodes, steps_per_episode):
#     # Q-learning parameters
#     alpha = 0.1  #α is the learning rate
#     gamma = 0.9  #γ is the discount factor
#     epsilon = 0.4  #ε is the exploration rate, 0.4 is the initial value

#     num_states = 4  #left, center, right, none
#     num_actions = 5  #move_left, move_right, move_forward, search_right, step_back
#     Q_table = np.zeros((num_states, num_actions))

#     # if isinstance(rob, SimulationRobobo):
#     #     rob.play_simulation()
#     time_list = []
#     for episode in range(episodes):
#         adjust_pan_and_tilt(rob)
#         if isinstance(rob, SimulationRobobo):
#             rob.play_simulation()
        
#         current_image = get_image(rob)
#         current_state_index = determine_state(current_image)
#         total_reward = 0
#         reward_list = [] #only used for plotting 
#         food_counter = 0


#         for step in range(steps_per_episode):
#             if np.random.rand() < epsilon: 
#                 action_index = np.random.randint(num_actions)  #chances of choosing a random action are getting lower as 
#                                                                 #the robot slowly learns
#                 # print("random action: ", action_index)
#             else:
#                 action_index = np.argmax(Q_table[current_state_index])  #and chances of choosing the best action
#                                                                         #are getting higher as robot learns
#                 # print("optimal action: ", action_index)

#             action_result, previous_image, current_image = approach_green_object(rob) #action
#             next_state_index = determine_state(current_image) #state
#             reward = calculate_reward(rob, action_result, previous_image, current_image, food_counter) #reward
#             if reward > 11: #getting the exact == value of reward didn't work for some reason but this works
#                 food_counter += 1
#                 if food_counter > 6: #this was SOMETIMES working (??) in my computer
#                     print("All food found!")
#                     print("Time needed to complete the task:", rob.get_sim_time()) 
#                     time_list.append(rob.get_sim_time())
#                     break

#             #update q_table
#             old_value = Q_table[current_state_index, action_index]
#             future_optimal_value = np.max(Q_table[next_state_index])
#             # print("old value: ", old_value)
#             # print("future optimal value: ", future_optimal_value)
            
#             #formula is: Q(s,a) = Q(s,a) + α(r + γmaxQ(s',a') - Q(s,a)) 
#             new_q_value = old_value + alpha * (reward + gamma * future_optimal_value - old_value)
#             # print("new q value: ", new_q_value)
#             Q_table[current_state_index, action_index] = new_q_value 
#             # print("Q_table: \n", Q_table)

#             current_state_index = next_state_index
#             total_reward += reward
#             reward_list.append(total_reward)


#         print(f"Episode {episode + 1}: Total Reward: {total_reward}")
#         print("Q_table of whole episode: \n", Q_table)
#         epsilon *= 0.97 #because 97's kids are cool and 0.99 was too slow
#         if isinstance(rob, SimulationRobobo):
#             rob.stop_simulation()
#     # if isinstance(rob, SimulationRobobo):
#     #     rob.stop_simulation()

#     print("Training completed.")
#     #save the Q_table
#     Q_table_array = np.array(Q_table)
#     with open(FIGRURES_DIR / "Q_table.pkl", "wb") as f: #figures dir was familiar so q-table was saved there
#         pickle.dump(Q_table_array, f)
#     print("Q_table saved.")
#     print("time_list: ", time_list)
#     return reward_list, Q_table, time_list
    


# def run_robot(rob): #Q_table is the trained Q_table from the training function, for robobo to perform what it learnt
#     print("Now running the run robot function")
    
#     try:
#         with open(FIGRURES_DIR / "Q_table.pkl", "rb") as f:
#             Q_table = pickle.load(f)
#         print("Q_table loaded.")
#     except FileNotFoundError:
#         print("Q_table not found.")
#         return
#     except Exception as e:
#         print("An error occurred while loading the Q_table:")
#         print(e)
#         return
    
#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()
#     adjust_pan_and_tilt(rob)
#     current_image = get_image(rob)
#     current_state_index = determine_state(current_image)
#     total_reward = 0
#     food_counter = 0
#     while True:
#         action_index = np.argmax(Q_table[current_state_index]) #choose best action
#         action_result, previous_image, current_image = approach_green_object(rob)
#         next_state_index = determine_state(current_image)
#         reward = calculate_reward(rob, action_result, previous_image, current_image, food_counter)
#         if reward > 11: #again, the exact == value did not really work
#             food_counter += 1
#             print(f"Food found! Total food found: {food_counter}")
#         total_reward += reward
#         current_state_index = next_state_index

#         if total_reward < -10 or food_counter > 6: #stop if robot doesn't perform well or if it has found all the food
#                                                 #but still sometimes the robot doesn't stop I dunno why
#             print("Ate it all!")
#             # print("Time needed to complete the task:", rob.get_sim_time(), "seconds")
#             break
        
#         print(f"Total Reward: {total_reward}")
#         if isinstance(rob, SimulationRobobo):
#             rob.stop_simulation()
            
        

#############Plots#################


# def plot_learning_curve(reward_list):
#     # reward_list, _ = train_model(rob, episodes, steps_per_episode)
#     plt.plot(reward_list)
#     plt.xlabel("Episodes")
#     plt.ylabel("Total Reward")
#     plt.title("Learning Curve")
#     plot_path = FIGRURES_DIR / "learning_curve.png"
#     plt.savefig(plot_path)
#     plt.close()
    

# #q-values heatmap
# def plot_q_values(Q_table):
#     # _, Q_table = train_model(rob, episodes, steps_per_episode)
#     plt.imshow(Q_table, cmap='hot', interpolation='nearest')
#     plt.xlabel("Actions")
#     plt.ylabel("States")
#     plt.title("Q-Value Evolution")
#     plot_path = FIGRURES_DIR / "Q_values_heatmap.png"
#     plt.savefig(plot_path)
#     plt.close()
    
# #actions' bar chart
# def plot_action_distribution(Q_table):
#     # _, Q_table = train_model(rob, episodes, steps_per_episode)
#     action_distribution = np.sum(Q_table, axis=0)
#     plt.bar(range(5), action_distribution)
#     plt.xticks(range(5), ['move_left', 'move_right', 'move_forward', 'search_right', 'step_back'])
#     plt.ylabel("Total Q-Value")
#     plt.title("Action Distribution")
#     plot_path = FIGRURES_DIR / "action_distribution.png"
#     plt.savefig(plot_path)
#     plt.close()
    
    
# def time_plot(times):
#     plt.plot(times)
#     plt.xlabel("Episodes")
#     plt.ylabel("Time")
#     plt.title("Time to complete the task")
#     plot_path = FIGRURES_DIR / "time_plot.png"
#     plt.savefig(plot_path)
#     plt.close()
    
# #you only have to call plots() in controller file to get plots    
# def plots(reward_list, Q_table, time_list):
#     plot_learning_curve(reward_list)
#     plot_q_values(Q_table)
#     plot_action_distribution(Q_table)
#     time_plot(time_list)
#     print("Plots saved.")

    













