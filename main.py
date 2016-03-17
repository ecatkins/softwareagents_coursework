from prettytable import PrettyTable
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt
import os
import pickle
import math
import pdb



class Room:
    
    def __init__(self, y_size, x_size, object_locations, can_locations):
        self.room_array = self.create_room(y_size,x_size, object_locations, can_locations)
        
    def create_room(self, y_size, x_size, object_locations, can_locations):
        x_vector = [''] * x_size
        array = [x_vector.copy() for i in range(y_size)]
        for obj in object_locations:
            y,x = obj
            array[y][x] = 'X'
        for can in can_locations:
            y,x = can
            array[y][x] = 'O'
        return array
        
    def update_room(self,y,x):
        ## Updates if a can was picked up
        if self.room_array[y][x] == 'O':
            self.room_array[y][x] = '.'
        return self.room_array
        
    def print_room(self,current_location):
        os.system('clear')
        room_image = self.room_array[:]
        table = PrettyTable()
        y,x = current_location
        room_image[y][x] = 'R'
        table.add_row([' '] + ['x_{}'.format(i) for i in range(len(room_image[0]))])
        for index, row in enumerate(room_image):
            full_row = ['y_{}'.format(index)] + row
            table.add_row(full_row)
        table.align = 'c'
        table.header = False
        print(table)


class Roomba:
    
    def __init__(self, y_size, x_size, object_locations, can_locations, start_location = (0,0), show_room = True):
        self.show_room = show_room

        # Define
        self.y_size = y_size
        self.x_size = x_size
        self.object_locations = object_locations
        self.can_locations = can_locations
        self.start_location = start_location
        self.num_cans = len(can_locations)
        self.history = []
        self.epsilon_history = []

        # Initialise
        self.room = Room(y_size, x_size, object_locations, can_locations)
        self.possible_states = self.get_possible_states()
        self.reward_matrix = self.get_reward_matrix()
        self.q_matrix = self.get_Q_matrix()

    def get_possible_states(self):
        room_array = self.room.room_array

        cans = []
        robot_positions = []

        ### Loop through the room array
        for y_index, y in enumerate(room_array):
            for x_index in range(len(room_array[0])):
                # Check if the square has a can
                if room_array[y_index][x_index] == 'O':
                    option_one = (y_index, x_index, 'can')
                    option_two = (y_index, x_index, 'picked')
                    cans.append([option_one,option_two])
                # check if square not valid
                if room_array[y_index][x_index] != 'X':
                    robot_positions.append((y_index, x_index))

        ### Creates buckets of all possible clean/not clean states and robot permutations
        final_list = [robot_positions] + cans
        states = list(itertools.product(*final_list))

        possible_states = {}
        ### MAKE SURE YOU GET RID OF STATES WHERE THE BOTTLE IS NOT PICKED UP
        #Creating a dictionary of all states
        index_count = 0
        for state in states:
            valid = True
            # Dictionary within the larger dictionary
            state_dict = {}
            ### Adds the current location
            current_location = state[0]
            state_dict['current_location'] = current_location
            squares_dict = {}
            squares = state[1:]
            # Loop through all squares with cans
            for square in squares:
                y_coord = square[0]
                x_coord = square[1]
                state = square[2]
                 # Key is location, value is can state
                squares_dict[(y_coord, x_coord)] = state
                # The robot can't be on a location, and that location also contain a can
                if (y_coord, x_coord) == current_location and state == 'can':
                    valid = False
            # Add squares dictionary
            state_dict['can_state'] = squares_dict
            
            # Add smaller dictionary to larger dictionary
            if valid:
                possible_states[index_count] = state_dict
                index_count += 1
        return possible_states
            
    def get_reward_matrix(self):
        
        #ROWS are stating coming FROM
        #COLUMNS are state going TO
        print("Number of states")
        print(len(self.possible_states))
        go_vector = ['.'] * len(self.possible_states)
        reward_matrix = [go_vector.copy() for x in range(len(self.possible_states))]
        reward_matrix = pd.DataFrame(reward_matrix)


        # Loop through all possible states
        for from_index in self.possible_states:
            # Gets the state out of the dict
            from_state = self.possible_states[from_index]
            # Location coming from
            from_location = from_state['current_location']
            #The state of all the cans in the FROM state
            from_canState = from_state['can_state']
    
            # Loop through all possible states
            for to_index in self.possible_states:
                # Gets the state out of the dict
                to_state = self.possible_states[to_index]
                # Location going to
                to_location = to_state['current_location']
                # The state of all the cans in the TO state
                to_canState = to_state['can_state']

                #Check if the movement is valid

                # Distance between one state and the other should be of length 1
                dist = distance.euclidean(np.array(from_location), np.array(to_location))
                
                #Checks to see how many of the cans are in the same state from one to the next
                shared_spots = shared_items = set(from_canState.items()) & set(to_canState.items())

                #If all the cans are in the same state and the distance is one, it is a VALID move but the reward is zero
                if len(shared_spots) == self.num_cans and dist == 1:
                    reward_matrix.iloc[from_index, to_index] = 0

                #If the number of cans in the same state is less than the number of total cans by one, and the can change is in the to location from 'picked' to 'can' then the reward is 1
                
                elif to_location in from_canState.keys() and len(shared_spots) == self.num_cans - 1 and to_canState[to_location] == 'picked' and from_canState[to_location] == 'can'and dist == 1:
                    reward_matrix.iloc[from_index, to_index] = 1

                # if the robot goes back to the start location and all the cans are picked
                # **** Think this might be buggy, doesn't account for if can sits right next to starting point on the way home
                if to_location == self.start_location and all(to_canState[x] == 'picked' for x in to_canState) and dist == 1 and (len(shared_spots) == self.num_cans):
                    reward_matrix.iloc[from_index, to_index] = 5

        return reward_matrix

    def get_Q_matrix(self):
        reward_matrix = self.reward_matrix
        q_matrix = reward_matrix.copy()
        q_matrix[q_matrix!= 0] = 0
        return q_matrix

    def find_start_state(self):
        for state in self.possible_states:
            state_dict = self.possible_states[state]
            if state_dict['current_location'] == self.current_location:
                can_state = state_dict['can_state']
                if all(can_state[x] == 'can' for x in can_state):
                    return state

    def reset_episode(self):
        self.room = Room(self.y_size, self.x_size, self.object_locations, self.can_locations)
        y = self.current_location[0]
        x = self.current_location[1]
        
        if self.show_room:
            self.room.print_room(self.current_location)
        
        self.current_state = self.find_start_state()

    def move(self,greedy = False, policy2=None):
        y = self.current_location[0]
        x = self.current_location[1]

        self.room.update_room(y,x)


        old_state = self.current_state
        
        #Given FROM state, choose all moves in TO state
        # in reward matrix
        all_moves_r = self.reward_matrix.iloc[old_state,:]
        
        # Gets all possible moves and their indexes
        possible_moves = all_moves_r[all_moves_r!='.']
        indexes = possible_moves.index

        #Given FROM state, choose all moves in TO state
        # in Q matrix
        all_moves_q = self.q_matrix.iloc[old_state,:]
        q_max_move = max(all_moves_q)

        # if greedy policy
        if greedy:
            
            # If all possible moves are zero, pick a random move
            if q_max_move == 0:
                return self.move(greedy=False)
            
            # Pick the best moves
            max_move_series = all_moves_q[all_moves_q == q_max_move]
            
            # There might be more than one so pick at random
            indexes = max_move_series.index
            chosen_index = random.choice(indexes)
            reward = self.reward_matrix.iloc[old_state,chosen_index]


       
        ## Implement a softmax policy
        elif policy2 == 'softmax':
            # pdb.set_trace()
            temperature = 0.7
            valid_moves_q = all_moves_q[all_moves_r != '.']

            sum_probs = np.sum([math.exp(i/temperature) for i in valid_moves_q])
            all_probs = [math.exp(i/temperature) /sum_probs for i in valid_moves_q]
            weighted_random_choice = np.random.choice(valid_moves_q, p=all_probs)
            matching_moves = valid_moves_q[valid_moves_q == weighted_random_choice]
            indexes = matching_moves.index
            chosen_index = random.choice(indexes)
            reward = self.reward_matrix.iloc[old_state, chosen_index]


        # if random policy just choose one of the possible
        # states at random
        else:
            chosen_index = random.choice(indexes)
            reward = possible_moves.ix[chosen_index]
        

        # Assign new current state and current location
        self.current_state = chosen_index
        self.current_location = self.possible_states[chosen_index]['current_location']

        # Update room and print
        if self.show_room:
            self.room.print_room(self.current_location)

        # Assign new q value
        self.q_update(old_state, self.current_state, reward)

        return reward

    def q_update(self, old_state, new_state, immediate_reward):
        #########
        gamma = self.gamma
        alpha = self.alpha
        ########
        
        q_old = self.q_matrix.iloc[old_state,new_state]
        next_state_q = self.q_matrix.iloc[new_state,:]
        
        q_new = q_old + alpha * (immediate_reward + gamma * max(next_state_q) - q_old)
        
        if self.show_room:
            print("Immediate Reward {}".format(immediate_reward))
            print("Old q {}".format(q_old))
            print("New q {}".format(q_new))

        self.q_matrix.iloc[old_state, new_state] = q_new    

    def run_episode(self, start_location, epsilon, policy2):
        self.current_location = start_location
        self.number_steps = 0
        self.reset_episode()
        terminate = False
        termination_steps = 100
        
        random_list = [True] * int(epsilon * 100) + [False] * int((1 - epsilon) * 100)
        
        while terminate == False:
            self.number_steps += 1
            random_bool = random.choice(random_list)
            
            ### Chooses random move with epsilon probability
            if random_bool:
                reward = self.move(greedy = False, policy2 = policy2)
            else:
                reward = self.move(greedy = True, policy2 = policy2)   
            if reward == 5:
                terminate = True
                self.history.append(self.number_steps)
                self.epsilon_history.append(epsilon)
                self.number_steps = 0

            # ## Terminate if number of steps over
            
            # if self.number_steps == 100:
            #     self.history.append(self.number_steps)
            #     self.epsilon_history.append(epsilon)
            #     terminate = True


    def check_convergence(self):
        # last  = self.history[-100:]
        # last_Q75 = np.percentile(last,75)
        # second_last = self.history[-200:-100]
        # second_last_Q75 = np.percentile(second_last,75)
        # if last_Q75 >= second_last_Q75:
        #     return True
        # else:
        #     return 
        last_five = self.history[-5:]
        last = self.history[-1]
        if last_five.count(last) >= 4 and last != 100:
            return True
        else:
            return False



    

    def run_model(self, iterations, gamma, alpha, policy, policy_start, policy_decay, plot):
        self.gamma = gamma
        self.alpha = alpha

        start_epsilon = 0.95


        policy_fac = policy_start

        for x in range(iterations):
            policy_factor = policy_start - x *  math.exp(- policy_decay * x)


       
        for x in range(iterations):
            if policy == 'linear':
                epsilon = start_epsilon - x * (start_epsilon - end_epsilon) / iterations

            elif policy == 'exponential5':
                exp_factor = 5
                epsilon = start_epsilon * math.exp(- exp_factor / iterations * x)

            elif policy =='exponential20':
                exp_factor = 20
                epsilon = start_epsilon * math.exp(- exp_factor / iterations * x)

            elif policy =='exponential50':
                exp_factor = 50
                epsilon = start_epsilon * math.exp(- exp_factor / iterations * x)

            elif policy == 'exponential100':
                exp_factor = 100
                epsilon = start_epsilon * math.exp(- exp_factor / iterations * x)

            if p


            print("NEW EPISODE ------> {}".format(x))
            print("Epsilon {}".format(epsilon))
            self.run_episode(self.start_location, epsilon, policy2 = policy2)
             

            # if x > 10:
            #     #Breaks the loop if convergence is reached
            #     if self.check_convergence():
            #       print("Convergence reached at {} iterations".format(x))
            #       print("Epsilon value is {}".format(epsilon))
            #       break


        x = [i for i in range(len(self.history))]
        y = self.history
        print(self.history)
        plt.plot(x,y)
        if plot == True:
            plt.show()

        name = 'model' + str(round(gamma,1))[-2:] +'_' + str(round(alpha,2))[-2:] + '_' + str(policy)
        model_run_description_string = "Gamma: {}, Alpha: {}, Policy: {}, Policy2: {}".format(gamma,alpha,policy, policy2)
        model_run_parameters = {'gamma':gamma,'alpha':alpha,'policy':policy}
        results = [name, model_run_description_string, model_run_parameters, self.history, self.epsilon_history]
        return results

            

# object_locations = ([3,2],[3,3],[3,4],[3,5],[3,6],[4,2],[5,2],[6,2])
# can_locations = ([7,1],[2,1])
# y_size = 10
# x_size = 10
gamma_list = [0.9]
alpha_list = [0.9]
policy_list = ['egreedy','softmax']
policy_start_list = [0.95,0.5,0.1]
policy_decay_list = [20,50]

def grid_search(iterations, gamma_list, alpha_list, policy_list,plot=True):
    # object_locations = ([3,2],[3,3],[3,4],[3,5],[3,6],[4,2],[5,2],[6,2],[8,1],[2,8])
    # can_locations = ([2,1],[7,1],[7,7],[1,2],[7,2],[4,6])
    # y_size = 10
    # x_size = 10

    object_locations = ([3,2],[3,3],[3,4],[3,5],[3,6],[4,2],[5,2],[6,2])
    can_locations = ([7,1],[2,1])
    y_size = 10
    x_size = 10
    start_location = (0,0)

    for gamma in gamma_list:
        for alpha in alpha_list:
            for policy in policy_list:
                for policy_start in policy_start_list:
                    for policy_decay in policy_decay_list:
                        roomba = Roomba(y_size = y_size, x_size = x_size, object_locations = object_locations, can_locations = can_locations, show_room = False)
                        results = roomba.run_model(iterations= iterations, gamma= gamma, alpha = alpha,policy = policy,policy_start= policy_start,policy_decay = policy_decay, plot=False)
                        pickle.dump(results, open('pickles2/{}.p'.format(results[0]), 'wb'))


grid_search(200,gamma_list, alpha_list,policy_list,plot=False)


#### Measurements
# 1. Value converged to
# 2. Total number steps to converge
# 3. Episodes to converge


