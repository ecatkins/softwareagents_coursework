from prettytable import PrettyTable
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt
import os



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
    
    def __init__(self, y_size, x_size, object_locations, can_locations, start_location = (0,0)):
        # Define
        self.y_size = y_size
        self.x_size = x_size
        self.object_locations = object_locations
        self.can_locations = can_locations
        self.start_location = start_location
        self.num_cans = len(can_locations)
        self.history = []

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
                if to_location == start_location and all(to_canState[x] == 'picked' for x in to_canState) and dist == 1 and (len(shared_spots) == self.num_cans):
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
        self.room.print_room(self.current_location)
        self.current_state = self.find_start_state()

    def move(self,greedy = False):
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

        # if random policy just choose one of possible
        # states at random
        else:
            chosen_index = random.choice(indexes)
            reward = possible_moves.ix[chosen_index]
        

        # Assign new current state and current location
        self.current_state = chosen_index
        self.current_location = self.possible_states[chosen_index]['current_location']

        # Update room and print
        self.room.print_room(self.current_location)

        # Assign new q value
        self.q_update(old_state, self.current_state, reward)

        return reward

    def q_update(self, old_state, new_state, immediate_reward):
        #########
        gamma = 0.8
        alpha = 0.9
        ########
        
        q_old = self.q_matrix.iloc[old_state,new_state]
        next_state_q = self.q_matrix.iloc[new_state,:]
       
        #### WRONG
        q_new = q_old + alpha *(immediate_reward + gamma * max(next_state_q) - q_old)
        print("Immediate Reward {}".format(immediate_reward))
        print("Old q {}".format(q_old))
        print("New q {}".format(q_new))
        self.q_matrix.iloc[old_state, new_state] = q_new    

    def run_episode(self, start_location, epsilon):
        self.current_location = start_location
        self.number_steps = 0
        os.system('clear')
        self.reset_episode()
        terminate = False
        
        random_list = [True] * int(epsilon * 100) + [False] * int((1 - epsilon) * 100)
        
        while terminate == False:
            self.number_steps += 1
            
            random_bool = random.choice(random_list)
            
            ### Chooses random move with epsilon probability
            if random_bool:
                reward = self.move(greedy = False)
            else:
                reward = self.move(greedy = True)   
            if reward == 5:
                terminate = True
                self.history.append(self.number_steps)
                self.number_steps = 0

    def run_model(self, iterations):
        epsilon = 0.95
        start_location = (0,0)
        iterative_step = epsilon / iterations
        for x in range(iterations):
            epsilon -= iterative_step
            print("NEW EPISODE")
            self.run_episode(start_location, epsilon)

        x = [i for i in range(len(self.history))]
        y = self.history
        print(self.history)
        plt.plot(x,y)
        plt.show()

            
# object_locations = ([1,1],[1,2])
# can_locations = ([3,3], [4,4])
object_locations = ([3,2],[3,3],[3,4],[3,5],[3,6],[4,2],[5,2],[6,2])
# can_locations = ([7,1],[2,1],[4,4],[8,8])
can_locations = ([7,1],[2,1])
y_size = 10
x_size = 10
start_location = (0,0)

roomba = Roomba(y_size = y_size, x_size = x_size, object_locations = object_locations, can_locations = can_locations)

roomba.run_model(1500)






