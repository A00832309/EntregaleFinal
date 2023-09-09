# UNA PARTE DE LAS FUNCIONES Y ALGORITMOS UTILIZADOS FUERON OBTENIDOS DE LA SIGUIENTE LIGA:
# https://colab.research.google.com/drive/1E2RViy7xmor0mhqskZV14_NUj2jMpJz3
# CODIGO CREADO POR DR. DANIEL SOPER (https://www.youtube.com/@DanielSoper)

import math
import json
from flask import Flask, request, jsonify
import agentpy as ap
import numpy as np
import random


app = Flask(__name__)

@app.route('/parameters', methods=['POST'])
def get_parameters():
    data = request.json
    return jsonify(data)


# Define the grid world model
class Campo(ap.Model):

    def setup(self):
        #define size parameters
        environment_rows, environment_columns = self.environment_rows, self.environment_columns = self.p.ancho+4, self.p.largo+4
        
        #define training parameters
        epsilon = self.p.epsilon #the percentage of time when we should take the best action (instead of a random action)
        discount_factor = self.p.discount_factor #discount factor for future rewards
        learning_rate = self.p.learning_rate #the rate at which the AI agent should learn
        
        #define grid and agents from agentpy
        self.grid = ap.Grid(self, (self.environment_rows, self.environment_columns))
        #self.agents = ap.AgentList(self, 1, QLearningAgent)
        #self.grid.add_agents(self.agents)
        
        q_values = self.q_values = np.zeros((self.environment_rows, self.environment_columns,4))  # Q-table
        actions = self.actions = ['up', 'right', 'down', 'left']
        #Create a 2D numpy array to hold the rewards for each state. 
        #The array contains 'ancho' rows and 'largo' columns (to match the shape of the environment), and each value is initialized to -100.
        rewards = np.full((self.environment_rows, self.environment_columns), -100.)

        
        #define aisle locations (i.e., white squares) for all rows
        aisles = {} #store locations in a dictionary
        for i in range(1, self.p.vueltas_harvester_cosecha + 2):
            aisles[i] = [i for i in range(1, self.environment_columns - 1)]
        for i in range(self.p.vueltas_harvester_cosecha + 2, self.environment_rows - self.p.vueltas_harvester_cosecha - 2):
            aisles[i] = [1, self.environment_columns - 2]
        for i in range(self.environment_rows - self.p.vueltas_harvester_cosecha - 2, self.environment_rows - 1):
            aisles[i] = [i for i in range(1, self.environment_columns - 1)]
        
        #set the rewards for all aisle locations (i.e., white squares)
        coordenadas_camino = []
        for row_index in range(1, len(aisles)+1):
            for column_index in aisles[row_index]:
                rewards[row_index, column_index] = -1.
                coordenadas_camino.append({"x": row_index, "y": 0, "z": column_index})
        
        rewards[self.p.posicion_x_harvester_actual, self.p.posicion_y_harvester_actual] = -100
        
        
        rewards[self.p.posicion_x_descarga, self.p.posicion_y_descarga] = 100. #set the reward for the packaging area (i.e., the goal) to 100
        coordenada_base = {"x": self.p.posicion_x_descarga, "y": 0.1, "z":  self.p.posicion_y_descarga}
        coordenada_harvester = {"x": self.p.posicion_x_harvester_actual, "y": 0, "z":  self.p.posicion_y_harvester_actual}
        
        
        #define a function that determines if the specified location is a terminal state
        def is_terminal_state(current_row_index, current_column_index):
            #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
            if rewards[current_row_index, current_column_index] == -1.:
                return False
            elif rewards[current_row_index, current_column_index] == -5.:
                return False
            else:
                return True

        #define a function that will choose a random, non-terminal starting location
        def get_starting_location():
            #get a random row and column index
            current_row_index = np.random.randint(environment_rows)
            current_column_index = np.random.randint(environment_columns)
            #continue choosing random row and column indexes until a non-terminal state is identified
            #(i.e., until the chosen state is a 'white square').
            while is_terminal_state(current_row_index, current_column_index):
                current_row_index = np.random.randint(environment_rows)
                current_column_index = np.random.randint(environment_columns)
            return current_row_index, current_column_index

        #define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
        def get_next_action(current_row_index, current_column_index, epsilon):
            #if a randomly chosen value between 0 and 1 is less than epsilon, 
            #then choose the most promising value from the Q-table for this state.
            if np.random.random() < epsilon:
                return np.argmax(q_values[current_row_index, current_column_index])
            else: #choose a random action
                return np.random.randint(4)

        #define a function that will get the next location based on the chosen action
        def get_next_location(current_row_index, current_column_index, action_index):
            new_row_index = current_row_index
            new_column_index = current_column_index
            if actions[action_index] == 'up' and current_row_index > 0:
                new_row_index -= 1
            elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
                new_column_index += 1
            elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
                new_row_index += 1
            elif actions[action_index] == 'left' and current_column_index > 0:
                new_column_index -= 1
            return new_row_index, new_column_index

        #Define a function that will get the shortest path between any location within the warehouse that 
        #the robot is allowed to travel and the item packaging location.
        def get_shortest_path(start_row_index, start_column_index):
            #return immediately if this is an invalid starting location

            if is_terminal_state(start_row_index, start_column_index):
                return []
            else: #if this is a 'legal' starting location
                current_row_index, current_column_index = start_row_index, start_column_index
                shortest_path = []
                shortest_path.append({"x": current_row_index, "y": 0, "z": current_column_index})
                #continue moving along the path until we reach the goal (i.e., the item packaging location)
                while not is_terminal_state(current_row_index, current_column_index):
                    #get the best action to take
                    action_index = get_next_action(current_row_index, current_column_index, 1.)
                    #move to the next location on the path, and add the new location to the list
                    current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
                    shortest_path.append({"x": current_row_index, "y": 0, "z": current_column_index})
                return shortest_path
        
        #run through 1000 training episodes
        for episode in range(2000):
            #get the starting location for this episode
            row_index, column_index = get_starting_location()

            #continue taking actions (i.e., moving) until we reach a terminal state
            #(i.e., until we reach the item packaging area or crash into an item storage location)
            while not is_terminal_state(row_index, column_index):
                #choose which action to take (i.e., where to move next)
                action_index = get_next_action(row_index, column_index, epsilon)

                #perform the chosen action, and transition to the next state (i.e., move to the next location)
                old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
                row_index, column_index = get_next_location(row_index, column_index, action_index)

                #receive the reward for moving to the new state, and calculate the temporal difference
                reward = rewards[row_index, column_index]
                old_q_value = q_values[old_row_index, old_column_index, action_index]
                temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

                #update the Q-value for the previous state and action pair
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                q_values[old_row_index, old_column_index, action_index] = new_q_value
            
            
        if self.p.posicion_x_harvester_actual > environment_rows/2:
            posicion_x_tractor_actual = self.p.posicion_x_harvester_actual + 1
        else:
            posicion_x_tractor_actual = self.p.posicion_x_harvester_actual - 1
        path = get_shortest_path(posicion_x_tractor_actual, self.p.posicion_y_harvester_actual)
        unity_json ={
                        "ancho": self.environment_rows,
                        "largo": self.environment_columns,
                        "waypoints": path,
                        "coordenadas_camino": coordenadas_camino,
                        "coordenada_base": coordenada_base,
                        "coordenada_harvester": coordenada_harvester
                    }
        print(unity_json)

        with open("/Users/luisangelgr01/Desktop/Multiagentes/Unity/Reto_Con_Python_v01/Assets/Resources/coordenadas.json", "w") as json_file:
            json.dump(unity_json, json_file)
    





if __name__ == "__main__":

    parameters = {
        'ancho': 28,
        'largo': 35,
        'epsilon': 0.9,
        'discount_factor': 0.9,
        'learning_rate': 0.9,
        'posicion_x_descarga': 1, # tiene que ser entre 1 y ('ancho' + 2), asumiendo que 'y' es 1 o ('largo' + 2)
        'posicion_y_descarga': 18, # tiene que ser entre 1 y ('largo' + 2), asumiendo que 'x' es 1 o ('ancho' + 2)
        'vueltas_harvester_cosecha': 12,
        'posicion_x_harvester_actual': 18,
        'posicion_y_harvester_actual': 5
    }
    model = Campo(parameters)
    model.run(1)


    # app.run(debug=True)
