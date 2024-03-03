import pandas as pd
import numpy as np
import random

from symbolic_regression import SymbolicRegressor, compute_rmse, Node, print_tree_w_indent


class QLearningSymbolicRegressor:
    def __init__(self, operators, constants, features, depth=5, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.operators = operators
        self.features = features
        self.constants = constants
        self.depth = depth
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.action_space = None
        self.regressor: SymbolicRegressor = None

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)
        else:
            found_key = False
            max_value = -float('inf')
            best_action = None
            for key, val in self.q_table.items():
                if key[0] == state and val > max_value:
                    found_key = True
                    max_value = val
                    best_action = key[1]
            # if state in self.q_table:
            if found_key:
                return best_action
            else:
                return random.choice(self.action_space)
            
    def take_action(self, action):
        action[0].value = action[1]

    def check_if_state_in_q_table(self, state):
        for key in self.q_table.keys():
            if key[0] == state:
                return True
        return False
        
    def update_q_table(self, state, action, reward, next_state):
        if not self.check_if_state_in_q_table(state):
            for action in self.action_space:
                self.q_table[(state, action)] = 0.
        if self.check_if_state_in_q_table(next_state):
            max_next_q = float('-inf')
            for state_action, q_value in self.q_table.items():
                state_ = state_action[0]
                if state_ == next_state:
                    max_next_q = max(max_next_q, q_value)
        else:
            max_next_q = 0.

        current_q = self.q_table[(state, action)]
        self.q_table[(state, action)] += self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

    def determine_possible_value(self, node: Node):
        if node.children:
            return list(self.regressor.operator_mapping.keys())
        else:
            return list(self.regressor.leaves_mapping.keys())

    def define_action_space(self):
        self.action_space = [(node, op) for node in self.regressor.get_all_nodes(self.regressor.tree) 
                             for op in self.determine_possible_value(node)]
    
    def train(self, X, y, num_episodes=100):
        self.regressor = SymbolicRegressor(self.operators, self.constants, self.features, self.depth)
        self.define_action_space()
        best_rmse = float('inf')
        for episode in range(num_episodes):
            state = tuple(self.regressor.encode_state_mapped())
            while True:
                action = self.select_action(state)
                self.take_action(action)
                next_state = tuple(self.regressor.encode_state_mapped())
                y_pred = self.regressor.evaluate_tree(self.regressor.tree, X)
                reward = -compute_rmse(y, y_pred)
                self.update_q_table(state, action, reward, next_state)
                state = tuple(next_state)
                rmse_val = -reward
                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                if stagnation_count >= 500:
                    print(f"Training terminated due to stagnation after {episode+1} episodes.")
                    return
                


if __name__ == "__main__":
    df = pd.read_csv('insurance.csv')
    df = pd.get_dummies(df)
    features = ['age', 'bmi', 'children', 'sex_female', 'sex_male',
        'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
        'region_southeast', 'region_southwest']
    target = 'charges'
    X = df[features].values
    y = df[target].values

    constants = list(map(str, range(0, 20)))
    operators = ['+', '-', '*']

    q_learning_regressor = QLearningSymbolicRegressor(operators, constants, features)
    q_learning_regressor.train(X, y)

    max_q_value = -float('inf')
    best_state = None
    for (state, action), q_value in q_learning_regressor.q_table.items():
        if q_value > max_q_value and q_value != 0.:
            best_state = state
            max_q_value = q_value

    best_tree = q_learning_regressor.regressor.build_tree_from_encoded_state(best_state)

    print_tree_w_indent(best_tree)

    y_pred = q_learning_regressor.regressor.evaluate_tree(best_tree, X)
    print("RMSE best tree", compute_rmse(y, y_pred))