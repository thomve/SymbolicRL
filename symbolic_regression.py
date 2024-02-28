import numpy as np
import pandas as pd


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class SymbolicRegressor:
    def __init__(self, features, depth):
        self.operators = ['+', '-', '*']
        self.features = features
        self.constants = list(map(str, range(1, 11)))
        self.max_children = 2
        self.tree = self.generate_random_tree(depth)

    def evaluate_tree(self, node, X):
        if node.value in self.operators:
            operator = node.value
            operands = [self.evaluate_tree(child, X) for child in node.children]
            if operator == '+':
                return np.sum(operands, axis=0)
            elif operator == '-':
                return operands[0] - np.sum(operands[1:], axis=0)
            elif operator == '*':
                return np.prod(operands, axis=0)
            # elif operator == '/':
            #     return operands[0] / np.prod(operands[1:], axis=0)
        elif node.value in self.features:
            return X[:, self.features.index(node.value)]
        elif node.value.isdigit():
            return float(node.value) * np.ones(len(X))
        else:
            raise ValueError("Invalid node value")

    def generate_random_tree(self, depth):
        if depth == 0:
            return Node(np.random.choice(self.features + self.constants))
        else:
            node = Node(np.random.choice(self.operators))
            for _ in range(np.random.randint(2, self.max_children+1)):
                node.add_child(self.generate_random_tree(depth - 1))
            return node

    def take_action(self, node):
        new_value = np.random.choice(self.operators) if node.children else np.random.choice(self.features + self.constants)
        original_value = node.value
        node.value = new_value
        return original_value

    def restore_node(self, node, original_value):
        node.value = original_value  # Restore the original value

    def get_reward(self, X, y_true):
        y_pred = self.evaluate_tree(self.tree, X)
        return compute_rmse(y_true, y_pred)

    def get_state(self):
        pass


def print_tree(node, indent=0):
    print(" " * indent + str(node.value))
    for child in node.children:
        print_tree(child, indent + 4)
        
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
    

df = pd.read_csv('insurance.csv')
df = pd.get_dummies(df)
features = ['age', 'bmi', 'children', 'sex_female', 'sex_male',
       'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
       'region_southeast', 'region_southwest']
target = 'charges'
X = df[features].values
y = df[target].values


sr = SymbolicRegressor(features, depth=4)

for i in range(10000):
    rmse_before = sr.get_reward(X, y)

    selected_node_index = np.random.choice(range(len(sr.tree.children)))  # Randomly select a child node index
    selected_node = sr.tree.children[selected_node_index]

    original_value = sr.take_action(selected_node)
    
    rmse_after = sr.get_reward(X, y)

    # Check if temporary modification improves performance
    if rmse_before < rmse_after:
        sr.restore_node(selected_node, original_value)
    else:
        rmse_before = rmse_after

print(rmse_before)

print_tree(sr.tree)