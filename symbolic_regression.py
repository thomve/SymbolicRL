import numpy as np
import pandas as pd


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class SymbolicRegressor:
    def __init__(self, operators, constants, features, depth):
        self.operators = operators
        self.features = features
        self.constants = constants
        self.max_children = 2
        self.tree = self.generate_random_tree(depth)
        self.operator_mapping = {op: idx for idx, op in enumerate(self.operators)}
        self.constant_mapping = {str(const): idx + len(self.operators) for idx, const in enumerate(range(len(self.constants)))}
        self.feature_mapping = {feat: idx + len(self.operators) + len(self.constants) for idx, feat in enumerate(self.features)}
        self.leaves_mapping = {**self.constant_mapping, **self.feature_mapping}

        self.inverse_operator_mapping = {idx: op for op, idx in self.operator_mapping.items()}
        self.inverse_constant_mapping = {idx: const for const, idx in self.constant_mapping.items()}
        self.inverse_feature_mapping = {idx: feat for feat, idx in self.feature_mapping.items()}



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
        
    def encode_state(self, node=None):
        if node is None:
            node = self.tree

        encoded_state = [node.value]

        if node.value in self.operators:
            for child in node.children:
                encoded_state.extend(self.encode_state(child))

        return encoded_state
    
    def encode_state_mapped(self, node=None):
        if node is None:
            node = self.tree

        encoded_state = []
        if node.value in self.operators:
            encoded_state.append(self.operator_mapping[node.value])
            for child in node.children:
                encoded_state.extend(self.encode_state_mapped(child))
        elif node.value in self.constants:
            encoded_state.append(self.constant_mapping[str(node.value)])
        elif node.value in self.features:
            encoded_state.append(self.feature_mapping[node.value])

        return encoded_state

    def take_action(self, node):
        new_value = np.random.choice(self.operators) if node.children else np.random.choice(self.features + self.constants)
        original_value = node.value
        node.value = new_value
        return original_value

    def restore_node(self, node, original_value):
        node.value = original_value

    def get_reward(self, X, y_true):
        y_pred = self.evaluate_tree(self.tree, X)
        return compute_rmse(y_true, y_pred)

    def get_state(self):
        return self.tree
    
    def count_nodes(self, node):
        if node is None:
            return 0
        else:
            return 1 + sum(self.count_nodes(child) for child in node.children)
        
    def get_all_nodes(self, node=None):
        if node is None:
            node = self.tree

        nodes = [node]
        for child in node.children:
            nodes.extend(self.get_all_nodes(child))

        return nodes
        
    def plot_tree(self):
        def plot_node(node):
            if not node.children:
                return str(node.value)
            else:
                return f"({node.value} {plot_node(node.children[0])} {plot_node(node.children[1])})"

        return plot_node(self.tree)
    
    def build_tree_from_encoded_state(self, encoded_state):
        root = Node(None)
        encoded_state = list(encoded_state)
        stack = [(root, encoded_state)]

        while stack:
            current_node, encoded_values = stack.pop()
            if not encoded_values:
                continue

            value = encoded_values.pop(0)
            if value in self.inverse_operator_mapping:
                current_node.value = self.inverse_operator_mapping[value]
                num_children = len(current_node.children)
                while num_children < self.max_children:
                    child = Node(None)
                    current_node.add_child(child)
                    stack.append((child, encoded_values))
                    num_children += 1
            elif value in self.inverse_constant_mapping:
                current_node.value = self.inverse_constant_mapping[value]
            elif value in self.inverse_feature_mapping:
                current_node.value = self.inverse_feature_mapping[value]

        return root


def print_tree_w_indent(node, indent=0):
    print(" " * indent + str(node.value))
    for child in node.children:
        print_tree_w_indent(child, indent + 4)
        
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
    


if __name__ == "__main__":
    df = pd.read_csv('insurance.csv')
    df = pd.get_dummies(df)
    features = ['age', 'bmi', 'children', 'sex_female', 'sex_male',
           'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
           'region_southeast', 'region_southwest']
    target = 'charges'
    X = df[features].values
    y = df[target].values

    constants = list(map(str, range(1, 11)))
    operators = ['+', '-', '*']

    sr = SymbolicRegressor(operators, constants, features, depth=4)

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

    print_tree_w_indent(sr.tree)