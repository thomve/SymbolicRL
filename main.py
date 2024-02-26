import numpy as np

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def evaluate_tree(node, X):
    if node.value in operators:
        operator = node.value
        operands = [evaluate_tree(child, X) for child in node.children]
        if operator == '+':
            return np.sum(operands, axis=0)
        elif operator == '-':
            return operands[0] - np.sum(operands[1:], axis=0)
        elif operator == '*':
            return np.prod(operands, axis=0)
        elif operator == '/':
            return operands[0] / np.prod(operands[1:], axis=0)
    elif node.value in features:
        return X[:, features.index(node.value)]
    elif node.value.isdigit():
        return float(node.value) * np.ones(len(X))
    else:
        raise ValueError("Invalid node value")

def generate_random_tree(depth):
    if depth == 0:
        return Node(np.random.choice(features + constants))
    else:
        node = Node(np.random.choice(operators))
        for _ in range(np.random.randint(2, max_children+1)):
            node.add_child(generate_random_tree(depth - 1))
        return node

def print_tree(node, indent=0):
    print(" " * indent + str(node.value))
    for child in node.children:
        print_tree(child, indent + 4)

def compute_output(true_formula, X):
    outputs = []
    for x in X:
        y = eval(true_formula, {'__builtins__': None}, {'x': x[0], 'y': x[1]})  # Evaluate the formula for each row of X
        outputs.append(y)
    return np.array(outputs)

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def temporary_modify_node(node):
    new_value = np.random.choice(operators) if node.children else np.random.choice(features + constants)
    original_value = node.value
    node.value = new_value
    return original_value

def restore_node(node, original_value):
    node.value = original_value  # Restore the original value


# Define operators, features, and constants
operators = ['+', '-', '*', '/']
features = ['x', 'y']
constants = list(map(str, range(1, 11)))
max_children = 2

# Generate synthetic dataset
X = np.random.rand(10, len(features))  # Example input features
true_coefficients = np.random.randint(-5, 5, size=len(features))  # Example coefficients
true_formula = ' + '.join([f'{true_coefficients[i]}*{feature}' for i, feature in enumerate(features)])  # True formula
y_true = compute_output(true_formula, X)


random_tree = generate_random_tree(depth=3)  # Depth of the parse tree

num_step = 1000
for i in range(num_step):
    y_pred = evaluate_tree(random_tree, X)
    rmse_before = compute_rmse(y_true, y_pred)

    selected_node_index = np.random.choice(range(len(random_tree.children)))  # Randomly select a child node index
    selected_node = random_tree.children[selected_node_index]

    original_value = temporary_modify_node(selected_node)
    temporary_y_pred = evaluate_tree(random_tree, X)

    rmse_after = compute_rmse(y_true, temporary_y_pred)

    # Check if temporary modification improves performance
    if rmse_before < rmse_after:
        restore_node(selected_node, original_value)
    else:
        rmse_before = rmse_after

print_tree(random_tree)