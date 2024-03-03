# SymbolicRL

## Description

The purpose of this project is to explore the use of reinforcement learning for symbolic regression. Symbolic regression is an optimization problem, with well-known algorithms to handle it (e.g., genetic programming, simulated annealing), and reinforcement learning may not be the best solution. However, algorithms like Q-learning might highlight some interesting behaviors during the exploratory phase.


## Installation

Install the conda environment:
```
conda env create -f environment.yml
```

Then:
```
conda activate symbolic-rl
```

## TODO

* compute the RMSE on test dataset
* better way to stop the training process
* look at the actions cumulative q values to have a prior for the exploration stage
* analyze the Q table