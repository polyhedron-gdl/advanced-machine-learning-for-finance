'''
Created on 19 apr 2022

@author: User
'''
# Define number of paths being generated for training and testing 
TRAINING_SAMPLE = 3000000
TESTING_SAMPLE = 100000

# Define the characteristic of the underlying stock process
# Annual Return for stock
MU = 0
# Annual Volatility
VOL = 0.2
# Initial Asset Value
S = 100
# Annual Risk Free Rate 
R = 0
# Annual Dividend
DIVIDEND = 0
# Annual Trading Day
T = 250

# Define the Call Option
# Option Strike Price
K = 100
# Option Day to Maturity
M = 10
# Number of possible positions [from 0 to 10] 
N_POSITION = 11
# Action [from -5 to 5]
N_ACTION   = 11

# Define variables for reinforcement learning training
# Constant for reward function
constant = 1
# Min epsilon
MIN_e = 0.05
# Intial epsilon:
e = 1
# Decay
DECAY = 0.999999

# Parameters for Q table update
ALPHA = 0.01
# Set number of state for stock price. Prices are round to the nearest integer
# Prices greater or equal to 107 are assumed to be the same state
# Prices smaller or equal to 93 are assumed to be the same state
STOCK_PRICE_STATE = 15 
