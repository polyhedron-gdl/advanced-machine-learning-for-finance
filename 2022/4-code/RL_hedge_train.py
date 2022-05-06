import pandas as pd
import numpy  as np
import random

from RL_finlib   import brownian_sim, bs_call
from tqdm        import tqdm

# Start program
import RL_setting as s
#
# Daily time step
dt = 1/s.T

# Generate time to maturity series
ttm = np.arange(s.M,-1, -1)
#
# Initiate Q Table with Q-value = 0 for all state-action pairs
# M = Option Day to Maturity
q = np.zeros((s.M+1,s.STOCK_PRICE_STATE,s.N_POSITION,s.N_ACTION))
print('Q table shape : ', np.shape(q))
#
# Training Phase    

# Asset paths simulation and BS-Parameters calculation for training phase
price_table = brownian_sim(s.TRAINING_SAMPLE,s.M+1, s.MU, s.VOL, s.S, dt)
call_price_table, delta_table = bs_call(s.VOL, ttm / s.T, price_table, s.K,s.R, s.DIVIDEND)
print("\n Asset path simulation and BS parameter calculation are completed")

if s.TRAINING_SAMPLE <= 10000:
    pd_price_table = pd.DataFrame(price_table)
    pd_call_price  = pd.DataFrame(call_price_table)
    pd_delta_table = pd.DataFrame(delta_table) 
    #
    pd_price_table.to_csv('.\price_table.csv', sep=';')
    pd_call_price.to_csv('.\call_table.csv', sep=';')
    pd_delta_table.to_csv('.\delta_table.csv', sep=';')


# Round asset prices and convert them into the 15 different price state 
# [<=93,94,95,96, ... 103,104,105,106, >= 107] - > [0,1,2,3,4,5,....12,13,14,15]
price_state_table = np.round(price_table).astype(int)
price_state_table = np.where(price_state_table < 93, 93, price_state_table)
price_state_table = np.where(price_state_table > 107, 107, price_state_table)

price_state_table = price_state_table - 93

e = 1
print("\n Training is in progress:")
for i in tqdm(range(len(price_table))):

    asset_price   = price_table[i]
    price_state   = price_state_table[i]
    bscall        = call_price_table[i]
    
    position      = 0
    position_list = np.empty(0, dtype=int)
    reward_list   = np.empty(0, dtype=int)
    action_list   = np.empty(0, dtype=int)
    total_reward  = np.empty(0, dtype=int)
    
    for t in range(s.M):
        if np.random.rand() <= e:
            #
            # EXPLORATION
            #
            # Note that the action is ranging from sell 5 shares to buy 5 shares. 
            # Therefore, the action needs to minus 5 here
            action = random.randrange(0,s.N_ACTION) - 5
        else:
            #
            # EXPLOITATION
            #
            # We assume at time 0, the agent has 0 share in hand
            if t == 0: 
                lookup_position = 0
            else:
                lookup_position = position_list[t-1]
            # If exploitation is triggered, find the minimum non_zero value from the 
            # look up table. Use try function here as the agent can reach a state in 
            # which she never visited before and the state has all Q values as zeros
            # If the agent reach the state that she never reached before, a random 
            # action is taken
            try:
                # np.nonzero Return the indices of the elements that are non-zero.
                non_zero = np.nonzero(q[t][price_state[t]][lookup_position])
                # np.min() function is used to get a minimum value.
                target   = np.min(q[t][price_state[t]][lookup_position][non_zero])
                action   = np.where(q[t][price_state[t]][lookup_position]== target)[0][0] - 5 
            except:
                action = random.randrange(0,s.N_ACTION) - 5
                
        # Forcing the agent to only keep between 0 and 10 shares
        if position + action >10:
            action = 10 - position
            new_position = 10
        elif position + action <0:
            action = -position
            new_position = 0             
        else:
            new_position = action + position 
        
        # Reward is calculated as variance of total pnl from hedging.
        # In "finding delta" case, the reward is essentially minimizing 
        # the variance of total pnl from hedging
        reward = (new_position * (asset_price[t+1] - asset_price[t]) - \
                            10 * (bscall[t+1]      - bscall[t]     ))**2
        
        action_list   = np.append(action_list, action)
        reward_list   = np.append(reward_list, reward)
        position_list = np.append(position_list,new_position)
        position      = new_position
    
    position_list = np.append(position_list,0)
    action_list   = np.append(action_list,0)
    total_reward  = np.append(total_reward,sum(reward_list))
    
    # Convert action list from -5 to 5 to from 0 to 10 as lookup table starts from index 0. 
    # i.e. Index 0 in lookup table means action = -5 and index 10 in lookup table means action = 5
    action_list = action_list + 5
    
    # Update Q table according to the Temporal Difference Learning rule 
    for t in range(s.M):
        if t == 0: 
            lookup_position = 0
        else:
            lookup_position = position_list[t-1]
        #  
        q_t_plus_one = q[t+1][price_state[t+1]][position_list[t]][action_list[t+1]]
        qnew = reward_list[t] + q_t_plus_one
        #
        q_t = q[t][price_state[t]][lookup_position][action_list[t]]
        q[t][price_state[t]][lookup_position][action_list[t]] = \
        q_t + s.ALPHA * (qnew - q_t) 
    
    # Decrease e to make the agent more likely to exploit 
    if e > s.MIN_e:
        e = e * s.DECAY

# Save Q at the end after training 
np.save('Q_RL_Hedging',q)     

print("Done it!")