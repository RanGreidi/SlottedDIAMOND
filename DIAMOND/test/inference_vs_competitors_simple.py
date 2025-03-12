import numpy as np
import random
import os
from datetime import datetime
import shutil


import sys
sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')
from environment import GraphEnvPower
from diamond import DIAMOND
from environment import generate_env
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


if __name__ == "__main__":
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)

    # ------------------------------------------------------------------------
    Simulation_Time_Resolution = 1e-1       # miliseconds (i.e. each time step is a milisecond - this is the duration of each time step in [SEC])
    BW_value_in_Hertz = 1e6                   # wanted BW in Hertz
    slot_duration = 1                     # [SEC] 
    Tot_num_of_timeslots = 20000               # [num of time slots]
    #------------------------------------------------------------------------

    # # number of nodes
    # N = 4

    # # Adjacency matrix
    # # create 3x3 mesh graph
    # A = np.array([[0, 1, 1, 1],  #means how connects to who
    #               [1, 0, 1, 1],
    #               [1, 1, 0, 1],
    #               [1, 1, 1, 0]])

    # # node positions
    # P = [(0, 0), (0, 1),                 #the position of each node
    #      (1, 0), (1, 1)] 

    # # BW matrix [MHz]
    # C = 1 * np.ones((N, N))
    # C = 1 * np.array([  [1, 1, 1, 1],  #means how connects to who
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1]])
    #------------------------------------------------------------------------
   
    N = 9

    # Adjacency matrix
    
    # A = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
    #               [1, 0, 1, 0, 1, 0, 0, 0, 0],
    #               [0, 1, 0, 0, 0, 1, 0, 0, 0],
    #               [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #               [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #               [0, 0, 1, 0, 1, 0, 0, 0, 1],
    #               [0, 0, 0, 1, 0, 0, 0, 1, 0],
    #               [0, 0, 0, 0, 1, 0, 1, 0, 1],
    #               [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    
    A = np.array([[0, 1, 0, 1, 1, 1, 0, 1, 0],
                 [1, 0, 1, 1, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 1, 0, 0, 0],
                 [1, 1, 0, 0, 1, 0, 1, 0, 0],
                 [1, 1, 0, 1, 0, 1, 0, 1, 0],
                 [1, 0, 1, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 1, 0, 1, 0, 1],
                 [0, 0, 0, 0, 0, 1, 0, 1, 0]])   
    
    # P = [(0.0, 0), (0.0, 0.01), (0.0, 0.02),
    #      (0.1, 0), (0.1, 0.01), (0.1, 0.02),
    #      (0.2, 0), (0.2, 0.01), (0.2, 0.02)]
   
    P = [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 1), (1, 2),
         (2, 0), (2, 1), (2, 2)]
    
    # --------
    # random_matrix = np.random.randint(2, size=(9, 9))
    # A = np.triu(random_matrix) + np.triu(random_matrix, 1).T
    # np.fill_diagonal(A, 0)
    # P = [(random.random(), random.random()) for _ in range(9)]
    # --------

    # BW matrix
    C = BW_value_in_Hertz * np.ones((N, N)) * Simulation_Time_Resolution
    #------------------------------------------------------------------------


    # number of paths to choose from
    action_size = 100                      #search space limitaions?

    # flow demands in KiloByte
    # F = [
    #     {"source": 0, "destination": 8, "packets": 3  *1e6    , "time_constrain": 10 , 'flow_idx': 0 , 'constant_flow_name': 0},
    #     {"source": 1, "destination": 8, "packets": 10 *1e6    , "time_constrain": 10, 'flow_idx': 1, 'constant_flow_name': 1},         #Packets [in Bits] 
    #     {"source": 3, "destination": 8, "packets": 30 *1e6    , "time_constrain": 10, 'flow_idx': 2, 'constant_flow_name': 2},         #Packets [in Bits]   
    #     {"source": 2, "destination": 8, "packets": 1000 *1e6    , "time_constrain": 10, 'flow_idx': 4, 'constant_flow_name': 3},
    #     {"source": 0, "destination": 8, "packets": 3  *1e6    , "time_constrain": 10 , 'flow_idx': 5 , 'constant_flow_name': 4},
    #     {"source": 1, "destination": 8, "packets": 5 *1e6    , "time_constrain": 10, 'flow_idx': 6, 'constant_flow_name': 5},        
    # ]

    F = [
        {"source": 0, "destination": 8, "packets": 3    *1e6, "time_constrain": 10, 'flow_idx': 0 , 'constant_flow_name': 0},
        {"source": 0, "destination": 7, "packets": 3    *1e6, "time_constrain": 10, 'flow_idx': 1 , 'constant_flow_name': 1},         #Packets [in Bits]
        {"source": 0, "destination": 6, "packets": 3    *1e6, "time_constrain": 10, 'flow_idx': 2 , 'constant_flow_name': 2},         #Packets [in Bits]
        {"source": 0, "destination": 5, "packets": 18   *1e6, "time_constrain": 10, 'flow_idx': 3 , 'constant_flow_name': 3},
        {"source": 0, "destination": 4, "packets": 15   *1e6, "time_constrain": 10, 'flow_idx': 4 , 'constant_flow_name': 4},
        {"source": 0, "destination": 3, "packets": 3    *1e6, "time_constrain": 10, 'flow_idx': 5 , 'constant_flow_name': 5},         #Packets [in Bits]
        {"source": 0, "destination": 2, "packets": 3    *1e6, "time_constrain": 10, 'flow_idx': 6 , 'constant_flow_name': 6},         #Packets [in Bits]
        {"source": 0, "destination": 1, "packets": 15   *1e6, "time_constrain": 10, 'flow_idx': 7 , 'constant_flow_name': 7},
        {"source": 0, "destination": 8, "packets": 30    *1e6, "time_constrain": 10, 'flow_idx': 8 , 'constant_flow_name': 8},
        {"source": 0, "destination": 7, "packets": 1000    *1e6, "time_constrain": 10, 'flow_idx': 9 , 'constant_flow_name': 9},         #Packets [in Bits]
        {"source": 0, "destination": 6, "packets": 3000    *1e6, "time_constrain": 10, 'flow_idx': 10 , 'constant_flow_name': 10},         #Packets [in Bits]
        {"source": 0, "destination": 5, "packets": 180   *1e6, "time_constrain": 10, 'flow_idx': 11 , 'constant_flow_name': 11},
        {"source": 0, "destination": 4, "packets": 150   *1e6, "time_constrain": 10, 'flow_idx': 12 , 'constant_flow_name': 12},
        {"source": 0, "destination": 3, "packets": 300    *1e6, "time_constrain": 10, 'flow_idx': 13 , 'constant_flow_name': 13},         #Packets [in Bits]
        {"source": 0, "destination": 2, "packets": 300    *1e6, "time_constrain": 10, 'flow_idx': 14 , 'constant_flow_name': 14},         #Packets [in Bits]
        {"source": 0, "destination": 1, "packets": 150   *1e6, "time_constrain": 10, 'flow_idx': 15 , 'constant_flow_name': 15}
    ]


    env = GraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        interference_matrix=None,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = True,
                                        direction = 'minimize',
                                        render_mode = True)

    slotted_diamond = DIAMOND(grrl_model_path=MODEL_PATH)
    
    diamond_paths, slotted_grrl_rates_data, slotted_grrl_delay_data = slotted_diamond(env, grrl_data=True)

