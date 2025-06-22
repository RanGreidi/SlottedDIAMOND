import numpy as np
import random
import os
from datetime import datetime
import shutil
import copy
import sys
import pickle
import threading
import networkx as nx


# Go 2 levels up from this script to get project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


# sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')

from DIAMOND.diamond import DIAMOND
from DIAMOND.plots.plot_results import plot_algorithm_metrics, plot_algorithm_mean_performance
from DIAMOND.slotted_diamond import SLOTTED_DIAMOND
from DIAMOND.environment import generate_env
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR
from DIAMOND.environment import GraphEnvPower as GraphEnv
from DIAMOND.environment.utils import *
from DIAMOND.environment.utils import keep_gpu_active_heavy
from DIAMOND.environment.Traffic_Probability_HawkesModel import HawkesModel


def create_topology_1():
    """
    Custom topology with 15 nodes and specified edges/positions.
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from(range(15))

    Gbase.add_edges_from([
        (13, 8), (13, 12), (10, 3), (7, 5), (11, 2), (9, 12), (3, 1),
        (12, 14), (6, 9), (14, 9), (4, 9), (7, 0), (4, 1), (4, 2),
        (13, 10), (7, 8), (11, 13), (7, 2), (5, 10), (0, 5), (3, 0),
        (10, 0), (0, 6), (6, 4), (14, 1)
    ])

    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    pos = np.array([
        [91.6, 73.5],
        [83.6, 78.3],
        [45.2, 95.6],
        [39.0, 52.9],
        [71.7, 22.4],
        [51.7, 48.2],
        [19.2, 33.3],
        [82.8, 34.6],
        [36.4, 26.4],
        [29.5, 79.9],
        [59.9, 26.2],
        [73.2, 93.8],
        [40.2, 71.9],
        [62.1, 35.7],
        [26.6, 57.6]
    ])

    flows = [{"source": 6, "destination": 7, "packets": 1, "name": 0},
             {"source": 8, "destination": 10, "packets": 1, "name": 1},
             {"source": 5, "destination": 14, "packets": 1, "name": 2},
             {"source": 5, "destination": 8, "packets": 1, "name": 3},
             {"source": 5, "destination": 1, "packets": 1, "name": 4},
             {"source": 14, "destination": 5, "packets": 1, "name": 5},
             {"source": 2, "destination": 10, "packets": 1, "name": 6},
             {"source": 1, "destination": 0, "packets": 1, "name": 7},
             {"source": 5, "destination": 1, "packets": 1, "name": 8},
             {"source": 13, "destination": 0, "packets": 1, "name": 9},
             ]

    return A, pos, flows


def create_topology_2():
    """
    Custom topology with 15 nodes and specified edges/positions.
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from(range(15))

    Gbase.add_edges_from([
        (3, 9), (2, 0), (4, 12), (10, 14), (9, 12), (4, 3), (9, 13),
        (14, 13), (3, 8), (11, 8), (13, 3), (0, 5), (1, 3), (13, 5),
        (2, 7), (0, 14), (10, 5), (4, 11), (8, 2), (8, 6), (12, 5),
        (2, 11), (14, 7), (4, 0), (10, 11)
    ])

    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    pos = np.array([
        [59.0, 88.7],
        [98.1, 53.7],
        [96.2, 59.4],
        [17.5, 36.8],
        [78.6, 97.9],
        [75.7, 93.5],
        [52.3, 23.9],
        [17.2, 59.4],
        [70.5, 10.4],
        [37.5, 68.3],
        [9.5, 13.7],
        [36.4, 93.2],
        [0.5, 48.4],
        [26.4, 28.3],
        [62.8, 53.8]
    ])

    flows = [{"source": 7, "destination": 0, "packets": 1, "name": 0},
             {"source": 7, "destination": 0, "packets": 1, "name": 1},
             {"source": 7, "destination": 10, "packets": 1, "name": 2},
             {"source": 4, "destination": 10, "packets": 1, "name": 3},
             {"source": 9, "destination": 8, "packets": 1, "name": 4},
             {"source": 5, "destination": 10, "packets": 1, "name": 5},
             {"source": 9, "destination": 2, "packets": 1, "name": 6},
             {"source": 7, "destination": 11, "packets": 1, "name": 7},
             {"source": 10, "destination": 5, "packets": 1, "name": 8},
             {"source": 4, "destination": 0, "packets": 1, "name": 9},
             ]

    return A, pos, flows


def create_topology_3():
    """
    Custom topology with 10 nodes and specified edges/positions.
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from(range(10))

    Gbase.add_edges_from([
        (0, 5), (6, 7), (1, 7), (2, 7), (2, 1), (1, 9), (0, 4), (5, 4),
        (6, 9), (1, 6), (6, 8), (9, 4), (0, 2), (8, 5), (3, 5), (9, 0),
        (3, 2), (3, 0), (6, 4), (2, 9)
    ])

    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    pos = np.array([
        [50.8,  1.0],
        [19.2, 80.7],
        [33.5, 72.4],
        [21.5, 18.6],
        [9.5, 60.6],
        [41.0, 66.3],
        [48.2, 86.9],
        [80.2, 56.3],
        [73.0, 15.9],
        [9.4, 44.9]
    ])

    flows = [{"source": 2, "destination": 0, "packets": 1, "name": 0},
             {"source": 7, "destination": 3, "packets": 1, "name": 1},
             {"source": 1, "destination": 6, "packets": 1, "name": 2},
             {"source": 7, "destination": 6, "packets": 1, "name": 3},
             {"source": 1, "destination": 2, "packets": 1, "name": 4},
             {"source": 7, "destination": 3, "packets": 1, "name": 5},
             {"source": 3, "destination": 5, "packets": 1, "name": 6},
             {"source": 7, "destination": 2, "packets": 1, "name": 7},
             {"source": 2, "destination": 3, "packets": 1, "name": 8},
             {"source": 7, "destination": 5, "packets": 1, "name": 9},
             {"source": 7, "destination": 2, "packets": 1, "name": 10},
             {"source": 5, "destination": 6, "packets": 1, "name": 11},
             ]

    return A, pos, flows


def create_topology_4():
    """
    Custom topology with 11 nodes and specified edges/positions.
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from(range(11))

    Gbase.add_edges_from([
        (6, 4), (5, 2), (1, 3), (3, 2), (4, 1), (5, 10), (8, 5), (2, 4),
        (10, 8), (9, 0), (3, 4), (1, 0), (0, 2), (7, 6), (8, 3), (3, 10), (6, 8)
    ])

    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    pos = np.array([
        [87.0, 93.3],
        [27.8, 33.1],
        [13.0, 6.7],
        [28.6, 33.7],
        [5.3, 5.3],
        [75.0, 41.3],
        [61.2, 57.5],
        [85.2, 17.9],
        [35.4, 26.0],
        [82.5, 15.4],
        [85.4, 58.9]
    ])

    flows = [{"source": 3, "destination": 2, "packets": 1, "name": 0},
             {"source": 6, "destination": 10, "packets": 1, "name": 1},
             {"source": 2, "destination": 4, "packets": 1, "name": 2},
             {"source": 10, "destination": 3, "packets": 1, "name": 3},
             {"source": 2, "destination": 0, "packets": 1, "name": 4},
             {"source": 0, "destination": 7, "packets": 1, "name": 5},
             {"source": 10, "destination": 9, "packets": 1, "name": 6},
             {"source": 5, "destination": 3, "packets": 1, "name": 7},
             {"source": 0, "destination": 2, "packets": 1, "name": 8},
             {"source": 6, "destination": 5, "packets": 1, "name": 9},
             {"source": 3, "destination": 8, "packets": 1, "name": 10},
             {"source": 4, "destination": 8, "packets": 1, "name": 11},
             {"source": 10, "destination": 8, "packets": 1, "name": 12},
             {"source": 0, "destination": 5, "packets": 1, "name": 13},
             ]

    return A, pos, flows


def create_topology_5():
    """
    Custom topology with 30 nodes and specified edges/positions.
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from(range(30))

    Gbase.add_edges_from([
        (10, 11), (5, 12), (27, 12), (1, 6), (7, 28), (1, 0), (4, 24),
        (16, 27), (3, 5), (27, 10), (5, 15), (21, 13), (1, 27), (19, 22),
        (5, 28), (15, 29), (12, 14), (29, 14), (21, 23), (28, 18), (13, 23),
        (15, 13), (23, 17), (4, 11), (26, 1), (2, 8), (29, 6), (25, 27),
        (10, 24), (16, 0), (2, 23), (28, 12), (28, 14), (21, 2), (26, 2),
        (19, 14), (2, 0), (15, 8), (27, 20), (7, 17), (20, 7), (24, 20),
        (10, 5), (17, 6), (7, 23), (8, 0), (12, 9), (24, 7), (24, 16), (2, 15)
    ])

    pos = np.array([
        [26.8, 34.0], [7.9, 28.4], [2.6, 83.1], [79.3, 76.9], [37.6, 31.9],
        [23.3, 81.4], [20.1, 97.9], [32.0, 41.4], [23.5, 63.1], [43.6, 62.6],
        [89.8, 72.3], [80.8, 14.6], [33.9, 49.6], [44.8, 52.9], [92.7, 0.7],
        [64.5, 7.3], [61.2, 76.1], [38.1, 58.5], [82.0, 81.7], [86.4, 71.5],
        [32.4, 93.2], [40.6, 11.1], [48.1, 32.9], [53.4, 48.5], [69.1, 43.4],
        [56.0, 97.5], [50.9, 9.3], [23.2, 51.2], [33.3, 10.3], [37.7, 96.0]
    ])

    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    flows = [{"source": 17, "destination": 8, "packets": 1, "name": 0},
             {"source": 11, "destination": 14, "packets": 1, "name": 1},
             {"source": 20, "destination": 17, "packets": 1, "name": 2},
             {"source": 8, "destination": 25, "packets": 1, "name": 3},
             {"source": 9, "destination": 22, "packets": 1, "name": 4},
             {"source": 12, "destination": 21, "packets": 1, "name": 5},
             {"source": 17, "destination": 26, "packets": 1, "name": 6},
             {"source": 21, "destination": 8, "packets": 1, "name": 7},
             {"source": 26, "destination": 7, "packets": 1, "name": 8},
             {"source": 24, "destination": 27, "packets": 1, "name": 9},
             {"source": 1, "destination": 19, "packets": 1, "name": 10},
             {"source": 2, "destination": 4, "packets": 1, "name": 11},
             {"source": 8, "destination": 15, "packets": 1, "name": 12},
             {"source": 5, "destination": 29, "packets": 1, "name": 13},
             {"source": 9, "destination": 27, "packets": 1, "name": 14},
             {"source": 23, "destination": 1, "packets": 1, "name": 15},
             {"source": 22, "destination": 25, "packets": 1, "name": 16},
             {"source": 7, "destination": 9, "packets": 1, "name": 17},
             {"source": 17, "destination": 20, "packets": 1, "name": 18},
             {"source": 23, "destination": 10, "packets": 1, "name": 19},
             {"source": 22, "destination": 12, "packets": 1, "name": 20},
             {"source": 22, "destination": 20, "packets": 1, "name": 21},
             ]

    return A, pos, flows


def create_topology_6():
    """
    Custom topology with 5 nodes and specified edges/positions.
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from(range(5))

    Gbase.add_edges_from([
        (1, 3), (0, 1), (0, 4), (1, 2), (3, 0),
        (2, 3), (3, 4), (1, 4), (2, 0), (4, 2)
    ])

    pos = np.array([
        [86.6, 78.2],
        [54.2, 31.4],
        [88.9, 44.5],
        [98.3, 33.3],
        [91.3, 73.2]
    ])

    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    flows = [{"source": 3, "destination": 4, "packets": 1, "name": 0},
             {"source": 1, "destination": 3, "packets": 1, "name": 1},
             {"source": 2, "destination": 1, "packets": 1, "name": 2},
             {"source": 3, "destination": 1, "packets": 1, "name": 3},
             {"source": 2, "destination": 0, "packets": 1, "name": 4},
             {"source": 2, "destination": 3, "packets": 1, "name": 5},
             {"source": 0, "destination": 4, "packets": 1, "name": 6},
             {"source": 1, "destination": 4, "packets": 1, "name": 7},
             ]

    return A, pos, flows


def get_env(topology, env_params):
    np.random.seed(env_params['seed'])

    if topology == "1":
        A, pos, flows = create_topology_1()
        num_nodes = 15
    elif topology == "2":
        A, pos, flows = create_topology_2()
        num_nodes = 15
    elif topology == "3":
        A, pos, flows = create_topology_3()
        num_nodes = 10
    elif topology == "4":
        A, pos, flows = create_topology_4()
        num_nodes = 11
    elif topology == "5":
        A, pos, flows = create_topology_5()
        num_nodes = 30
    elif topology == "6":
        A, pos, flows = create_topology_6()
        num_nodes = 5

    delta=10
    packets = list(range(int(env_params['min_flow_demand']), int(env_params['max_flow_demand']) + delta, delta))
    capacity_matrix = np.random.randint(low=env_params['min_capacity'], high=env_params['max_capacity'] + 1, size=A.shape)
    # ------------------ Get flows statistics from Hawkes Model ---------- #
    HawkesParams = env_params["HawkesParams"]
    flows_statistics = []
    for flow in flows:
        src, dst, name = flow['source'], flow['destination'], flow['name']

        flow_statistics = HawkesModel(  # alpha * exp(-beta*t)
                                        lambda0=HawkesParams['lambda0'],
                                        alpha=HawkesParams['alpha'],
                                        beta=HawkesParams['beta'],

                                        source=src,
                                        destination=dst,
                                        flow_name=name,
                                        num_slots=num_slots,
                                        slot_duration=slot_duration,
                                        history_num_slots=HawkesParams['history_num_slots'],
                                        pkt_arrival_sample_rate=pkt_arrival_sample_rate,

                                        type='elephent' if name < HawkesParams['elephent_flows_num'] else 'mice',  #  type='elephent' if name < HawkesParams['elephent_flows_num'] else 'mice, random.choice(types)
                                        mice_scaler=HawkesParams['mice_scaler'],
                                        elephent_scaler=HawkesParams['elephent_scaler'],

                                        ManualAdded_Fixed_InitalPkts=HawkesParams['ManualAdded_Fixed_InitalPkts'],
                                        seed=env_params['seed'])

        # flow['packets'] = flow_statistics.initial_count if HawkesParams['allow_Hawkes_arrivals'] else None
        flow['packets'] = random.choice(packets)
        flows_statistics.append(flow_statistics)
    # --------------------------------------------------------------------- #
    # capacity_matrix = 400 * np.ones(shape=A.shape)  # might change to get different capacity
    interference_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    env = GraphEnv(
        adjacency_matrix=A,
        bandwidth_matrix=capacity_matrix,
        interference_matrix=interference_matrix,
        node_positions=pos,
        flows=flows,
        **env_params
    )

    env_configurations = dict(adjacency_matrix=A,
                                bandwidth_matrix=capacity_matrix,
                                interference_matrix=interference_matrix,
                                node_positions=pos,
                                flows=flows,
                                **env_params)

    return env, env_configurations, flows_statistics


def create_initial_slot_data():
        data = {
            'SlotedDIAMOND_active_flows': None,
            'SlotedDIAMOND_delay': 0,
            'SlotedDIAMOND_rates': 0,
        }
        return data


def generate_flows_with_fixed_pkt(flows, pkt_size):
        """
        :param flows: List of dictionaries with keys 'source', 'destination', and 'packets'
        :param pkt_size: The fixed value to set for the 'packets' key
        :return: New list of dictionaries with 'packets' key set to fixed_pkt_value
        """
        new_flows = []
        for flow in flows:
            if flow['packets'] > 0:
                new_flow = {
                    'source': flow['source'],
                    'destination': flow['destination'],
                    'packets': pkt_size,
                    'name': flow['name']
                }
                new_flows.append(new_flow)
        return new_flows


def create_slot_flows(pkt_size, Algos_Global_flows):
        '''
        input: takes as input global flows for each algo
        output: flows list for each algo for a signle slot

        This function is used to create flows for each algo for a single slot.
        if the algo has one less flow to allocate, the slot_flows needs to be updated accordingly
        it counts how many flows left for each algo and creates flows for each algo for a single slot with a fixes packet size (demand)
        '''
        new_Algos_slot_flows = {}
        pkt_size = pkt_size
        for algo, flows in Algos_Global_flows.items():
            new_Algos_slot_flows[algo] = generate_flows_with_fixed_pkt(flows, pkt_size)  # Replace self.modify_flow(flow) with your desired operation
        return new_Algos_slot_flows


def update_Global_flows(Algos_Global_flows, flows_statistics, slot_data, slot,
                        pkt_arrival_sample_rate,algos, units,slot_duration,predictor_mode ):
        '''
        input:  1. flows list for each algo according to its current state
                2. rate and delay data for each algo
        output: updated flow list for each algo

        This function takes in the current state of the flows for each algo and the rate and delay data for each algo, and updates the flows packets for each algo
        according to the performance of each algo in the previous slot

        in the future, flows that needs to be added in a slot will be added here.

        Units:
        slot_duration [sec]
        rate [Mbps]
        initial_delay [micro sec]
        BW [MHz]
        delivered_packets [Megabit]
        '''

        # TODO: Add packets only if incoming demand

        # adding flow pkts according to arrival statistics
        if predictor_mode == "predictor_off":
            pass
        else:
            if slot % pkt_arrival_sample_rate == 0:  # and slot != 0:
                for flow_statistic in flows_statistics:
                    entered_new_pkts = flow_statistic.step(slot=slot)
                    flow_name = flow_statistic.flow_name
                    for algo in algos:
                        flow = get_flow_by_name(Algos_Global_flows[algo], flow_name)
                        flow['packets'] += entered_new_pkts

        # removing flow pkts according to arrvied pkts
        for algo in algos:
            algo_active_flows = slot_data[f"{algo}_active_flows"]
            algo_delay = slot_data[f"{algo}_delay"]
            algo_rate = slot_data[f"{algo}_rates"]

            # how many packets will be delivered in slot_duration
            units = units  # 1e6
            initial_delay = algo_delay
            #  -[Megabit]-       -[Mbps]-       ------[microsec]-----    --[micro sec]--     -[micro sec]-
            delivered_packets = algo_rate * (slot_duration * units - initial_delay) / units  # rate [Mbps] * (slot_duration [micro sec])/microsec

            # reduce deliver packets from the flows
            if algo_active_flows:
                for idx_in_metrics_for_flow, flow_name in enumerate(algo_active_flows):
                    flow = get_flow_by_name(Algos_Global_flows[algo],
                                            flow_name)  # flow is a pointer to the current flow in the Algos_Global_flows
                    if flow['packets'] <= delivered_packets[idx_in_metrics_for_flow]:
                        flow['packets'] = 0
                    else:
                        flow['packets'] -= delivered_packets[idx_in_metrics_for_flow]
        return Algos_Global_flows


def create_initial_run_data(num_slots):

    data = {
        'SlotedDIAMOND_delay': np.zeros(num_slots),
        'SlotedDIAMOND_rates': np.zeros(num_slots),
        'SlotedDIAMOND_active_flows': np.zeros(num_slots),

    }

    return data


def prepare_one_end_of_run_data(full_run_data, num_slots, algos):
    Episode_Avarge_data = create_initial_run_data(num_slots=num_slots)
    for slot in range(num_slots):
        for c in algos:
            Episode_Avarge_data[f"{c}_delay"][slot] = np.average(full_run_data[slot][f"{c}_delay"])
            Episode_Avarge_data[f"{c}_rates"][slot] = np.average(full_run_data[slot][f"{c}_rates"])
            Episode_Avarge_data[f"{c}_active_flows"][slot] = len(full_run_data[slot][f"{c}_active_flows"]) if \
            full_run_data[slot][f"{c}_active_flows"] else 0

    return Episode_Avarge_data


def main(topology, env_params, grrl_model_path, slot_duration,num_slots,pkt_size,predictor_mode,
         pkt_arrival_sample_rate, temperature, nb3r_steps):

    # -------------------- Initialization -------------- #
    algos = ['SlotedDIAMOND']
    first_step_actions = {algo: [] for algo in algos}
    # -------------------------------------------------- #



    # Algos defenitions
    slotted_diamond = SLOTTED_DIAMOND(grrl_model_path=grrl_model_path,
                                           nb3r_tmpr=temperature,
                                           nb3r_steps=0,
                                           slot_duration=slot_duration,
                                           num_slots=num_slots,
                                           pkt_size=pkt_size,
                                           predictor_mode=predictor_mode,
                                           pkt_arrival_sample_rate=pkt_arrival_sample_rate)

    # diamond = DIAMOND(grrl_model_path=grrl_model_path, nb3r_tmpr=temperature,
    #                        nb3r_steps=nb3r_steps)

    Gloval_env, env_configurations, flows_statistics = get_env(topology=topology, env_params=env_params)

    # Run Slotted DIAMOND
    _, first_step_actions['SlotedDIAMOND'], all_slotted_paths = slotted_diamond(copy.deepcopy(Gloval_env),
                                                             env_configurations, flows_statistics,
                                                             grrl_data=True)
    # Run DIAMOND
    # print(f"Started DIAMOND \n")
    # _, _, _, _, first_step_actions['DIAMOND'] = diamond(copy.deepcopy(Gloval_env), grrl_data=True)
    # print(f"Finished DIAMOND \n")

    # initialize Global flows list for each algo (entire run flows)
    Algos_Global_flows = {
        'SlotedDIAMOND': copy.deepcopy(env_configurations['flows']),
        # 'DIAMOND': copy.deepcopy(env_configurations['flows']),
    }

    full_run_data = []

    print(f'\nstarting real run\n')

    for slot in range(num_slots):

        # initialize data for slot
        slot_data = create_initial_slot_data()

        # create flows for slot for each algo
        algos_slot_flows = create_slot_flows(pkt_size, Algos_Global_flows)

        # Create env for slot
        Algos_step_envs = {algo: GraphEnv(adjacency_matrix=env_configurations['adjacency_matrix'],
                                          bandwidth_matrix=env_configurations['bandwidth_matrix'],
                                          interference_matrix=env_configurations['interference_matrix'],
                                          node_positions=env_configurations['node_positions'],
                                          flows=algos_slot_flows[algo],
                                          k=env_configurations['k'],
                                          direction=env_configurations['direction'],
                                          reward_balance=env_configurations['reward_balance'],
                                          algo=algo,
                                          seed=env_configurations['seed'],
                                          # num_nodes=Gloval_env.num_nodes,
                                          # num_edges=Gloval_env.num_edges,
                                          # num_flows=Gloval_env.num_flows,
                                          trx_power_mode=env_params['trx_power_mode'],
                                          rayleigh_scale=env_params['rayleigh_scale'],
                                          max_trx_power=env_params['max_trx_power'],
                                          channel_gain=env_configurations['channel_gain'],
                                          )

                           for algo in algos}

        print(f'slot {slot}, {len(Algos_step_envs["SlotedDIAMOND"].flows)}/{len(Gloval_env.flows)} flows alive')

        # -----Run ALGOS------

        # Run SlottedDIAMOND
        if Algos_step_envs['SlotedDIAMOND'].flows:
            SlotedDIAMOND_rates_data, SlotedDIAMOND_delay_data = run_Slotted_predefined_actions(
                Algos_step_envs['SlotedDIAMOND'], first_step_actions['SlotedDIAMOND'], slot)
            slot_data['SlotedDIAMOND_active_flows'] = [flow['name'] for flow in Algos_step_envs['SlotedDIAMOND'].flows]
            slot_data['SlotedDIAMOND_delay'] = SlotedDIAMOND_delay_data['delay_per_flow']
            slot_data['SlotedDIAMOND_rates'] = SlotedDIAMOND_rates_data['rate_per_flow']


        # Update Algos_Global_flows according to preformance of each algo
        Algos_Global_flows = update_Global_flows(Algos_Global_flows, flows_statistics, slot_data, slot,
                                                 pkt_arrival_sample_rate=pkt_arrival_sample_rate,
                                                 algos=algos,
                                                 units=units,
                                                 slot_duration=slot_duration,predictor_mode=predictor_mode)

        full_run_data.append(slot_data)

        print(f"Finished slot {slot + 1}/{num_slots} In Real Slotted_DIAMOND \n")

    this_Episode_Avarge_data = prepare_one_end_of_run_data(full_run_data, num_slots=num_slots, algos=algos,)

    # plot individual episode results.
    save = True
    subfolder_path = plot_algorithm_metrics(this_Episode_Avarge_data,
                                            Gloval_env=Gloval_env,
                                            graph_mode=f'topology_{topology}', save_fig=save)
    # save generate_env args
    file_path = os.path.join(subfolder_path, "generate_env_args.json")
    save_arguments_to_file(filename=file_path, args=env_args)
    # save Hawks model params
    file_path = os.path.join(subfolder_path, "Hawkes_params.json")
    save_arguments_to_file(filename=file_path, args=HawkesParams)
    # save all_paths
    file_path = os.path.join(subfolder_path, "all_slotted_paths.json")
    save_arguments_to_file(filename=file_path, args=all_slotted_paths)
    # save link capacities for NS3
    file_path = os.path.join(subfolder_path, "link_capacities.json")
    capacity_list = [int(x) for x in list(Gloval_env.bandwidth_edge_list)]
    save_arguments_to_file(filename=file_path, args=capacity_list)
    # save flows for NS3
    file_path = os.path.join(subfolder_path, "flows.json")
    flows_list = Gloval_env.flows
    save_arguments_to_file(filename=file_path, args=flows_list)
    # save graph image
    Gloval_env.show_graph(save_path=os.path.join(subfolder_path, "graph.png"), show_fig=False)

    return all_slotted_paths


if __name__ == "__main__":

    BASE_PATH = os.path.join("..", "results", "inference_vs_competitors")
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.abspath(__file__)

    # general params
    num_nodes = 80  # 60
    num_edges = 120  # 90
    num_actions = 15  # 15, 4
    temperature = 1.2
    num_episodes = 3  # 3
    episode_from = 7500  # 7500  # 7501
    nb3r_steps = 20  # 1, 100

    trx_power_mode = 'equal'
    rayleigh_scale = 1
    max_trx_power = 10
    channel_gain = 1
    min_capacity = 200  # 200
    max_capacity = 500  # 500
    min_flow_demand = 500   # 5, 500
    max_flow_demand = 5000  # 200, 2000

    pkt_size = 500  # 500, 100
    units = 1e6

    slot_duration = 1
    num_slots = 50  # 150

    pkt_arrival_sample_rate = 10  # 5

    topology = "6"

    # Hawkes parms
    HawkesParams = dict(
        lambda0=0.005,  # 0.9
        alpha=0.025,  # 0.5 0.4
        beta=0.0001,  # 0.7 0.8
        history_num_slots=200,  # 100
        allow_Hawkes_arrivals=True,
        elephent_flows_num=10,
        mice_scaler=5,  # 0.1, 50, 150, 20, 5
        elephent_scaler=5, # 5
        ManualAdded_Fixed_InitalPkts=100)  # 0.1, 100, 350, 50

    # predictor params
    predictor_mode = 'predictor_off'  # 'predictor_on' # 'predictor_off', 'Ideal'

    env_args = dict(k=num_actions,
                    direction="minimize",
                    reward_balance=0.8,
                    seed=episode_from,
                    pkt_size=pkt_size,
                    units=units,
                    pkt_arrival_sample_rate=pkt_arrival_sample_rate,
                    min_capacity=min_capacity,
                    max_capacity=max_capacity,
                    min_flow_demand=min_flow_demand,
                    max_flow_demand=max_flow_demand,
                    trx_power_mode=trx_power_mode,
                    rayleigh_scale=rayleigh_scale,
                    max_trx_power=max_trx_power,
                    channel_gain=channel_gain,
                    HawkesParams=HawkesParams,)

    # ---------------------- Run Chosen Topology ------------- #
    all_slotted_paths = main(topology=topology,
                             env_params=env_args,
                             grrl_model_path=MODEL_PATH,
                             slot_duration=slot_duration,
                             num_slots=num_slots,
                             predictor_mode=predictor_mode,
                             pkt_arrival_sample_rate=pkt_arrival_sample_rate,
                             pkt_size=pkt_size,
                             temperature=temperature,
                             nb3r_steps=nb3r_steps)

    print('finished')