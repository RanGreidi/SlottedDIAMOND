import numpy as np
import random
import os
from datetime import datetime
import shutil
import numpy as np
import copy
import sys
import pickle
import threading


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

# -------------- Start Cluster GPU ping thread in background for avoiding IDLE state ------------- #
using_cluster = True
if torch.cuda.is_available() and using_cluster:
    keep_gpu_active_heavy(interval_seconds=60)  # Try 60 seconds
# ------------------------------------------------------------------------------------------------ #

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


class TestvsCompetitors:
    def __init__(self,
                 grrl_model_path,
                 num_episodes=100,
                 num_rb_trials=1,
                 run_competition=True,
                 **kwargs):

        self.num_episodes = num_episodes

        self.episode_from = kwargs.get('episode_from', 0)
        self.slot_duration = kwargs.get('slot_duration', 1)
        self.num_slots = kwargs.get('num_slots', 100)
        self.pkt_size = kwargs.get('pkt_size', 100)
        self.units = kwargs.get('units', 1e6)
        self.predictor_mode = kwargs.get('predictor_mode', 'Ideal')
        self.pkt_arrival_sample_rate = kwargs.get('pkt_arrival_sample_rate', 1)
        self.HawkesParams = kwargs.get('HawkesParams', {})
        self.refinement_steps = kwargs.get('refinement_steps', 10)

        self.algos = ['SlotedDIAMOND', 'DIAMOND', 'GRRL', 'DQN+GNN', 'OSPF', 'RandomBL', 'DIAR', 'IACR'] if run_competition else ['SlotedDIAMOND']
        self.num_of_algos = len(self.algos)
        self.first_step_actions = {algo: [] for algo in self.algos}

        # Algos defenitions
        self.slotted_diamond = SLOTTED_DIAMOND(grrl_model_path=grrl_model_path,
                                               nb3r_tmpr=kwargs.get('nb3r_tmpr', 1),
                                               nb3r_steps=self.refinement_steps,  # 0,
                                               slot_duration=self.slot_duration,
                                               num_slots=self.num_slots,
                                               pkt_size=self.pkt_size,
                                               predictor_mode=self.predictor_mode,
                                               pkt_arrival_sample_rate=self.pkt_arrival_sample_rate)

        self.diamond = DIAMOND(grrl_model_path=grrl_model_path, nb3r_tmpr=kwargs.get('nb3r_tmpr', 1),
                               nb3r_steps=kwargs.get('nb3r_steps', 10))

        self.grrl = DIAMOND(grrl_model_path=grrl_model_path, nb3r_tmpr=kwargs.get('nb3r_tmpr', 1),
                            nb3r_steps=0)

        self.competitors = {
            'DQN+GNN': DQN_GNN(k=4),
            'OSPF': OSPF(),
            'RandomBL': RandomBaseline(num_trials=num_rb_trials),
            'DIAR': DIAR(n_iter=2),
            'IACR': IACR(delta=0.5, alpha=1.3),
        }

        # to monitor if you want to calculate competition
        self.run_competition = run_competition

    def __call__(self, data_paths_list=None, **kwargs):

        # update variables
        self.num_nodes = kwargs.get('num_nodes', 10)
        self.num_edges = kwargs.get('num_edges', 20)
        self.num_flows = kwargs.get('num_flows', 20)
        self.num_actions = kwargs.get('num_actions', 4)

        Episode_Avarge_data = self.create_initial_run_data()

        average_data_through_time = self.create_initial_mean_through_time_data(Episode_Avarge_data)

        for episode in range(self.num_episodes):

            if not data_paths_list:

                print(f'\nStarting episode {episode + 1}/{self.num_episodes}\n')

                # seed
                seed = SEED + (episode + 1) + self.episode_from + 1
                print(f'seed: {seed}')

                env_args = dict(num_nodes=self.num_nodes,
                                num_edges=self.num_edges,
                                num_actions=self.num_actions,
                                num_flows=self.num_flows,
                                slot_duration=self.slot_duration,
                                num_slots=self.num_slots,
                                pkt_size=self.pkt_size,
                                units=self.units,
                                pkt_arrival_sample_rate=self.pkt_arrival_sample_rate,
                                min_flow_demand=kwargs.get('min_flow_demand', 500),  # 500, 200
                                max_flow_demand=kwargs.get('max_flow_demand', 1500),  # 3000, 1000
                                min_capacity=kwargs.get('min_capacity', 200),  # 200, 100
                                max_capacity=kwargs.get('max_capacity', 500),  # 500, 200
                                seed=seed,
                                graph_mode=kwargs.get('graph_mode', 'random'),
                                trx_power_mode=kwargs.get('trx_power_mode', 'equal'),
                                rayleigh_scale=kwargs.get('rayleigh_scale'),
                                max_trx_power=kwargs.get('max_trx_power'),
                                channel_gain=kwargs.get('channel_gain'))

                # generate env
                Gloval_env, env_configurations, flows_statistics = generate_env(**env_args,
                                                                                HawkesParams=self.HawkesParams,
                                                                                )

                # generate first decisions

                # Run Slotted DIAMOND
                _, self.first_step_actions['SlotedDIAMOND'], all_slotted_paths, paths_4_ns3 = self.slotted_diamond(copy.deepcopy(Gloval_env),
                                                                                                env_configurations, flows_statistics,
                                                                                                grrl_data=False)  # True
                if self.run_competition:
                    # Run DIAMOND
                    print(f"Started DIAMOND \n")
                    _, _, _, _, self.first_step_actions['DIAMOND'] = self.diamond(copy.deepcopy(Gloval_env), grrl_data=True)
                    print(f"Finished DIAMOND \n")

                    # Run GRRL
                    _, _, _, self.first_step_actions['GRRL'], _ = self.grrl(copy.deepcopy(Gloval_env), grrl_data=True)
                    # Run competitors
                    print(f"Started all competition \n")
                    for name, comp in zip(self.competitors.keys(), self.competitors.values()):
                        _, _, _, _, self.first_step_actions[name] = comp.run(copy.deepcopy(Gloval_env), seed)
                    print(f"Finished all competition \n")

                else:  # Todo: manually give everyone same decisions to avoid bugs

                    pass

                    # Run GRRL
                    # print(f"Started GRRL \n")
                    # _, _, _, self.first_step_actions['DIAMOND'], _ = self.grrl(copy.deepcopy(Gloval_env), grrl_data=True)
                    #
                    # print(f"Finished GRRL \n")
                    #
                    # self.first_step_actions['GRRL'] = copy.deepcopy(self.first_step_actions['DIAMOND'])
                    #
                    # for name, comp in zip(self.competitors.keys(), self.competitors.values()):
                    #     self.first_step_actions[name] = copy.deepcopy(self.first_step_actions['DIAMOND'])

                # initialize Global flows list for each algo (entire run flows)
                if self.run_competition:
                    Algos_Global_flows = {
                        'SlotedDIAMOND': copy.deepcopy(env_configurations['flows']),
                        'DIAMOND': copy.deepcopy(env_configurations['flows']),
                        'GRRL': copy.deepcopy(env_configurations['flows']),
                        'DQN+GNN': copy.deepcopy(env_configurations['flows']),
                        'OSPF': copy.deepcopy(env_configurations['flows']),
                        'RandomBL': copy.deepcopy(env_configurations['flows']),
                        'DIAR': copy.deepcopy(env_configurations['flows']),
                        'IACR': copy.deepcopy(env_configurations['flows']),
                    }
                else:
                    Algos_Global_flows = {
                        'SlotedDIAMOND': copy.deepcopy(env_configurations['flows'])
                    }


                full_run_data = []

                print(f'\nstarting real run\n')
                for slot in range(self.num_slots):

                    # initalze data for slot
                    slot_data = self.create_initial_slot_data()

                    # create flows for slot for each algo
                    algos_slot_flows = self.create_slot_flows(Algos_Global_flows)

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
                                                      **kwargs)

                                       for algo in self.algos}

                    print(f'slot {slot}, {len(Algos_step_envs["SlotedDIAMOND"].flows)}/{len(Gloval_env.flows)} flows alive')

                    # -----Run ALGOS------
                    # Run SlottedDIAMOND
                    if Algos_step_envs['SlotedDIAMOND'].flows:
                        SlotedDIAMOND_rates_data, SlotedDIAMOND_delay_data = run_Slotted_predefined_actions(Algos_step_envs['SlotedDIAMOND'], self.first_step_actions['SlotedDIAMOND'], slot)
                        slot_data['SlotedDIAMOND_active_flows'] = [flow['name'] for flow in Algos_step_envs['SlotedDIAMOND'].flows]
                        slot_data['SlotedDIAMOND_delay'] = SlotedDIAMOND_delay_data['delay_per_flow']
                        slot_data['SlotedDIAMOND_rates'] = SlotedDIAMOND_rates_data['rate_per_flow']

                    if self.run_competition:
                        # --------- Run Competition ---------- #

                        # Run DIAMOND
                        if Algos_step_envs['DIAMOND'].flows:
                            diamond_rates_data, diamond_delay_data = run_predefined_actions(Algos_step_envs['DIAMOND'], self.first_step_actions['DIAMOND'])
                            slot_data['DIAMOND_active_flows'] = [flow['name'] for flow in Algos_step_envs['DIAMOND'].flows]
                            slot_data['DIAMOND_delay'] = diamond_delay_data['delay_per_flow']
                            slot_data['DIAMOND_rates'] = diamond_rates_data['rate_per_flow']

                        # Run GRRL
                        if Algos_step_envs['GRRL'].flows:
                            grrl_rates_data, grrl_delay_data = run_predefined_actions(Algos_step_envs['GRRL'],
                                                                                      self.first_step_actions['GRRL'])
                            slot_data['GRRL_active_flows'] = [flow['name'] for flow in Algos_step_envs['GRRL'].flows]
                            slot_data['GRRL_delay'] = grrl_delay_data['delay_per_flow']
                            slot_data['GRRL_rates'] = grrl_rates_data['rate_per_flow']

                        # Run competitors
                        for name, comp in zip(self.competitors.keys(), self.competitors.values()):
                            if Algos_step_envs[name].flows:
                                rates_data, delay_data = run_predefined_actions(Algos_step_envs[name],
                                                                                self.first_step_actions[name])
                                slot_data[f"{name}_active_flows"] = [flow['name'] for flow in Algos_step_envs[name].flows]
                                slot_data[f"{name}_delay"] = delay_data['delay_per_flow']
                                slot_data[f"{name}_rates"] = rates_data['rate_per_flow']

                            # Update Algos_Global_flows according to preformance of each algo
                    Algos_Global_flows = self.update_Global_flows(Algos_Global_flows, flows_statistics, slot_data, slot)

                    # gather data from all slots to be avarge over all episode
                    full_run_data.append(slot_data)

                    print(f"Finished slot {slot + 1}/{self.num_slots} In Real Slotted_DIAMOND \n")

                # Modify Global_data to be avarged over all episodes
                Episode_Avarge_data = self.prepare_end_of_run_data(full_run_data, Episode_Avarge_data)

                this_Episode_Avarge_data = self.prepare_one_end_of_run_data(full_run_data)

                # plot individual episode results.
                save = True
                subfolder_path = plot_algorithm_metrics(this_Episode_Avarge_data,
                                                        Gloval_env=Gloval_env,
                                                        graph_mode=kwargs['graph_mode'], save_fig=save,
                                                        refinement_steps=self.refinement_steps)
                # save generate_env args
                file_path = os.path.join(subfolder_path, "generate_env_args.json")
                save_arguments_to_file(filename=file_path, args=env_args)
                # save Hawks model params
                file_path = os.path.join(subfolder_path, "Hawkes_params.json")
                save_arguments_to_file(filename=file_path, args=self.HawkesParams)
                # save all_paths
                file_path = os.path.join(subfolder_path, "all_slotted_paths.json")
                save_arguments_to_file(filename=file_path, args=paths_4_ns3)
                # save link capacities for NS3
                file_path = os.path.join(subfolder_path, "link_capacities.json")
                capacity_list = [int(x / 100 ) for x in list(Gloval_env.bandwidth_edge_list)]
                save_arguments_to_file(filename=file_path, args=capacity_list)
                # save flows for NS3
                file_path = os.path.join(subfolder_path, "flows.json")
                flows_list = Gloval_env.flows
                new_flow_list = []
                for flow in flows_list:
                    new_flow = flow
                    new_flow['packets'] /= 100
                    new_flow_list.append(new_flow)
                save_arguments_to_file(filename=file_path, args=new_flow_list)    
                save_arguments_to_file(filename=file_path, args=flows_list)
                # save node positions for NS3
                file_path = os.path.join(subfolder_path, "node_positions.npy")
                node_positions = Gloval_env.node_positions
                np.save(file_path, node_positions)
                # save links indices for NS3
                file_path = os.path.join(subfolder_path, "links_indices.json")
                link_indices = Gloval_env.eids
                # Convert edge keys to "u-v" string format
                serializable_eids = {f"{u}-{v}": eid for (u, v), eid in Gloval_env.eids.items()}
                save_arguments_to_file(filename=file_path, args=serializable_eids)
                # save graph image
                Gloval_env.show_graph(save_path=os.path.join(subfolder_path, "graph.png"), show_fig=False)

            if data_paths_list:
                # Load the data from the pickle file
                subfolder_path = data_paths_list[episode]
                with open(os.path.join(subfolder_path, "data.pkl"), "rb") as pickle_file:
                    this_Episode_Avarge_data = pickle.load(pickle_file)

            """
            My adding
            take average through time at the end of an episode
            """
            for key in this_Episode_Avarge_data.keys():  # take average over this in each individual episode and not the commulative
                if "active_flows" in key:
                    continue

                if not self.run_competition and 'SlotedDIAMOND' not in key:  # TODO: handles the case of when some runs in "this_Episode_Avarge_data" had also competition but we are not interested
                    continue
                else:
                    average_data_through_time[key] += np.mean(this_Episode_Avarge_data[key][this_Episode_Avarge_data[key] != 0])
            # ---------------------------------------------------------------------- #

        # average
        Episode_Avarge_data = {key: value / self.num_episodes for key, value in Episode_Avarge_data.items()}

        average_data_through_time = {key: value / self.num_episodes for key, value in average_data_through_time.items()}
        # separate rates, delays
        average_rates_through_time, average_delays_through_time = self.separate_rates_delays(average_data=average_data_through_time)

        print(f"==========================================================================================")

        labels = self.algos

        # manual plot for debug
        # plot_algorithm_metrics(Episode_Avarge_data, num_flows=num_flows, seed=seed)

        return Episode_Avarge_data, labels, average_rates_through_time, average_delays_through_time, subfolder_path

    def create_slot_flows(self, Algos_Global_flows):
        '''
        input: takes as input global flows for each algo
        output: flows list for each algo for a signle slot

        This function is used to create flows for each algo for a single slot.
        if the algo has one less flow to allocate, the slot_flows needs to be updated accordingly
        it counts how many flows left for each algo and creates flows for each algo for a single slot with a fixes packet size (demand)
        '''
        new_Algos_slot_flows = {}
        pkt_size = self.pkt_size
        for algo, flows in Algos_Global_flows.items():
            new_Algos_slot_flows[algo] = self.generate_flows_with_fixed_pkt(flows,
                                                                            pkt_size)  # Replace self.modify_flow(flow) with your desired operation
        return new_Algos_slot_flows

    def update_Global_flows(self, Algos_Global_flows, flows_statistics, slot_data, slot):
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

        # adding flow pkts according to arrival statistics only if predictor is "on" or "Ideal" else just subtract
        if self.predictor_mode == 'predictor_off':
            pass

        else:
            if slot % self.pkt_arrival_sample_rate == 0:  # and slot != 0:
                for flow_statistic in flows_statistics:
                    entered_new_pkts = flow_statistic.step(slot=slot)
                    flow_name = flow_statistic.flow_name
                    for algo in self.algos:
                        flow = get_flow_by_name(Algos_Global_flows[algo], flow_name)
                        flow['packets'] += entered_new_pkts

        # removing flow pkts according to arrvied pkts
        for algo in self.algos:
            algo_active_flows = slot_data[f"{algo}_active_flows"]
            algo_delay = slot_data[f"{algo}_delay"]
            algo_rate = slot_data[f"{algo}_rates"]

            # how many packets will be delivered in slot_duration
            units = self.units  # 1e6
            initial_delay = algo_delay
            #  -[Megabit]-       -[Mbps]-       ------[microsec]-----    --[micro sec]--     -[micro sec]-
            delivered_packets = algo_rate * (self.slot_duration * units - initial_delay) / units  # rate [Mbps] * (slot_duration [micro sec])/microsec

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

    def generate_flows_with_fixed_pkt(self, flows, pkt_size):
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

    def create_initial_slot_data(self):

        if self.run_competition:
            data = {
                'SlotedDIAMOND_active_flows': None,
                'SlotedDIAMOND_delay': 0,
                'SlotedDIAMOND_rates': 0,
                'DIAMOND_active_flows': None,
                'DIAMOND_delay': 0,
                'DIAMOND_rates': 0,
                'GRRL_active_flows': None,
                'GRRL_delay': 0,
                'GRRL_rates': 0,
            }
            for c in self.competitors.keys():
                data[f"{c}_active_flows"] = None
                data[f"{c}_delay"] = 0
                data[f"{c}_rates"] = 0

        else:
            data = {
                'SlotedDIAMOND_active_flows': None,
                'SlotedDIAMOND_delay': 0,
                'SlotedDIAMOND_rates': 0,
            }

        return data

    def create_initial_run_data(self):
        if self.run_competition:
            data = {
                'SlotedDIAMOND_delay': np.zeros(self.num_slots),
                'SlotedDIAMOND_rates': np.zeros(self.num_slots),
                'SlotedDIAMOND_active_flows': np.zeros(self.num_slots),

                'DIAMOND_delay': np.zeros(self.num_slots),
                'DIAMOND_rates': np.zeros(self.num_slots),
                'DIAMOND_active_flows': np.zeros(self.num_slots),

                'GRRL_delay': np.zeros(self.num_slots),
                'GRRL_rates': np.zeros(self.num_slots),
                'GRRL_active_flows': np.zeros(self.num_slots),
            }
            for c in self.competitors.keys():
                data[f"{c}_delay"] = np.zeros(self.num_slots)
                data[f"{c}_rates"] = np.zeros(self.num_slots)
                data[f"{c}_active_flows"] = np.zeros(self.num_slots)

        else:
            data = {
                'SlotedDIAMOND_delay': np.zeros(self.num_slots),
                'SlotedDIAMOND_rates': np.zeros(self.num_slots),
                'SlotedDIAMOND_active_flows': np.zeros(self.num_slots),
            }

        return data

    def prepare_end_of_run_data(self, full_run_data, Episode_Avarge_data):
        for slot in range(self.num_slots):
            for c in self.algos:
                Episode_Avarge_data[f"{c}_delay"][slot] += np.average(full_run_data[slot][f"{c}_delay"])
                Episode_Avarge_data[f"{c}_rates"][slot] += np.average(full_run_data[slot][f"{c}_rates"])
                Episode_Avarge_data[f"{c}_active_flows"][slot] += len(full_run_data[slot][f"{c}_active_flows"]) if \
                full_run_data[slot][f"{c}_active_flows"] else 0
        return Episode_Avarge_data

    def prepare_one_end_of_run_data(self, full_run_data):
        Episode_Avarge_data = self.create_initial_run_data()
        for slot in range(self.num_slots):
            for c in self.algos:
                Episode_Avarge_data[f"{c}_delay"][slot] = np.average(full_run_data[slot][f"{c}_delay"])
                Episode_Avarge_data[f"{c}_rates"][slot] = np.average(full_run_data[slot][f"{c}_rates"])
                Episode_Avarge_data[f"{c}_active_flows"][slot] = len(full_run_data[slot][f"{c}_active_flows"]) if \
                full_run_data[slot][f"{c}_active_flows"] else 0
        return Episode_Avarge_data

    def create_initial_mean_through_time_data(self, Episode_Avarge_data):

        data = {}
        for c in Episode_Avarge_data.keys():
            if "active_flows" in c:
                continue
            else:
                data[f"{c}"] = 0

        return data

    def separate_rates_delays(self, average_data):
        rates_data = {}
        delays_data = {}
        for key, value in average_data.items():
            if "rates" in key:
                rates_data[key] = value
            elif "delay" in key:
                delays_data[key] = value
        return rates_data, delays_data


if __name__ == "__main__":

    BASE_PATH = os.path.join("..", "results", "inference_vs_competitors")
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.abspath(__file__)

    # general params
    num_nodes = 25  # 60
    num_edges = 35  # 90
    num_actions = 8  # 15, 4
    temperature = 1.2
    num_episodes = 10  # 3
    episode_from = 7500  # 7500  # 7501
    nb3r_steps = 0 #20  # , 100
    refinement_steps = 10  # number of NB3R steps in slotted_diamond

    trx_power_mode = 'equal'
    rayleigh_scale = 1
    max_trx_power = 10
    channel_gain = 1
    min_capacity = 200  # 200
    max_capacity = 500  # 500
    min_flow_demand = 5   # 5, 500
    max_flow_demand = 200  # 200, 2000

    pkt_size = 5000  # 500, 100
    units = 1e6

    slot_duration = 1
    num_slots = 50  # 150

    pkt_arrival_sample_rate = 10  # 5

    # Hawkes parms
    HawkesParams = dict(
        lambda0=0.005,  # 0.9
        alpha=0.025,  # 0.5 0.4
        beta=0.0001,  # 0.7 0.8
        history_num_slots=200,  # 100
        allow_Hawkes_arrivals=True,     # this initiate a random first demand to each flow (not the initial history count)
        elephent_flows_num=3,
        mice_scaler=0.5,  # 0.1, 50, 150, 20
        elephent_scaler=30,
        ManualAdded_Fixed_InitalPkts=100)  # 0.1, 100, 350, 50

    # predictor params
    predictor_mode = 'predictor_off' #'predictor_on'  # 'predictor_on' # 'predictor_off', 'Ideal'  # predictor_off - dont allow the predicted arrivals to be added to the algorithm each slot
    
    # TO TURN use only emptying flows, AND TO NOT LET NEW PACKET arrive DURING NUP:
    # set allow_Hawkes_arrivals=False AND predictor_mode = 'predictor_off'
    
    run_competition = True  # True if regular case, False if want to run only our algo

    for GRAPH_MODE in ['nepal','random']:  # ['random', 'nsfnet', 'geant', 'random_internet', 'nepal']
        for trx_power_mode in ['equal']:

            print("----------------------------")
            print(trx_power_mode, GRAPH_MODE)
            print("----------------------------")

            data_rates = []
            data_delay = []

            data_paths_list_for_all_flows = [

                                            [

                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250807_111507_40_Flows/',
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250807_122403_40_Flows/',
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250807_132057_40_Flows/',
                                            ],

                                            [

                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250807_144828_50_Flows/',
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250807_162736_50_Flows/',
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250807_175109_50_Flows/',
                                            ],

                                            [

                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250808_133843_60_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250809_163402_60_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250810_101956_60_Flows/',
                                            ],

                                            [

                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250812_203518_70_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250810_181414_70_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250810_124850_70_Flows/',
                                            ],

                                            [

                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250812_234418_80_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250818_001601_80_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250811_013809_80_Flows/',
                                             ],

                                            [
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250813_035747_90_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250818_045529_90_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250821_155015_90_Flows/',
                                            ],

                                            [
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250813_155410_100_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250818_154615_100_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250821_210453_100_Flows/',
                                            ],


                                            [
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250814_154424_110_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250820_040153_110_Flows/',
                                             r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250823_010309_110_Flows/',

                                            ],

                                            [
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250816_230118_120_Flows/',
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250821_023251_120_Flows/',
                                                r'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/120_Nodes_200_Edges/20250904_162720_120_Flows/',

                                            ],



                                             ]

            flows = [20,40,60]#,60,70,80,90,100,110,120]

            for num_flows_idx, num_flows in enumerate(flows):

                print(f' Starting {num_flows} flows Run\n')

                alg = TestvsCompetitors(grrl_model_path=MODEL_PATH, num_episodes=num_episodes, run_competition=run_competition,
                                        episode_from=episode_from,
                                        temperature=temperature, nb3r_steps=nb3r_steps, refinement_steps=refinement_steps, num_slots=num_slots,
                                        slot_duration=slot_duration, predictor_mode=predictor_mode,
                                        pkt_arrival_sample_rate=pkt_arrival_sample_rate, pkt_size=pkt_size, units=units,
                                        HawkesParams=HawkesParams)

                loading_graphs = False
                data, labels, average_rates_through_time, average_delays_through_time, subfolder_path = alg(data_paths_list=data_paths_list_for_all_flows[num_flows_idx] if loading_graphs else None,  # data_paths_list_for_all_flows[num_flows_idx], None
                                                                                                            num_nodes=num_nodes, num_edges=num_edges, num_flows=num_flows,
                                                                                                            num_actions=num_actions,
                                                                                                            graph_mode=GRAPH_MODE,
                                                                                                            trx_power_mode=trx_power_mode, rayleigh_scale=rayleigh_scale,
                                                                                                            max_trx_power=max_trx_power, channel_gain=channel_gain,
                                                                                                            min_capacity=min_capacity, max_capacity=max_capacity,
                                                                                                            min_flow_demand=min_flow_demand, max_flow_demand=max_flow_demand)

                data_rates.append(list(average_rates_through_time.values()))
                data_delay.append(list(average_delays_through_time.values()))

            save_fig = True
            plot_algorithm_mean_performance(num_episodes=num_episodes, flows=flows, algo_names=labels, algo_rates=data_rates,algo_delays=data_delay,
                                            subfolder_path=subfolder_path, save_fig=save_fig)


    print('done')
