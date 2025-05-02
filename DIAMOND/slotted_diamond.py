import numpy as np
import os
import copy
from stage1_grrl import GRRL
from stage2_nb3r import nb3r
from environment import GraphEnvPower as GraphEnv
from environment.utils import *
from environment.FlowPrediction import FlowPrediction

class SLOTTED_DIAMOND:
    def __init__(self,
                 grrl_model_path,
                 nb3r_steps=100,
                 nb3r_tmpr=10,
                 slot_duration=1,
                 num_slots=100,
                 pkt_size=100,
                 predictor_mode='Ideal',
                 pkt_arrival_sample_rate=1):
        
        if grrl_model_path is None:
            grrl_model_path = os.path.join(".", "pretrained", "model_20221113_212726_480.pt")
        
        self.grrl = GRRL(path=grrl_model_path)
        self.nb3r_steps = nb3r_steps
        self.nb3r_tmpr = nb3r_tmpr
        self.slot_duration = slot_duration
        self.num_slots = num_slots
        self.pkt_size = pkt_size
        self.predictor_mode = predictor_mode
        self.pkt_arrival_sample_rate = pkt_arrival_sample_rate
        
    def __call__(self, Gloval_env, env_configurations, flows_statistics, grrl_data=False):


        # initialize Global flows list for each algo (entire run flows)
        Global_flows = Gloval_env.flows
        full_run_data = []
        Actions = []
        all_slotted_paths = [[] for _ in range(self.num_slots)]
        self.flows_statistics = flows_statistics
        if self.predictor_mode == 'predictor_on':
            self.flows_predicted_statistics = self.predict_demand(flows_statistics)

        for slot in range(self.num_slots):  

            # initalze data for slot
            slot_data = self.create_initial_slot_data()

            # create flows for slot for each algo
            slot_flows = self.create_slot_flows(Global_flows)

            print(f'slot {slot}, {len(slot_flows)}/{len(Global_flows)} flows alive')

            # Create env for slot
            step_env = GraphEnv(adjacency_matrix=env_configurations['adjacency_matrix'],
                                            bandwidth_matrix=env_configurations['bandwidth_matrix'],
                                            interference_matrix=env_configurations['interference_matrix'],
                                            node_positions=env_configurations['node_positions'],
                                            flows=slot_flows,
                                            k=env_configurations['k'],
                                            direction=env_configurations['direction'],
                                            reward_balance=env_configurations['reward_balance'],
                                            seed=env_configurations['seed']) 

            # Run SLotted DIAMOND
            if step_env.flows:
                slot_paths, _, _, slot_action = self.run_slot(step_env, grrl_data=grrl_data)
                all_slotted_paths[slot].append(slot_paths)
                SlotedDIAMOND_delay_data = step_env.get_delay_data()
                SlotedDIAMOND_rates_data = step_env.get_rates_data()
                slot_data['SlotedDIAMOND_active_flows'] = [flow['name'] for flow in step_env.flows]
                slot_data['SlotedDIAMOND_delay'] = SlotedDIAMOND_delay_data['delay_per_flow']
                slot_data['SlotedDIAMOND_rates'] = SlotedDIAMOND_rates_data['rate_per_flow']
                
                # Actions is in the form of: {flow_name_1:flow_action_1 , flow_name_2:flow_action_2 ... }
                # Actions.append( [{flow:action} for flow,action in zip(slot_data['SlotedDIAMOND_active_flows'],slot_action)] )
                Actions.append( {flow:action for flow,action in zip(slot_data['SlotedDIAMOND_active_flows'],slot_action)} )
            else:
                Actions.append([])

            # Update Algos_Global_flows according to preformance of each algo
            Global_flows = self.update_Global_flows(Global_flows, slot_data, slot)

            # gather data from all slots to be avarge over all episode
            full_run_data.append(slot_data)

            print(f"Finished slot {slot + 1}/{self.num_slots} In initial Slotted_DIAMOND \n")

        return full_run_data, Actions

    def run_slot(self, env, grrl_data=False):
        # stage 1
        rl_actions, rl_paths, rl_reward = self.grrl.run(env=env)
        rl_actions.sort(key=lambda x: x[0])
        rl_actions = [x[1] for x in rl_actions]
        rl_delay_data = env.get_delay_data()
        rl_rates_data = env.get_rates_data()

        # stage 2
        # self.nb3r_steps = int(env.num_flows * 5)
        nb3r_action = nb3r(
                           objective=lambda a: -self.rates_objective(env, a),
                           # objective=lambda a: -self.reward_objective(env, a),
                           # objective=lambda a: -self.delay_objective(env, a),
                           state_space=env.get_state_space(),
                           num_iterations=self.nb3r_steps,  # max(self.nb3r_steps, int(env.num_flows * 5)),
                           initial_state=rl_actions.copy(),
                           verbose=False,
                           seed=env.seed,
                           return_history=False,
                           initial_temperature=self.nb3r_tmpr)
        # routs
        routs = env.get_routs(nb3r_action)

        action = rl_actions #rl_actions

        if grrl_data:
            return routs, rl_rates_data, rl_delay_data, action
        return routs

    def create_slot_flows(self, Global_flows):
        '''
        input: takes as input global flows for each algo
        output: flows list for each algo for a signle slot

        This function is used to create flows for each algo for a single slot. 
        if the algo has one less flow to allocate, the slot_flows needs to be updated accordingly
        it counts how many flows left for each algo and creates flows for each algo for a single slot with a fixes packet size (demand)
        '''
        pkt_size = self.pkt_size
        new_slot_flows = self.generate_flows_with_fixed_pkt(Global_flows ,pkt_size)  # Replace self.modify_flow(flow) with your desired operation
        return new_slot_flows

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

    def update_Global_flows(self, Global_flows, data, slot):

        '''
        input:  1. flows list for each algo according to its current state
                2. rate and delay data for each algo
        output: updated flow list for each algo

        This function takes in the current state of the flows for each algo and the rate and delay data for each algo, and updates the flows packets for each algo
        according to the performance of each algo in the previous slot

        in the future,  : flows that need to be added in a slot will be added here !!------according to the prediction----!!.
        
        Units: 
        slot_duration [sec]
        rate [Mbps]
        initial_delay [micro sec]
        BW [MHz]
        delivered_packets [Megabit]
        '''
        
        algo_active_flows = data["SlotedDIAMOND_active_flows"]
        algo_delay = data["SlotedDIAMOND_delay"]
        algo_rate = data["SlotedDIAMOND_rates"]
        
        # how many packets will be delivered in slot_duration
        units = 1e6
        initial_delay = algo_delay
        #  -[Megabit]-       -[Mbps]-       ------[microsec]-----    --[micro sec]--     -[micro sec]-
        delivered_packets = algo_rate * (self.slot_duration*units - initial_delay) / units          # rate [Mbps] * (slot_duration [micro sec])/microsec

        # removing flow pkts according to arrvied pkts
        if algo_active_flows:
            for idx_in_metrics_for_flow, flow_name in enumerate(algo_active_flows):
                flow = get_flow_by_name(Global_flows,flow_name)  # flow is a pointer to the current flow in the Algos_Global_flows
                if flow['packets'] <= delivered_packets[idx_in_metrics_for_flow]:
                    flow['packets'] = 0
                else:
                    flow['packets'] = (flow['packets'] - delivered_packets[idx_in_metrics_for_flow])  # my change, packets are ints
        
        # adding flow pkts according to arrivle PREDICTION
        if slot % self.pkt_arrival_sample_rate == 0:  # and slot != 0:
            if self.predictor_mode == 'Ideal':
                for flow_statistic in self.flows_statistics:
                    # entered_new_pkts = flow_statistic.future_events[slot] * flow_statistic.type_scaler  # my change to match multiplication in .step() .flow_statistic.future_events[slot]
                    entered_new_pkts = flow_statistic.step(slot)
                    flow_name = flow_statistic.flow_name
                    flow = get_flow_by_name(Global_flows, flow_name)
                    flow['packets'] += entered_new_pkts

            if self.predictor_mode == 'predictor_on':

                for flow_predicted_statistic in self.flows_predicted_statistics:
                    entered_new_pkts = flow_predicted_statistic.step(slot)
                    flow_name = flow_predicted_statistic.flow_name
                    flow = get_flow_by_name(Global_flows, flow_name)
                    flow['packets'] += entered_new_pkts

            if self.predictor_mode == 'predictor_off':
                pass

        return Global_flows
    
    def create_initial_slot_data(self):
        return {
                'SlotedDIAMOND_active_flows': None,
                'SlotedDIAMOND_delay': 0,
                'SlotedDIAMOND_rates': 0,
                }

    def predict_demand(self, flows_statistics):

        flows_predicted_statistics = []
        for flow_statistics in flows_statistics:

            flow_predicted_statistics = FlowPrediction(flow_statistics)

            flows_predicted_statistics.append(flow_predicted_statistics)

        return flows_predicted_statistics

    @staticmethod
    def rates_objective(env, actions):
        env.reset()
        env.eval_all(actions)
        return np.sum(env.get_rates_data()['sum_flow_rates'])

    @staticmethod
    def reward_objective(env, actions):
        env.reset()
        return env.eval_all(actions)

    @staticmethod
    def delay_objective(env, actions):
        env.reset()
        env.eval_all(actions)
        return np.sum(env.get_delay_data()['mean_delay'])
