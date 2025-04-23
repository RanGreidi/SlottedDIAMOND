import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from scipy.interpolate import interp1d
sys.path.insert(0, 'DIAMOND')

class HawkesModel:
    def __init__(self,
                lambda0,
                alpha,
                beta,
                source,
                destination,
                flow_name,
                num_slots,
                slot_duration,
                history_num_slots,
                pkt_arrival_sample_rate,
                type,
                mice_scaler=1,
                elephent_scaler=1,
                ManualAdded_Fixed_InitalPkts=1,
                seed = 123):
        
        """
        Initializes the Markov chain.
        
        :param transition_matrix: 2D NumPy array representing the transition probabilities between states.
        :param states: List of possible states.
        :param initial_state: (Optional) Initial state. If None, a random state is chosen.
        """

        # general params
        self.source = source
        self.destination = destination
        self.flow_name = flow_name
        self.seed = seed + 10000*self.flow_name
        self.rng = np.random.default_rng(self.seed)  # Use NumPy's random generator for better reproducibility and to isolate from the rest of my project

        self.mice_scaler=mice_scaler
        self.elephent_scaler=elephent_scaler

        # run time params
        self.slot_duration = slot_duration
        self.num_slots = num_slots
        self.history_num_slots = history_num_slots
        self.total_num_slots = num_slots + self.history_num_slots
        self.pkt_arrival_sample_rate = pkt_arrival_sample_rate

        # Hawkes Params
        self.lambda0 = lambda0
        self.alpha = alpha
        self.beta = beta
        self.T = self.total_num_slots
        self.events, self.orig_counts = self.__generate_hawkes_events(self.lambda0, self.alpha, self.beta, self.T)
        self.counts = self.__generate_hawkes_counts()
        

        self.future_count = self.counts[self.history_num_slots:]
        self.history_count = self.counts[:self.history_num_slots]
        
        self.history_events, self.future_events, self.total_events = self.simulate_events()

        # type params 
        self.type = type
        if self.type == 'mice': 
            self.type_scaler = self.mice_scaler 
        elif self.type == 'elephent':
            self.type_scaler = self.elephent_scaler
        else:
            self.type_scaler = 1
        
        # initial count
        self.ManualAdded_Fixed_InitalPkts = ManualAdded_Fixed_InitalPkts
        self.initial_count = (self.counts[self.history_num_slots] + self.ManualAdded_Fixed_InitalPkts) * self.type_scaler

        

    def step(self, slot):
        """
        Takes one step in the Markov Chain by transitioning to the next state based on the transition matrix.
        """
        # current_event = self.future_events[slot] * self.type_scaler
        current_event = sum(self.future_events[slot-self.pkt_arrival_sample_rate:slot]) * self.type_scaler
        return current_event
    
    def simulate_events(self):
        total_run = np.append(self.history_count,self.future_count)
        history_events = self.history_count
        future_events = self.future_count
        

        # history
        history_events = []
        prev_count = 0
        for count in self.history_count:
            delta = count - prev_count
            history_events.append(delta)
            prev_count = count
        # future
        future_events = []
        for count in self.future_count:
            delta = count - prev_count
            future_events.append(delta)
            prev_count = count            
        
        # total run
        total_events = []
        prev_count = 0
        for count in total_run:
            delta = count - prev_count
            total_events.append(delta)
            prev_count = count
        return history_events, future_events, total_events

    def simulate_counts(self):
        total_run_count = np.append(self.history_count,self.future_count)
        return self.history_count, self.future_count, total_run_count
 
    def plot_interpoalted_vs_original(self):
        # interpolated
        history_count, future_count, total_run_count = self.simulate_counts()
        total_run_count *= self.type_scaler
        plt.figure(figsize=(10, 6))
        plt.step([step for step in range(len(total_run_count))],total_run_count, marker='*', label="Count Process N(t) Interpolated")
        # original
        plt.step(self.events, self.orig_counts, where='post', color='r', label="Count Process N(t) ORIGINAL")
        plt.axvline(x=self.history_num_slots, color='black', linestyle='dashed')
        plt.title("Hawkes Process - Count Process Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig('Hawkes_compare_interpoalted_vs_original original')

    def __generate_hawkes_events(self, lambda0, alpha, beta, T):
        """
        Simulates a 1D Hawkes process using Ogata's thinning algorithm.
        
        Parameters:
        lambda0 : float  - Baseline intensity
        alpha   : float  - Excitation factor
        beta    : float  - Decay rate
        T       : float  - End time for simulation
        
        Returns:
        events  : list   - List of event timestamps
        counts  : list   - Count process N(t) at each event time
        """
        events = []
        counts = []
        t = 0  # Start time

        while t < T:
            # Compute upper bound on intensity (λ*)
            lambda_star = lambda0 + sum(alpha * np.exp(-beta * (t - ti)) for ti in events)

            # Generate next event time
            tau = np.random.exponential(1 / lambda_star)
            t += tau
            if t > T:
                break  # Stop if we exceed simulation time
            
            # Compute true intensity at new time t
            lambda_t = lambda0 + sum(alpha * np.exp(-beta * (t - ti)) for ti in events)

            # Accept or reject with probability λ(t) / λ*
            if np.random.rand() < lambda_t / lambda_star:
                events.append(t)  # Accept event
                counts.append(len(events))  # Store total event count

        return events, counts

    def __generate_hawkes_counts(self):
        interpolated_counts = np.zeros(self.total_num_slots)
        for event_time in self.events:
            interpolated_counts[math.ceil(event_time):] += 1
        return interpolated_counts

if __name__ == "__main__":


    model = HawkesModel(               
                    lambda0 = 0.9,
                    alpha = 0.5,
                    beta = 0.7,

                    source=0,
                    destination=1,
                    flow_name=0,
                    num_slots=60,
                    slot_duration=1,
                    history_num_slots=100,
                    pkt_arrival_sample_rate=1,
                    type='elephent',
                    mice_scaler = 0.1,
                    elephent_scaler = 0.1,

                    seed = 123
                    )

    model.plot_interpoalted_vs_original()

    # # Simulate the Markov Chain for 50 steps
    # _,_, counts = mc.simulate_counts()

    # # Optionally, plot the state transitions
    # plt.figure(figsize=(10, 6))
    # plt.step([step for step in range(len(counts))],counts, label="Count Process N(t)")
    # plt.title("State Transitions in Markov Chain")
    # plt.xlabel("Step")
    # plt.ylabel("State")
    # plt.grid(True)
    # plt.savefig('Probability_Model')