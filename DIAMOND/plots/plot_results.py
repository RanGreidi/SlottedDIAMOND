import os.path

# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')


def plot_rate_delay(topology_path, rate_path, delay_path, **kwargs):
    delay_data = pd.read_csv(delay_path, comment='#')
    rate_data = pd.read_csv(rate_path, comment='#')

    f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

    ax0.imshow(plt.imread(topology_path))
    ax0.axis('off')

    ax1.plot(rate_data["N"], rate_data["DIAMOND"], color='blue', marker='o', label="DIAMOND")
    ax1.plot(rate_data["N"], rate_data["IACR"], color='red', marker='p', label="IACR")
    ax1.plot(rate_data["N"], rate_data["DIAR"], color='darkviolet', marker='+', label="DIAR")
    ax1.plot(rate_data["N"], rate_data["DQN+GNN"], color='orange', marker='x', label="DQN+GNN")
    ax1.plot(rate_data["N"], rate_data["OSPF"], color='green', marker='^', label="OSPF")
    ax1.plot(rate_data["N"], rate_data["RandomBL"], color='violet', marker='*', label="RandomBL")
    ax1.set_xlabel("Number of Flows $N$")
    ax1.set_ylabel("Avg. Flow Rate [Mbps]")
    ax1.legend()
    if kwargs.get('rate_log', False):
        ax1.yscale('log')

    ax2.plot(delay_data["N"], delay_data["RandomBL"], color='violet', marker='*', label="RandomBL")
    ax2.plot(delay_data["N"], delay_data["OSPF"], color='green', marker='^', label="OSPF")
    ax2.plot(delay_data["N"], delay_data["DQN+GNN"], color='orange', marker='x', label="DQN+GNN")
    ax2.plot(delay_data["N"], delay_data["DIAR"], color='darkviolet', marker='+', label="DIAR")
    ax2.plot(delay_data["N"], delay_data["IACR"], color='red', marker='p', label="IACR")
    ax2.plot(delay_data["N"], delay_data["DIAMOND"], color='blue', marker='o', label="DIAMOND")
    ax2.set_xlabel("Number of Flows $N$")
    ax2.set_ylabel("Delay [timesteps]")
    ax2.legend()
    if kwargs.get('delay_log', False):
        ax2.yscale('log')

    if kwargs.get('title'):
        f.suptitle(kwargs.get('title'))
    plt.show()


def plot_all():
    # V60E90
    plot_rate_delay(topology_path='v60e90/rand_graph_60_90.png',
                    rate_path='v60e90/random_rates_V60_E90_all_comps.csv',
                    delay_path='v60e90/random_delay_V60_E90_all_comps.csv',
                    title='V60E90')
    # NSFNET
    plot_rate_delay(topology_path='nsfnet/nsfnet.png',
                    rate_path='nsfnet/nsfnet_rates_all_comps.csv',
                    delay_path='nsfnet/nsfnet_delay_all_comps.csv',
                    title='NSFNET')
    # GEANT2
    plot_rate_delay(topology_path='geant2/geant2.png',
                    rate_path='geant2/geant2_rates_all_comps.csv',
                    delay_path='geant2/geant2_delay_all_comps.csv',
                    title='GEANT2')
    # V30E50
    plot_rate_delay(topology_path='v30e50/rand_graph_30_50.png',
                    rate_path='v30e50/random_equal_rates_V30E50.csv',
                    delay_path='v30e50/random_equal_delay_V30E50.csv',
                    title='V30E50, equal')
    plot_rate_delay(topology_path='v30e50/rand_graph_30_50.png',
                    rate_path='v30e50/random_steps_rates_V30E50.csv',
                    delay_path='v30e50/random_steps_delay_V30E50.csv',
                    title='V30E50, steps')
    plot_rate_delay(topology_path='v30e50/rand_graph_30_50.png',
                    rate_path='v30e50/random_rayleigh_rates_V30_E50.csv',
                    delay_path='v30e50/random_rayleigh_delay_V30_E50.csv',
                    title='V30E50, rayleigh')

def plot_algorithm_metrics(data_dict, num_flows, seed):
    """
    Plots three graphs for delay, rate, and active flows for different algorithms and saves the figure.
    
    Parameters:
        data_dict (dict): Dictionary containing algorithm metrics as numpy arrays.
                          Keys should be in the format '<Algorithm>_<Metric>'
                          (e.g., 'GRRL_delay', 'GRRL_rate', 'GRRL_active_flows').
        save_path (str): File path to save the figure.
    """
    save_path = f"metrics_{num_flows}flows_{seed}seed.png"
    algorithms = ['SlotedDIAMOND', 'DIAMOND', 'GRRL', 'DQN+GNN', 'OSPF', 'RandomBL', 'DIAR', 'IACR']
    metrics = ['delay', 'rates', 'active_flows']
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    colors = plt.cm.get_cmap("tab10", len(algorithms))  # Use a colormap for distinct colors
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, algo in enumerate(algorithms):
            key = f"{algo}_{metric}"
            if key in data_dict:
                ax.plot(data_dict[key], label=algo, color=colors(i), alpha=0.8)
        
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":

    plot_all()
    print('ok')
