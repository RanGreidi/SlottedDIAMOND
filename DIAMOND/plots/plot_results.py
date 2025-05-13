import os.path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

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


def plot_algorithm_metrics(data_dict, num_flows, seed, Gloval_env, graph_mode, save_fig):
    """
    Plots three graphs for delay, rate, and active flows for different algorithms and saves the figure.

    Parameters:
        data_dict (dict): Dictionary containing algorithm metrics as numpy arrays.
                          Keys should be in the format '<Algorithm>_<Metric>'
                          (e.g., 'GRRL_delay', 'GRRL_rate', 'GRRL_active_flows').
        save_path (str): File path to save the figure.
    """

    base_path = r"C:\Users\beaviv\DIAMOND-slotted_manual_Plots\with_arrivals"
    # base_path = r'/home/beaviv/DIAMOND-slotted_manual_Plots/with_arrivals'  # with_arrivals   # For claster
    base_path = os.path.join(base_path, f"{graph_mode}", f"{Gloval_env.kwargs['trx_power_mode']}")
    subfolder_name = f"{Gloval_env.num_nodes}_Nodes_{Gloval_env.num_edges // 2}_Edges"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp
    subfolder_path = os.path.join(base_path, subfolder_name, f"{timestamp}_{Gloval_env.num_flows}_Flows")
    # Ensure the directory exists
    os.makedirs(subfolder_path, exist_ok=True)
    save_data_path_pickle = os.path.join(subfolder_path, "data.pkl")  # Save as Pickle
    # Save data_dict as Pickle (optional)
    with open(save_data_path_pickle, "wb") as pickle_file:
        pickle.dump(data_dict, pickle_file)

    save_path = os.path.join(subfolder_path, "Metrics_Plots.png")

    # save_path = f"metrics_{num_flows}flows_{seed}seed.png"
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
    if save_fig:
        plt.savefig(save_path)
    # plt.show()
    # plt.close()
    return subfolder_path


def plot_algorithm_mean_performance(flows, algo_names, algo_rates, algo_delays, subfolder_path, save_fig):
    """
    Plots rates and delays of multiple algorithms against flow numbers in subplots.

    :param flows: List of flow numbers (X-axis).
    :param algo_names: List of algorithm names.
    :param algo_rates: List of lists, where each sublist contains rate values for each algorithm.
    :param algo_delays: List of lists, where each sublist contains delay values for each algorithm.
    :param save_arguments: Boolean flag to save the plot.
    :param subfolder_path: Path to save the figure if save_arguments is True.
    """

    # File paths for saving data
    save_plot_path = os.path.join(subfolder_path, 'algorithm_performance_rates_delays.png')
    save_data_path_pickle = os.path.join(subfolder_path, "performance_data.pkl")  # Save as Pickle

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots side by side

    # Define unique colors and markers
    colors = ['b', 'r', 'darkviolet', 'orange', 'green', 'violet', 'k', 'y']
    markers = ['o', 'p', '+', '*', '^', '+', 'p', 'v']

    # 📌 **Plot Algorithm Rates**
    ax1 = axes[0]  # First subplot for rates
    for idx, algo_name in enumerate(algo_names):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        rates = [algo_rates[i][idx] for i in range(len(flows))]

        ax1.plot(flows, rates, linestyle='-', color=color, marker=marker, markersize=8,
                 markerfacecolor='none', markeredgecolor=color, label=algo_name)

    ax1.set_xticks(flows)
    ax1.set_xticklabels(flows, fontsize=10)
    ax1.set_xlabel("Number of Flows", fontsize=12)
    ax1.set_ylabel("Avg. Flow Rate [Mbps]", fontsize=12)
    ax1.set_title("Algorithm Performance: Rates vs Flows", fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 📌 **Plot Algorithm Delays**
    ax2 = axes[1]  # Second subplot for delays
    for idx, algo_name in enumerate(algo_names):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        delays = [algo_delays[i][idx] for i in range(len(flows))]

        ax2.plot(flows, delays, linestyle='-', color=color, marker=marker, markersize=8,
                 markerfacecolor='none', markeredgecolor=color, label=algo_name)

    ax2.set_xticks(flows)
    ax2.set_xticklabels(flows, fontsize=10)
    ax2.set_xlabel("Number of Flows", fontsize=12)
    ax2.set_ylabel("Avg. Flow Delays [s]", fontsize=12)
    ax2.set_title("Algorithm Performance: Delays vs Flows", fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save and show the plot
    # Ensure the directory exists
    # os.makedirs(subfolder_path, exist_ok=True)
    if save_fig:
        plt.savefig(save_plot_path, dpi=300)

    # plt.show()

    # Save data in Pickle format
    data_to_save = {
        "flows": flows,
        "algo_names": algo_names,
        "algo_rates": algo_rates,
        "algo_delays": algo_delays
    }
    # Save data in Pickle format (for faster loading in Python)
    with open(save_data_path_pickle, "wb") as pickle_file:
        pickle.dump(data_to_save, pickle_file)

if __name__ == "__main__":

    plot_all()
    print('ok')
