import copy
import os.path

import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from DIAMOND.environment.utils import load_pickle_file

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


def plot_algorithm_metrics(data_dict, Gloval_env, graph_mode, save_fig, refinement_steps):
    """
    Plots three graphs for delay, rate, and active flows for different algorithms and saves the figure.

    Parameters:
        data_dict (dict): Dictionary containing algorithm metrics as numpy arrays.
                          Keys should be in the format '<Algorithm>_<Metric>'
                          (e.g., 'GRRL_delay', 'GRRL_rate', 'GRRL_active_flows').
        save_path (str): File path to save the figure.
    """

    # base_path = r"C:\Users\beaviv\DIAMOND-slotted_manual_Plots\without_arrivals"
    base_path = f'/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_{refinement_steps}/with_prediction'  # with_arrivals   # For claster
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


def plot_algorithm_metrics_with_manual_addition(with_pred_folder, with_arrival_folder, save_fig=True):

    # Loading data
    with_pred_data_path = os.path.join(with_pred_folder, "data.pkl")
    with_pred_data = load_pickle_file(with_pred_data_path)

    with_arrival_data_path = os.path.join(with_arrival_folder, "data.pkl")
    with_arrival_data = load_pickle_file(with_arrival_data_path)

    # ------------------ Add Manually with_arrival SlottedDIAMOND data to with_prediction data ----------- #

    # Add Ideal case to data_dict
    with_pred_data['SlottedDIAMOND_Ideal_delay'] = with_arrival_data['SlotedDIAMOND_delay']
    with_pred_data['SlottedDIAMOND_Ideal_rates'] = with_arrival_data['SlotedDIAMOND_rates']
    with_pred_data['SlottedDIAMOND_Ideal_active_flows'] = with_arrival_data['SlotedDIAMOND_active_flows']

    # change name in data dict
    with_pred_data['SlottedDIAMOND_prediction_delay'] = with_pred_data['SlotedDIAMOND_delay']
    with_pred_data['SlottedDIAMOND_prediction_rates'] = with_pred_data['SlotedDIAMOND_rates']
    with_pred_data['SlottedDIAMOND_prediction_active_flows'] = with_pred_data['SlotedDIAMOND_active_flows']

    # remove incorrect name from data
    del with_pred_data['SlotedDIAMOND_delay']
    del with_pred_data['SlotedDIAMOND_rates']
    del with_pred_data['SlotedDIAMOND_active_flows']

    # ---------------------------------------------------------------------------------------------------- #

    save_data_path_pickle = os.path.join(with_pred_folder, "data_with_addition.pkl")  # Save as Pickle
    # Save data_dict as Pickle (optional)
    with open(save_data_path_pickle, "wb") as pickle_file:
        pickle.dump(with_pred_data, pickle_file)

    save_path = os.path.join(with_pred_folder, "Metrics_Plots_after_change.png")

    # save_path = f"metrics_{num_flows}flows_{seed}seed.png"
    algorithms = ['SlottedDIAMOND_prediction', 'SlottedDIAMOND_Ideal', 'DIAMOND', 'GRRL', 'DQN+GNN', 'OSPF', 'RandomBL', 'DIAR', 'IACR']
    metrics = ['delay', 'rates', 'active_flows']

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    colors = plt.cm.get_cmap("tab10", len(algorithms))  # Use a colormap for distinct colors

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, algo in enumerate(algorithms):
            key = f"{algo}_{metric}"
            if key in with_pred_data:
                if i == 1:
                    # Just for SlottedDIAMOND_Ideal
                    ax.plot(with_pred_data[key], label=algo, color=colors(i-1), linestyle='--', alpha=0.8)
                else:
                    ax.plot(with_pred_data[key], label=algo, color=colors(i), alpha=0.8)

        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path)




def plot_algorithm_mean_performance(num_episodes, flows, algo_names, algo_rates, algo_delays, subfolder_path, save_fig):
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
    save_plot_path = os.path.join(subfolder_path, f'algorithm_performance_rates_delays_{num_episodes}.png')
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


def plot_algorithm_mean_performance_with_manual_addition(with_pred_folder, with_arrival_folder, num_episodes=3, save_fig=True):

    #  ----------------- Loading data -------------------------------- #
    with_pred_data_path = os.path.join(with_pred_folder, "performance_data.pkl")
    with_pred_data = load_pickle_file(with_pred_data_path)

    with_arrival_data_path = os.path.join(with_arrival_folder, "performance_data.pkl")
    with_arrival_data = load_pickle_file(with_arrival_data_path)

    # ----------------------------------------------------------------- #
    # with pred data
    flows = with_pred_data['flows']
    algo_names = with_pred_data['algo_names']
    algo_rates = with_pred_data['algo_rates']
    algo_delays = with_pred_data['algo_delays']

    # algo_names[0] = 'SlottedDIAMOND_Prediction'
    algo_names[0] = 'Algo_Prediction'

    # algo_names.insert(1, 'SlottedDIAMOND_Ideal')
    algo_names[1] = 'Algo_Ideal'  # Not using "DIAMOND"

    # ---------------- Add Manually with_arrival SlottedDIAMOND data to with_prediction data ----------- #
    for num_flows_index in range(len(flows)):
        # algo_rates[num_flows_index].insert(1, with_arrival_data['algo_rates'][num_flows_index][0])  # TODO: in algo_rates[num_flows_index]. [0] represents the results for slotted_diamond
        algo_rates[num_flows_index][1] = with_arrival_data['algo_rates'][num_flows_index][0]  # Not using DIAMOND

        # algo_delays[num_flows_index].insert(1, with_arrival_data['algo_delays'][num_flows_index][0])
        algo_delays[num_flows_index][1] = with_arrival_data['algo_delays'][num_flows_index][0]  # Not using DIAMOND

    # ---------------------------------------------------------------------------------------------------- #


    # File paths for saving data
    # save_plot_path = os.path.join(with_pred_folder, f'algorithm_performance_rates_delays_{num_episodes}_after_change.png')
    save_plot_path = os.path.join(with_pred_folder, f'algorithm_performance_rates_delays_{num_episodes}_no_DIAMOND.png')

    # save_data_path_pickle = os.path.join(with_pred_folder, "performance_data_after_change.pkl")  # Save as Pickle
    save_data_path_pickle = os.path.join(with_pred_folder, "performance_data_no_DIAMOND.pkl")  # Save as Pickle

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots side by side

    # Define unique colors and markers
    colors = ['b', 'b', 'r', 'darkviolet', 'orange', 'green', 'violet', 'k', 'y']
    markers = ['o', 'o', 'p', '+', '*', '^', '+', 'p', 'v']

    # 📌 **Plot Algorithm Rates**
    ax1 = axes[0]  # First subplot for rates
    for idx, algo_name in enumerate(algo_names):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        rates = [algo_rates[i][idx] for i in range(len(flows))]

        if algo_name == 'Algo_Ideal':  # 'SlottedDIAMOND_Ideal'
            ax1.plot(flows, rates, linestyle='--', color=color, marker=marker, markersize=8,
                     markerfacecolor='none', markeredgecolor=color, label=algo_name)

        else:
            ax1.plot(flows, rates, linestyle='-', color=color, marker=marker, markersize=8,
                     markerfacecolor='none', markeredgecolor=color, label=algo_name)

    ax1.set_xticks(flows)
    ax1.set_xticklabels(flows, fontsize=10)
    ax1.set_xlabel("Number of Flows", fontsize=12)
    ax1.set_ylabel("Avg. Flow Rate [Mbps]", fontsize=12)
    # ax1.set_title("Algorithm Performance: Rates vs Flows", fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 📌 **Plot Algorithm Delays**
    ax2 = axes[1]  # Second subplot for delays
    for idx, algo_name in enumerate(algo_names):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        delays = [algo_delays[i][idx] for i in range(len(flows))]

        if algo_name == 'Algo_Ideal':  # 'SlottedDIAMOND_Ideal'
            ax2.plot(flows, delays, linestyle='--', color=color, marker=marker, markersize=8,
                 markerfacecolor='none', markeredgecolor=color, label=algo_name)

        else:
            ax2.plot(flows, delays, linestyle='-', color=color, marker=marker, markersize=8,
                     markerfacecolor='none', markeredgecolor=color, label=algo_name)

    ax2.set_xticks(flows)
    ax2.set_xticklabels(flows, fontsize=10)
    ax2.set_xlabel("Number of Flows", fontsize=12)
    ax2.set_ylabel("Avg. Flow Delays [µs]", fontsize=12)
    # ax2.set_title("Algorithm Performance: Delays vs Flows", fontsize=14)
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


def plot_algorithm_mean_performance_for_paper(with_pred_folder,num_episodes=3, save_fig=True):

    #  ----------------- Loading data -------------------------------- #
    with_pred_data_path = os.path.join(with_pred_folder, "performance_data.pkl")
    with_pred_data = load_pickle_file(with_pred_data_path)

    # ----------------------------------------------------------------- #
    # with pred data
    flows = with_pred_data['flows']
    algo_names = with_pred_data['algo_names']
    algo_rates = with_pred_data['algo_rates']
    algo_delays = with_pred_data['algo_delays']

    # algo_names[0] = 'SlottedDIAMOND_Prediction'
    algo_names[0] = 'DPIR'

    # algo_names.insert(1, 'SlottedDIAMOND_Ideal')
    algo_names.pop(1)  # Not using "DIAMOND"

    og_algos = ['SlotedDIAMOND', 'GRRL', 'DQN+GNN', 'OSPF', 'RandomBL', 'DIAR', 'IACR']

    algo_names = ['DPIR', 'GRRL', 'IACR', 'DQN+GNN', 'OSPF', 'DIAR', 'RandomBL']  #'DQN+GNN'
    algo_names_delays = ['RandomBL', 'OSPF', 'DIAR', 'DQN+GNN', 'IACR', 'GRRL', 'DPIR']  #'DQN+GNN'

    # ---------------- Add Manually with_arrival SlottedDIAMOND data to with_prediction data ----------- #
    for num_flows_index in range(len(flows)):

        algo_rates[num_flows_index].pop(1)  # Not using DIAMOND
        dqn_data = copy.deepcopy(algo_rates[num_flows_index][2])
        ospf_data = copy.deepcopy(algo_rates[num_flows_index][3])
        rb_data = copy.deepcopy(algo_rates[num_flows_index][4])
        diar_data = copy.deepcopy(algo_rates[num_flows_index][5])
        iacr_data = copy.deepcopy(algo_rates[num_flows_index][6])

        algo_rates[num_flows_index][2] = iacr_data
        algo_rates[num_flows_index][3] = dqn_data
        algo_rates[num_flows_index][4] = ospf_data
        algo_rates[num_flows_index][5] = diar_data
        algo_rates[num_flows_index][6] = rb_data


        # algo_delays[num_flows_index].insert(1, with_arrival_data['algo_delays'][num_flows_index][0])
        algo_delays[num_flows_index].pop(1)  # Not using DIAMOND

        dpir_data = copy.deepcopy(algo_delays[num_flows_index][0])
        grrl_data = copy.deepcopy(algo_delays[num_flows_index][1])

        dqn_data = copy.deepcopy(algo_delays[num_flows_index][2])
        ospf_data = copy.deepcopy(algo_delays[num_flows_index][3])
        rb_data = copy.deepcopy(algo_delays[num_flows_index][4])
        diar_data = copy.deepcopy(algo_delays[num_flows_index][5])
        iacr_data = copy.deepcopy(algo_delays[num_flows_index][6])

        algo_delays[num_flows_index][0] = rb_data
        algo_delays[num_flows_index][1] = ospf_data
        algo_delays[num_flows_index][2] = diar_data
        algo_delays[num_flows_index][3] = dqn_data
        algo_delays[num_flows_index][4] = iacr_data
        algo_delays[num_flows_index][5] = grrl_data
        algo_delays[num_flows_index][6] = dpir_data
    # ---------------------------------------------------------------------------------------------------- #


    # File paths for saving data
    # save_plot_path = os.path.join(with_pred_folder, f'algorithm_performance_rates_delays_{num_episodes}_after_change.png')
    save_plot_path = os.path.join(with_pred_folder, f'algorithm_performance_rates_delays_{num_episodes}_for_paper2.png')

    # save_data_path_pickle = os.path.join(with_pred_folder, "performance_data_after_change.pkl")  # Save as Pickle
    save_data_path_pickle = os.path.join(with_pred_folder, "performance_data_for_paper.pkl")  # Save as Pickle

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create two subplots side by side

    # Define unique colors and markers
    colors = ['b', 'r', 'darkviolet', 'green', 'orange', 'violet', 'k']
    colors_delays = ['k', 'orange', 'violet', 'green', 'darkviolet', 'r', 'b']

    markers = ['o', 'p', '+', '^', '*', '+', 'p']
    markers_delays = ['p', '*', '+', '^', '+', 'p', 'o']

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
    # ax1.set_title("Algorithm Performance: Rates vs Flows", fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 📌 **Plot Algorithm Delays**
    ax2 = axes[1]  # Second subplot for delays
    for idx, algo_name in enumerate(algo_names_delays):
        color = colors_delays[idx % len(colors_delays)]
        marker = markers_delays[idx % len(markers_delays)]
        delays = [algo_delays[i][idx] for i in range(len(flows))]

        ax2.plot(flows, delays, linestyle='-', color=color, marker=marker, markersize=8,
                 markerfacecolor='none', markeredgecolor=color, label=algo_name)

    ax2.set_xticks(flows)
    ax2.set_xticklabels(flows, fontsize=10)
    ax2.set_xlabel("Number of Flows", fontsize=12)
    ax2.set_ylabel("Avg. Flow Delays [timesteps]", fontsize=12)
    # ax2.set_title("Algorithm Performance: Delays vs Flows", fontsize=14)
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

    # with_pred_folder = r'C:\Users\beaviv\DIAMOND-slotted_manual_Plots\with_prediction\random\equal\80_Nodes_120_Edges\20250528_205647_120_Flows'
    #
    # with_arrival_folder = r'C:\Users\beaviv\DIAMOND-slotted_manual_Plots\with_arrivals\random\equal\80_Nodes_120_Edges\20250613_153038_120_Flows'
    #
    # plot_algorithm_mean_performance_with_manual_addition(with_pred_folder, with_arrival_folder)

    with_pred_folder = r'/sise/home/beaviv/DIAMOND-slotted_manual_Plots/refinement_steps_10/with_prediction/random/equal/70_Nodes_140_Edges/20250720_165935_120_Flows/'

    plot_algorithm_mean_performance_for_paper(with_pred_folder)

    print(f'finished')
