import random

from DIAMOND.environment.graph_env_power import GraphEnvPower as GraphEnv
from DIAMOND.environment.utils import *
from DIAMOND.environment.Traffic_Probability_Model import Traffic_Probability_Model
from DIAMOND.environment.Traffic_Probability_HawkesModel import HawkesModel


def _get_random_flows_no_arrivals(num_nodes, num_flows, demands=[100], seed=1):
    """
    generates random flows
    :param num_nodes: number of nodes in the communication graph
    :param num_flows: number of flows in the communication graph
    :param demands: list of packets demands for flows to choose from
    :param seed: random seed
    :return: list of flows as (src, dst, pkt)
    """
    flow_demand = [(1000/(pow(i, 3))) for i in range(1, num_flows+1)] #[2, 20, 50 ,100, 200, 9, 7, 500 ,200, 1000][::-1]
    random.seed(seed)
    result = []
    for name in range(num_flows):
        src, dst = random.sample(range(num_nodes), 2)
        f = {"source": src,
             "destination": dst,
             "packets": random.choice(demands), #random.choice(demands),  # flow_demand[name],
             "name": name}  # to be changes in the future to markov.state
        result.append(f)
    return result


def _get_random_flows_with_arrivals(num_nodes, num_flows, demands, slot_duration, num_slots, pkt_arrival_sample_rate, HawkesParams, seed=1):
    """
    generates random flows
    :param num_nodes: number of nodes in the communication graph
    :param num_flows: number of flows in the communication graph
    :param demands: list of packets demands for flows to choose from
    :param seed: random seed
    :return: list of flows as (src, dst, pkt)
    """

    # py_rng = random.Random(seed)
    random.seed(seed)
    flow_demand = demands

    flows = []
    flows_statistics = []

    types = ['elephent', 'mice']
    for name in range(num_flows):
        src, dst = random.sample(range(num_nodes), 2)

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
                                        seed=seed)

        f = {"source": src,
             "destination": dst,
             "packets": flow_statistics.initial_count if HawkesParams['allow_Hawkes_arrivals'] else flow_demand[name], # flow_statistics.initial_count if HawkesParams['allow_Hawkes_arrivals'] else flow_demand[name] ,random.choice(demands)
             "name": name}  # to be changes in the future to markov.state

        flows.append(f)
        flows_statistics.append(flow_statistics)

    # for flow in flows:
    #     flow_statistics = HawkesModel(  # alpha * exp(-beta*t)
    #                                     lambda0=HawkesParams['lambda0'],
    #                                     alpha=HawkesParams['alpha'],
    #                                     beta=HawkesParams['beta'],
    #
    #                                     source=flow["source"],
    #                                     destination=flow["destination"],
    #                                     flow_name=flow["name"],
    #                                     num_slots=num_slots,
    #                                     slot_duration=slot_duration,
    #                                     history_num_slots=HawkesParams['history_num_slots'],
    #
    #                                     type=random.choice(types),  #  type='elephent' if name < HawkesParams['elephent_flows_num'] else 'mice'
    #                                     mice_scaler=HawkesParams['mice_scaler'],
    #                                     elephent_scaler=HawkesParams['elephent_scaler'],
    #
    #                                     seed=seed)
    #     flows_statistics.append(flow_statistics)

    return flows, flows_statistics


def generate_env(num_nodes=10,
                 num_edges=20,

                 num_flows=2,
                 min_flow_demand=100,
                 max_flow_demand=200,

                 num_actions=4,

                 min_capacity=1000,
                 max_capacity=1000,

                 direction="minimize",
                 reward_balance=0.8,
                 seed=37,
                 graph_mode='random',
                 slot_duration=None,
                 num_slots=None,
                 **kwargs):
    # assert graph_mode.lower() in ['random', 'nsfnet', 'geant']
    assert graph_mode.lower() in ['random', 'random_internet', 'nsfnet', 'geant', 'grid', 'irregular_grid_8x8', 'irregular_grid_6x6']

    # 1. create graph
    if graph_mode == 'random':
        adjacency, positions = generate_random_graph(n=num_nodes, e=num_edges, seed=seed)
    elif graph_mode == 'random_internet':
        adjacency, positions = generate_random_internet_graph(n_total=num_nodes, n_clusters=3, p_intra=0.4, p_inter=0.02, seed=seed)        
    elif graph_mode == 'nsfnet':
        adjacency, positions = create_nsfnet_graph()
        num_nodes = 14
    elif graph_mode == 'geant':
        adjacency, positions = create_geant2_graph()
        num_nodes = 24
    elif graph_mode == 'grid':
        adjacency, positions = gen_grid_graph(n=num_nodes)
        num_nodes = num_nodes ** 2
    elif graph_mode == 'irregular_grid_8x8':
        adjacency, positions = gen_grid_graph(n=8, special_edges_add=[[[1, 0], [2, 2]],
                                                                      [[3, 1], [4, 5]],
                                                                      [[0, 2], [3, 3]],
                                                                      [[4, 1], [6, 2]],
                                                                      [[4, 2], [5, 4]],
                                                                      [[0, 4], [3, 6]],
                                                                      [[4, 3], [7, 4]]
                                                                      ])
        num_nodes = 64
    elif graph_mode == 'irregular_grid_6x6':
        adjacency, positions = gen_grid_graph(n=6, special_edges_remove=[[[1, 0], [1, 1]],
                                                                         [[0, 1], [1, 1]],
                                                                         [[2, 0], [2, 1]],
                                                                         [[2, 1], [3, 1]],
                                                                         [[2, 3], [3, 3]],
                                                                         [[2, 4], [3, 4]],
                                                                         [[2, 5], [3, 5]],
                                                                         [[3, 0], [3, 1]],
                                                                         [[4, 0], [4, 1]],
                                                                         [[4, 1], [5, 1]]
                                                                         ])
        num_nodes = 36

    # 2. create random flows
    delta = 10
    packets = list(range(int(min_flow_demand), int(max_flow_demand) + delta, delta))
    # demands = [random.choice(packets) for _ in range(num_flows)]
    HawkesParams = kwargs.get('HawkesParams')
    pkt_arrival_sample_rate = kwargs.get('pkt_arrival_sample_rate')
    flows, flows_statistics = _get_random_flows_with_arrivals(num_nodes=num_nodes, num_flows=num_flows, demands=packets, slot_duration=slot_duration, num_slots=num_slots, pkt_arrival_sample_rate=pkt_arrival_sample_rate, HawkesParams=HawkesParams,  seed=seed)

    # 3. generate env instance
    # capacity_matrix = np.random.randint(low=min_capacity, high=max_capacity + 1, size=(num_nodes, num_nodes))
    np.random.seed(seed)
    capacity_matrix = np.random.randint(low=min_capacity, high=max_capacity + 1, size=adjacency.shape)

    # interference matrix
    interference_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    env = GraphEnv(adjacency_matrix=adjacency,
                   bandwidth_matrix=capacity_matrix,
                   interference_matrix=interference_matrix,
                   node_positions=positions,
                   flows=flows,
                   k=num_actions,
                   direction=direction,
                   reward_balance=reward_balance,
                   seed=seed,
                   **kwargs)

    env_configurations = dict(adjacency_matrix=adjacency,
                                bandwidth_matrix=capacity_matrix,
                                interference_matrix=interference_matrix,
                                node_positions=positions,
                                flows=flows,
                                k=num_actions,
                                direction=direction,
                                reward_balance=reward_balance,
                                seed=seed,
                                **kwargs)

    return env, env_configurations, flows_statistics
