import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import os
import random
import json
import torch
import time
import threading
import pickle


def init_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)


def update_graph_weight_from_path(G: nx.Graph,
                                  P: list,
                                  x=None):
    """
    @param G: graph
    @param P: path in G (list of nodes)
    @param x: factor to be added to graph weights

    @precondition: x must be power of two

    """
    # copy graph to avoid changing the original one
    A = G.copy()

    if x is None:
        x = 2 ** nx.algorithms.distance_measures.diameter(G)

    d = P[-1]  # destination node
    V = set(G.nodes)  # all nodes in G
    U = {d}  # finished nodes
    S = set()  # unfinished nodes

    # add weights to all edges of P and their first neighbors
    for i in range(len(P) - 1):
        u = P[i]
        v = P[i + 1]
        A.get_edge_data(u, v)['weight'] += x
        for j in A.neighbors(u):
            if j != v and j not in P:
                A.get_edge_data(u, j)['weight'] += x / 2
                S.add(j)
        U.add(u)

    # add weights to d's edges
    for j in A.neighbors(d):
        if j not in P:
            A.get_edge_data(d, j)['weight'] += x / 2
            S.add(j)

    x /= 4
    # add weights iteratively to all other edges w.r.t their distance to P
    while U != V and len(S) > 0:
        T = set()
        for u in S:
            for v in A.neighbors(u):
                if v in V - U:
                    A.get_edge_data(u, v)['weight'] += x
                    if v not in S:
                        T.add(v)
            U.add(u)
        x /= 2
        S = T.copy()

    return A


def show_graph(G: nx.Graph):
    """ draw graph"""
    plt.figure()
    # Create positions of all nodes
    pos = nx.spring_layout(G)
    # Draw the graph according to node positions
    nx.draw_networkx(G, pos, with_labels=True, node_color="tab:blue")
    # Create edge labels
    labels = {(u, v): e['weight'] for u, v, e in G.edges(data=True)}
    # Draw edge labels according to node positions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.axis('off')
    plt.show()


def get_k_paths(G: nx.Graph,
                s: int,
                d: int,
                k: int,
                x=None,
                method='dijkstra'):
    """
    Returns k (shortest) most separated paths in G from s to d
    """
    k_sp = []
    A = G.copy()

    # init weights
    for u, v, attr in A.edges(data=True):
        attr['weight'] = 1

    # all_possible_paths = list(nx.shortest_simple_paths(G, source=s, target=d, weight='weight'))
    gen_path = nx.shortest_simple_paths(G, source=s, target=d, weight='weight')
    # not exactly all possible paths but enough options to choose from, just to speed things
    all_possible_paths = []
    for _ in range(k ** 2):
        try:
            all_possible_paths.append(next(gen_path))
        except StopIteration:
            break

    while len(k_sp) < k and len(k_sp) < len(all_possible_paths):
        try:
            p = nx.shortest_path(A, source=s, target=d, weight='weight', method=method)
        except:
            break
        if p not in k_sp:
            k_sp.append(p)
        else:
            # if the chosen path already exists, try to take one of the other paths (Yen's algorithm)
            sp = nx.shortest_simple_paths(G, source=s, target=d, weight='weight')  # generator, doesn't actually calculates all paths at ones
            next(sp)  # first item the p, so we skip it
            for _ in range(k):
                try:
                    p = next(sp)
                except StopIteration:
                    break
                if p not in k_sp:
                    k_sp.append(p)
                    break
        A = update_graph_weight_from_path(A, p, x)

    return k_sp


def one_link_transmission(c, packets):
    """one transmission on channel with capacity c for flows with packets
    @param c: link's capacity
    @param packets: list of flows load (packets)

    returns list of packets remaining to be sent for each flow
    """
    if sum(packets) <= c:
        return [0] * len(packets)

    count = len(packets) - len([p for p in packets if p == 0])
    q = c // count
    r = c % count
    new_packs = []
    for p in packets:
        new_packs.append(p - min(p, q))
        if p > 0:
            r += q - min(p, q)
    i = 0
    while r > 0 and i < len(new_packs):
        if new_packs[i] > 0:
            rem = min(r, new_packs[i])
            new_packs[i] -= rem
            r -= rem
        i += 1
    return new_packs


def link_queue_history_using_mac_protocol(c, packets):
    """
    returns history of transmission for one or more flows on a channel with capacity c
    assuming some MAC protocol exists on the link.
    When multiple agents transmit together on a shared link, the split the capacity until all have been sent

    @param c: link's capacity
    @param packets: list of flows load (packets)
    """
    hist = []
    packs = packets.copy()
    while sum(packs) > 0:
        hist.append(packs)
        packs = one_link_transmission(c, packs)
    return np.array(hist)


def calc_transmission_rate(link_mac):
    """
    calculates the transmission rate (based on bottleneck)
    :param link_mac:
    :return:
    """
    if len(link_mac) == 1:
        return link_mac[0]
    trans = np.stack([link_mac[i] - link_mac[i+1] for i in range(len(link_mac)-1)], axis=0)
    # masked_trans = np.where(trans > 0, trans, np.inf)
    # inf_counts = np.sum(np.isinf(masked_trans), axis=0)
    # if np.any(inf_counts > 2):  # Check if any element is np.inf
    #     raise ValueError("Array contains at least two np.inf value!")        
    return trans[0] #np.min(trans, axis=0)


def generate_random_graph(n, e, seed=None):
    """
    generate random graph in [0,1]^2 contains of n nodes
    :param n: number of nodes in the graph
    :param n: number of edges in the graph
    :return: nx.Graph
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # adjacency matrix
    adjacency = np.zeros((n, n))
    # sample nodes positions in [0,1]^2
    positions = np.random.uniform(low=0, high=1, size=(n, 2))

    # connect nodes via path to make sure the graph is connected
    for i in range(n-1):
        adjacency[i, i+1] = 1
        adjacency[i+1, i] = 1

    # sample random edges
    for _ in range(e-n+1):
        i, j = random.sample(range(n), 2)
        if np.array_equal(adjacency, np.ones((n,n))-np.eye(n)):
            print(f"full graph has {(n ** 2 - n) // 2} ({n ** 2 - n}) edges instead of {e} ({e*2})")
            break
        while adjacency[i, j]:
            i, j = random.sample(range(n), 2)
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    g = nx.from_numpy_array(adjacency, create_using=nx.Graph)
    for u, v, attr in g.edges(data=True):
        # set node position
        if g.nodes[u] == {}:
            g.nodes[u]['pos'] = positions[u]
        if g.nodes[v] == {}:
            g.nodes[v]['pos'] = positions[v]

    # return g
    return adjacency, positions


def generate_random_internet_graph(n_total, n_clusters=3, p_intra=0.4, p_inter=0.02, seed=None):
    """
    Generate a random clustered communication-like graph that is fully connected.

    Parameters
    ----------
    n_total : int
        Total number of nodes in the graph.
    n_clusters : int
        Number of subnetworks (clusters).
    p_intra : float
        Probability of intra-cluster edges.
    p_inter : float
        Probability of extra inter-cluster edges beyond the guaranteed ones.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    A : np.ndarray
        Adjacency matrix of shape (n_total, n_total)
    pos : np.ndarray
        Node position matrix of shape (n_total, 2)
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    # Divide nodes among clusters
    cluster_sizes = np.full(n_clusters, n_total // n_clusters)
    cluster_sizes[:n_total % n_clusters] += 1
    cluster_nodes = []
    start = 0

    for size in cluster_sizes:
        nodes = range(start, start + size)
        start += size
        cluster_nodes.append(list(nodes))

        # Create dense subgraph
        subG = nx.erdos_renyi_graph(size, p_intra, seed=int(rng.integers(1e9)))
        mapping = {old: new for old, new in zip(subG.nodes(), nodes)}
        subG = nx.relabel_nodes(subG, mapping)
        G = nx.compose(G, subG)

    # Step 1: Ensure connectivity between clusters (chain)
    for i in range(n_clusters - 1):
        node_i = rng.choice(cluster_nodes[i])
        node_j = rng.choice(cluster_nodes[i + 1])
        G.add_edge(node_i, node_j)

    # Step 2: Add a few random inter-cluster edges (optional)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if rng.random() < p_inter:
                node_i = rng.choice(cluster_nodes[i])
                node_j = rng.choice(cluster_nodes[j])
                G.add_edge(node_i, node_j)

    # Step 3: Double-check connectivity (rarely needed, but ensures safety)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for c1, c2 in zip(components[:-1], components[1:]):
            node1 = rng.choice(list(c1))
            node2 = rng.choice(list(c2))
            G.add_edge(node1, node2)

    # Generate clustered positions
    centers = rng.random((n_clusters, 2)) * 0.8 + 0.1
    pos = np.zeros((n_total, 2))
    for i, nodes in enumerate(cluster_nodes):
        for node in nodes:
            pos[node] = centers[i] + 0.05 * rng.standard_normal(2)

    # Convert to adjacency and position arrays
    A = nx.to_numpy_array(G, nodelist=range(n_total))
    return A, pos


def create_geant2_graph():
    """
    nodes and edges from: https://github.com/knowledgedefinednetworking/DRL-GNN/blob/master/DQN/gym-environments/gym_environments/envs/environment1.py
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12, 19), (12, 21),
         (14, 15), (15, 16), (16, 17), (17, 18), (18, 21), (19, 23), (21, 22), (22, 23)])
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = nx.spring_layout(Gbase, seed=1234)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def create_nsfnet_graph():
    """
    nodes and edges from: https://github.com/knowledgedefinednetworking/DRL-GNN/blob/master/DQN/gym-environments/gym_environments/envs/environment1.py
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    G = nx.from_numpy_array(A, create_using=nx.Graph)  # g = nx.from_numpy_array(adjacency, create_using=nx.Graph)
    pos = nx.spring_layout(G, seed=124)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def gen_grid_graph(n, special_edges_add=None, special_edges_remove=None):
    """
    Generates adjacency matrix for an nxn grid network
    :param n: number of nodes in each edge (total n**2 nodes)
    :param special_edges_add: list(list(tuple)) indicates extra edges to add. example: [[(0,0), (3, 2)]]
    :param special_edges_remove: list(list(tuple)) indicates edges to remove. example: [[(0,0), (0, 1)]]
    :return: adjacency matrix of a mesh grid network
    """
    N = int(n ** 2)
    A = np.zeros((N, N))

    # gen basic grid
    for i in range(n):
        for j in range(n):
            my_id = i * n + j
            if i - 1 >= 0:
                neighbor_id = (i - 1) * n + j
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1
            if i + 1 <= n - 1:
                neighbor_id = (i + 1) * n + j
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1
            if j - 1 >= 0:
                neighbor_id = i * n + (j - 1)
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1
            if j + 1 <= n - 1:
                neighbor_id = i * n + (j + 1)
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1

    # add specified edges
    if special_edges_add is not None:
        for i, j in special_edges_add:
            i_id = i[0] * n + i[1]
            j_id = j[0] * n + j[1]
            A[i_id, j_id] = 1
            A[j_id, i_id] = 1

    # remove specified edges
    if special_edges_remove is not None:
        for i, j in special_edges_remove:
            i_id = i[0] * n + i[1]
            j_id = j[0] * n + j[1]
            A[i_id, j_id] = 0
            A[j_id, i_id] = 0

    # nodes positions in [0,1]^2
    x = np.linspace(0, 1, n)
    positions = np.array([[i, j] for i in x for j in x[::-1]])

    return A, positions

def get_flow_by_name(dicts_list, name):
    for d in dicts_list:
        if d['name'] == name:
            return d
    return None

def run_predefined_actions(env, actions):
        '''      
        the actions in actions list needs to be in the order of all active flows, 
        for example: if env.flows has 3 flows with the names 2, 3, 4, we need to take index 2 3 and 4 from the inital actions (initail actions is orders according to inital global flows, thus all flows must be active
        at first run).
        '''
        if type(actions[0]) != tuple:
            current_slot_actions = []
            for active_flow in env.flows:
                current_slot_actions.append(actions[active_flow['name']])
        if type(actions[0]) == tuple:
            current_slot_actions = []
            for active_flow in env.flows:
                current_slot_actions.append(actions[active_flow['name']][1])

        if type(actions[0]) == tuple:
            # OSPF or DIAR (where actions are as paths not indexs)
            action_as_idx=False
            eval_paths=True
        else:
            action_as_idx=True
            eval_paths=False
        env.reset()
        env.eval_all(current_slot_actions, eval_paths)
        rate_date = env.get_rates_data()
        delay_data = env.get_delay_data(action_as_idx=action_as_idx)
        return rate_date, delay_data
        

def run_Slotted_predefined_actions(env, actions, slot):
    '''  
    This function is identical to run_predefined_actions, but it allocates a new flow with other method (OSPF), when the given actions does not match the
    incoming flows to run - suitable for the slottted algorithms
    '''
    current_action_from_slotted_algo = actions[slot] # Actions is in the form of: {flow_name_1:flow_action_1 , flow_name_2:flow_action_2 ... }
    current_slot_actions = []
    for active_flow in env.flows:
        # if target flow is in current_action_from_slotted_algo
        if active_flow['name'] in current_action_from_slotted_algo:
            current_slot_actions.append(current_action_from_slotted_algo[active_flow['name']])
        # else allocate action randomly or perform ospf
        else:
            # random decision each slot. TODO: add radnom/OSPF/whatever
            current_slot_actions.append(random.randint(0, env.k-1))

    # Note! actions - this list needs to be ordered as env.flows is ordered! for example:
    #  if env.flows is [flow_name_2, flow_name_3, flow_name_4] than actions needs to be 
    # ordered as [decision_for_flow_name_2, decision_for_flow_name_3, decision_for_flow_name_4]

    action_as_idx = True
    eval_paths = False
    env.reset()
    env.eval_all(current_slot_actions, eval_paths)
    rate_date = env.get_rates_data()
    delay_data = env.get_delay_data(action_as_idx=action_as_idx)
    return rate_date, delay_data


def save_arguments_to_file(filename, **kwargs):
    """
    Save the provided keyword arguments to a JSON file.
    :param filename: Name of the file to save the arguments.
    :param kwargs: Arguments to save.
    """
    with open(filename, 'w') as file:
        json.dump(kwargs, file, indent=4)


def load_json_file(filename):
    with open(filename, 'r') as file:
        arguments = json.load(file)
    return arguments


def load_pickle_file(filename):
    with open(filename, "rb") as pickle_file:
        data_dict = pickle.load(pickle_file)
    return data_dict

def keep_gpu_active_heavy(interval_seconds=60):
    """
    Keeps the GPU busy by running a moderately heavy matrix op loop.
    Should trick idle detectors by holding GPU usage longer.
    """
    def loop():
        counter = 0
        dummy = torch.randn(2048, 2048, device="cuda")  # ~32MB tensor
        while True:
            try:
                start = time.time()
                for _ in range(50):  # Perform multiple ops back to back
                    dummy = torch.relu(torch.matmul(dummy, dummy))
                torch.cuda.synchronize()
                duration = time.time() - start
                counter += 1
                print(f"[keep_gpu_active] Ping #{counter} | Duration: {duration:.2f}s at {time.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"[keep_gpu_active] GPU activity failed: {e}")
            time.sleep(interval_seconds)

    threading.Thread(target=loop, daemon=True).start()