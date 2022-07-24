import networkx as nx
import numpy as np
import torch


def min_max(a, b):
    return (min(a,b), max(a, b))


def find_r_neighbor_edges(graph:nx.Graph, i, r):
    neighbor_r_edges = set([])
    neighbor_i = list(graph.neighbors(i))
    for j in neighbor_i:
        neighbor_r_edges.add(min_max(i, j))
        for k in neighbor_i[neighbor_i.index(j)+1:]:
            neighbor_r_edges |= set([min_max(edge[0], edge[1]) for edge in sorted(sum(nx.all_simple_edge_paths(graph, j, k, r), start=[]))])

    return neighbor_r_edges


class NeighborSet:
    def __init__(self, edges, exclude_node) -> None:
        self.node_set = set()
        self.edges = edges
        self.exclude_node = exclude_node
        for i, j in self.edges:
            self.node_set.add(i)
            self.node_set.add(j)
        self.node_set.discard(self.exclude_node)
        self.belief = 0.5 + np.random.randn()/100
        pass
    
    def update_belief(self, beta, J, h, neighbor_sets, dtype):
        local_partition_function = 0.0
        local_enumerator = 0.0
        for s in range(2**(len(self.node_set) + 1)):
            config = np.fromiter(np.binary_repr(s, len(self.node_set) + 1), dtype=dtype) * 2 - 1
            local_nodes = [self.exclude_node] + list(self.node_set)
            J_local = J[local_nodes][:, local_nodes]
            h_local = h[local_nodes]
            factor = np.exp(-beta * -(0.5 * config @ J_local @ config + h_local @ config) + np.log([neighbor_sets[k][self.exclude_node].belief if config[j+1] == 1 else 1-neighbor_sets[k][self.exclude_node].belief for j, k in enumerate(self.node_set)]).sum())
            local_partition_function += factor
            if config[0] == 1:
                local_enumerator += factor
            print('-'*5, config, -(0.5 * config @ J_local @ config + h_local @ config), [neighbor_sets[k][self.exclude_node].belief if config[j+1] == 1 else 1-neighbor_sets[k][self.exclude_node].belief for j, k in enumerate(self.node_set)], local_enumerator, local_partition_function)
        self.belief = local_enumerator / local_partition_function


class LoopyBeliefPropagation:
    def __init__(self, graph:nx.Graph, r:int, J, h, beta:float, dtype) -> None:
        self.graph = graph
        self.node_set = sorted(graph.nodes)
        # self.neighbors = {i: list(graph.neighbors(i)) for i in self.node_set}
        self.neighbor_r_nodes = {i : NeighborSet(find_r_neighbor_edges(graph, i, r), i) for i in self.node_set}
        self.neighbor_difference = {}
        self.neighbor_intersection = {}
        for i in self.node_set:
            self.neighbor_difference[i] = {}
            self.neighbor_intersection[i] = {}
            for j in self.neighbor_r_nodes[i].node_set:
                self.neighbor_difference[i][j] = NeighborSet(self.neighbor_r_nodes[i].edges.difference(self.neighbor_r_nodes[j].edges), i)
                self.neighbor_intersection[i][j] = NeighborSet(self.neighbor_r_nodes[i].edges.intersection(self.neighbor_r_nodes[j].edges), j)
        self.dtype = dtype
        self.J = J
        self.h = h
        self.beta = beta
        pass

    def iteration(self, iter_num:int=5):
        for iter in range(iter_num):
            # print('-'*10, iter, '-'*10)
            for i in self.node_set:
                for j in self.neighbor_r_nodes[i].node_set:
                    original_belief = self.neighbor_difference[i][j].belief
                    self.neighbor_difference[i][j].update_belief(self.beta, self.J, self.h, self.neighbor_difference, self.dtype)
                    print((i, j), original_belief, self.neighbor_difference[i][j].belief)

    def internal_energy(self):
        energy_sites = np.zeros(len(self.node_set), dtype=self.dtype)
        for i in range(len(self.node_set)):
            local_node_set = [self.node_set[i]] + list(self.neighbor_r_nodes[self.node_set[i]].node_set)
            assert len(set(local_node_set)) == len(local_node_set)
            local_partition_function = 0.0
            local_enumerator = 0.0
            J_local = self.J[local_node_set][:, local_node_set]
            h_local = self.h[local_node_set]
            print(self.node_set[i], J_local[0, :], local_node_set)
            for s in range(2 ** len(local_node_set)):
                config = np.fromiter(np.binary_repr(s, len(local_node_set)), dtype=self.dtype) * 2 - 1
                # local_nodes = list(local_node_set)
                factor = np.exp(
                    -self.beta * -(0.5 * config @ J_local @ config + h_local @ config) + \
                    np.log([self.neighbor_difference[j][i].belief if config[ind+1] == 1 else 1-self.neighbor_difference[j][i].belief for ind, j in enumerate(self.neighbor_r_nodes[self.node_set[i]].node_set)]).sum()
                )
                energy_local = -(0.5 * config[0] * J_local[0, :] @ config + h_local[0] * config[0])
                print(config, energy_local, factor, -(0.5 * config @ J_local @ config + h_local @ config), [self.neighbor_difference[j][i].belief if config[ind+1] == 1 else 1-self.neighbor_difference[j][i].belief for ind, j in enumerate(self.neighbor_r_nodes[self.node_set[i]].node_set)])
                local_partition_function += factor
                local_enumerator += factor * energy_local
            energy_sites[i] = local_enumerator / local_partition_function
        return energy_sites.sum()

    def marginal(self):
        marginal = np.zeros(len(self.node_set), dtype=self.dtype)
        for i in range(len(self.node_set)):
            local_node_set = [self.node_set[i]] + list(self.neighbor_r_nodes[self.node_set[i]].node_set)
            assert len(set(local_node_set)) == len(local_node_set)
            local_partition_function = 0.0
            local_enumerator = 0.0
            J_local = self.J[local_node_set][:, local_node_set]
            h_local = self.h[local_node_set]
            for s in range(2 ** len(local_node_set)):
                config = np.fromiter(np.binary_repr(s, len(local_node_set)), dtype=self.dtype) * 2 - 1
                # local_nodes = list(local_node_set)
                factor = np.exp(
                    -self.beta * -(0.5 * config @ J_local @ config + h_local @ config) + \
                    np.log([self.neighbor_difference[j][i].belief if config[ind+1] == 1 else 1-self.neighbor_difference[j][i].belief for ind, j in enumerate(self.neighbor_r_nodes[self.node_set[i]].node_set)]).sum()
                )
                print(config, factor, -(0.5 * config @ J_local @ config + h_local @ config), [self.neighbor_difference[j][i].belief if config[ind+1] == 1 else 1-self.neighbor_difference[j][i].belief for ind, j in enumerate(self.neighbor_r_nodes[self.node_set[i]].node_set)])
                local_partition_function += factor
                if config[0] == 1:
                    local_enumerator += factor
            marginal[i] = local_enumerator / local_partition_function
        return marginal
