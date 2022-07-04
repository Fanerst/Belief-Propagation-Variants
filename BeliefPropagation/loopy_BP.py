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
        self.belief = 0.5
        pass
    
    def update_belief(self, beta, J, h, neighbor_sets, dtype):
        local_partition_function = 0.0
        local_enumerator = 0.0
        for s in range(2**(len(self.node_set) + 1)):
            config = np.fromiter(np.binary_repr(s, len(self.node_set) + 1), dtype=dtype)
            local_nodes = [self.exclude_node] + list(self.node_set)
            J_local = J[local_nodes][:, local_nodes]
            h_local = h[local_nodes]
            factor = -beta * (config @ J_local @ config + h_local @ config) + config[1:] @ np.log([neighbor_sets[k][self.exclude_node].belief if config[j+1] == 1 else 1-neighbor_sets[k][self.exclude_node].belief for j, k in enumerate(self.node_set)])
            local_partition_function += factor
            if config[0] == 1:
                local_enumerator += factor
        new_belief = local_enumerator / local_partition_function
        return new_belief

class LoopyBeliefPropagation:
    def __init__(self, graph:nx.Graph, r:int, J, h, dtype) -> None:
        self.graph = graph
        self.node_set = sorted(graph.nodes)
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
        pass

    def iteration(self, beta:float):
        for i in self.node_set:
            for j in self.neighbor_r_nodes[i].node_set:
                original_belief = self.neighbor_difference[i][j].belief
                self.neighbor_difference[i][j].update_belief(beta, self.J, self.h, self.neighbor_difference, self.dtype)
                print((i, j), original_belief, self.neighbor_difference[i][j].belief)


if __name__ == '__main__':
    dtype = np.float64
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (1, 12), (1, 13), (2, 3), (2, 15), (3, 4), (3, 9), (3, 10),
        (4, 5), (5, 6), (5, 7), (5, 8), (9, 10), (9, 11), (12, 13), (12, 14), (13, 14), (15, 16), (15, 17),
        # (12, 15)
    ]
    g = nx.Graph()
    g.add_edges_from(edges)
    print(g.number_of_nodes(), g.number_of_edges())
    n = g.number_of_nodes()
    for i in g.nodes:
        print(i, list(g.neighbors(i)))
    print(find_r_neighbor_edges(g, 0, 1))

    weights = np.ones(len(edges), dtype=dtype)
    fields = np.zeros(n, dtype=dtype)
    J = np.zeros([n, n], dtype=dtype)
    idx = np.array(edges)
    J[idx[:, 0], idx[:, 1]] = weights
    J[idx[:, 1], idx[:, 0]] = weights

    lbp = LoopyBeliefPropagation(g, 2, J, fields, dtype)
    for i in lbp.node_set:
        print(i, lbp.neighbor_r_nodes[i].node_set, lbp.neighbor_r_nodes[i].edges)
        for j in lbp.neighbor_r_nodes[i].node_set:
            print((i,j), lbp.neighbor_difference[i][j].exclude_node, lbp.neighbor_difference[i][j].node_set, lbp.neighbor_difference[i][j].edges)
            print((i,j), lbp.neighbor_intersection[i][j].exclude_node, lbp.neighbor_intersection[i][j].node_set, lbp.neighbor_intersection[i][j].edges)
    
    lbp.iteration(beta=1.0)