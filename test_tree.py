from numba import jit
import networkx as nx
import torch
import numpy as np

@jit()
def exact_config(D):
    config = np.empty((2 ** D, D))
    for i in range(2 ** D - 1, -1, -1):
        num = i
        for j in range(D - 1, -1, -1):
            config[i, D - j - 1] = num // 2 ** j
            if num - 2 ** j >= 0:
                num -= 2 ** j

    return config * 2.0 - 1.0

@jit()
def exact_config01(D):
    config = np.empty((2 ** D, D))
    for i in range(2 ** D - 1, -1, -1):
        num = i
        for j in range(D - 1, -1, -1):
            config[i, D - j - 1] = num // 2 ** j
            if num - 2 ** j >= 0:
                num -= 2 ** j

    return config


class Exact:
    def __init__(self, G, J, h, beta, device, seed):
        self.G = G
        self.beta = beta
        self.device = device
        self.seed = seed
        self.dtype = torch.float64
        self.D = self.G.number_of_nodes()
        self.J = J
        self.h = h

    def FVS_decomposition(self):
        rng = np.random.RandomState(self.seed)
        G1 = self.G.copy()
        fvs = []
        while G1.number_of_nodes():
            flag = True
            while flag:
                temp = []
                flag = False
                for i in list(G1.nodes):
                    if G1.degree[i] <= 1:
                        temp.append(i)
                        flag = True
                if not flag:
                    break
                G1.remove_nodes_from(temp)
            if not G1.number_of_nodes():
                break
            degrees = np.array(G1.degree)
            degree_max = degrees[rng.choice(np.where(degrees[:, 1] == max(degrees[:, 1]))[0]), 0]
            fvs.append(degree_max)
            G1.remove_node(degree_max)

        tree_hierarchy, tree_order = self.tree_hierarchize(fvs)

        return fvs, tree_order, tree_hierarchy

    def tree_hierarchize(self, frozen_nodes):
        G1 = self.G.copy()
        G1.remove_nodes_from(frozen_nodes)
        ccs = list(nx.connected_components(G1))
        trees = {}.fromkeys(np.arange(len(ccs)))
        for key in trees.keys():
            trees[key] = []
        for l in range(len(ccs)):
            tree = self.G.subgraph(ccs[l]).copy()
            while tree.number_of_nodes():
                temp = []
                for j in list(tree.nodes):
                    if tree.number_of_nodes() == 1 or tree.number_of_nodes() == 2:
                        temp.append(j)
                        break
                    if tree.degree[j] == 1:
                        temp.append(j)
                tree.remove_nodes_from(temp)
                trees[l].append(temp)

        tree_order = []
        tree_hierarchy = []
        max_length = 0
        for key in trees.keys():
            l = len(trees[key])
            if l >= max_length:
                max_length = l

        for j in range(max_length):
            tree_hierarchy.append([])
            for key in trees.keys():
                if j < len(trees[key]):
                    tree_hierarchy[j] += trees[key][j]
            tree_order += tree_hierarchy[j]

        return tree_hierarchy, tree_order

    def effective_energy(self, sample, frozen_nodes, tree_order, tree_hierarchy):
        h = sample.matmul(self.J[frozen_nodes, :]) + self.h
        tree_energy = torch.zeros(sample.shape[0], device=self.device, dtype=sample.dtype)
        tree = torch.from_numpy(np.array(tree_order)).to(self.device).long()
        for layer in tree_hierarchy:
            index_matrix = torch.zeros(len(layer), 2, dtype=torch.int64,
                                       device=self.device)
            index_matrix[:, 0] = torch.arange(len(layer))
            if len(self.J[layer][:, tree].nonzero()) != 0:
                index_matrix.index_copy_(0,
                                         self.J[layer][:, tree].nonzero()[:, 0],
                                         self.J[layer][:, tree].nonzero())
            index = index_matrix[:, 1]
            root = tree[index]

            hpj = self.J[layer, root] + h[:, layer]
            hmj = -self.J[layer, root] + h[:, layer]

            tree_energy += -torch.log(2 * (torch.cosh(self.beta * hpj) *
                                           torch.cosh(self.beta * hmj)).sqrt()).sum(dim=1) / self.beta
            for k in range(len(root)):
                h[:, root[k]] += torch.log(torch.cosh(self.beta * hpj) /
                                           torch.cosh(self.beta * hmj))[:, k] / (2 * self.beta)
            tree = tree[len(layer):]

        batch = sample.shape[0]
        assert sample.shape[1] == len(frozen_nodes)
        J = self.J[frozen_nodes][:, frozen_nodes].to_sparse()
        fvs_energy = -torch.bmm(sample.view(batch, 1, len(frozen_nodes)),
                                torch.sparse.mm(J, sample.t()).t().view(batch, len(frozen_nodes), 1)).reshape(batch) / 2
        fvs_energy -= sample @ self.h[frozen_nodes]

        energy = fvs_energy + tree_energy

        return self.beta * energy

    def correlation(self):
        edges = list(self.G.edges)
        FVS, tree1, tree_hierarchy = self.FVS_decomposition()
        sample = torch.from_numpy(exact_config(len(FVS))).to(self.dtype).to(self.device)
        calc = sample.shape[0]
        effective_energy = self.effective_energy(sample, FVS, tree1, tree_hierarchy)
        lnZ = torch.logsumexp(-effective_energy, dim=0)
        config_prob = torch.exp(-effective_energy - lnZ)
        logZ = -effective_energy
        correlation = torch.empty(self.G.number_of_edges(), device=self.device, dtype=self.dtype)
        connected_correlation = torch.empty(self.G.number_of_edges(), device=self.device, dtype=self.dtype)
        sample_add = torch.ones([calc, 1], device=self.device, dtype=self.dtype)

        for i in range(self.G.number_of_edges()):
            m, n = edges[i][0], edges[i][1]
            frozen_nodes = list(FVS)
            if m not in frozen_nodes:
                frozen_nodes.append(m)
            if n not in frozen_nodes:
                frozen_nodes.append(n)
            frozen_tree_hierarchy, frozen_tree = self.tree_hierarchize(frozen_nodes)

            if len(frozen_nodes) - len(FVS) == 2:
                sample_prime = torch.empty([4, calc, len(frozen_nodes)],
                                           device=self.device, dtype=self.dtype)
                sample_prime[0] = torch.cat((sample, sample_add, sample_add), dim=1)
                sample_prime[1] = torch.cat((sample, -sample_add, -sample_add), dim=1)
                sample_prime[2] = torch.cat((sample, -sample_add, sample_add), dim=1)
                sample_prime[3] = torch.cat((sample, sample_add, -sample_add), dim=1)

                fe_tree_prime = torch.zeros([4, calc], device=self.device, dtype=self.dtype)
                for k in range(4):
                    fe_tree_prime[k] = self.effective_energy(sample_prime[k],
                                                             frozen_nodes,
                                                             frozen_tree,
                                                             frozen_tree_hierarchy)

                p11 = torch.exp(-fe_tree_prime[0] - lnZ).sum()
                p00 = torch.exp(-fe_tree_prime[1] - lnZ).sum()
                p01 = torch.exp(-fe_tree_prime[2] - lnZ).sum()
                p10 = torch.exp(-fe_tree_prime[3] - lnZ).sum()

                correlation[i] = p11 + p00 - p01 - p10
                connected_correlation[i] = p11 + p00 - p01 - p10 - (p11 + p10 - p01 - p00) * (p11 + p01 - p10 - p00)

            elif len(frozen_nodes) - len(FVS) == 1:
                sample_prime = torch.empty([2, calc, len(frozen_nodes)],
                                           device=self.device, dtype=self.dtype)
                if m in FVS:
                    FVS_node = m
                else:
                    FVS_node = n
                FVS_index = FVS.index(FVS_node)
                sample_prime[0] = torch.cat((sample, sample_add), dim=1)
                sample_prime[1] = torch.cat((sample, -sample_add), dim=1)

                fe_tree_prime = torch.zeros([2, calc], device=self.device, dtype=self.dtype)
                for k in range(2):
                    fe_tree_prime[k] = self.effective_energy(sample_prime[k],
                                                             frozen_nodes,
                                                             frozen_tree,
                                                             frozen_tree_hierarchy)
                p1 = torch.exp(-fe_tree_prime[0] - lnZ)
                p0 = torch.exp(-fe_tree_prime[1] - lnZ)
                correlation[i] = (sample[:, FVS_index] * (p1 - p0)).sum()
                connected_correlation[i] = (sample[:, FVS_index] * (p1 - p0)).sum() - \
                                           (p1 - p0).sum() * config_prob @ sample[:, FVS_index]

            else:
                correlation[i] = (sample[:, FVS.index(m)] * sample[:, FVS.index(n)] * config_prob).sum()
                connected_correlation[i] = (sample[:, FVS.index(m)] * sample[:, FVS.index(n)] * config_prob).sum() - \
                                           config_prob @ sample[:, FVS.index(m)] * config_prob @ sample[:, FVS.index(n)]

        return correlation, edges, connected_correlation

    def magnetization(self):
        FVS, tree1, tree_hierarchy = self.FVS_decomposition()
        sample = torch.from_numpy(exact_config(len(FVS))).to(self.dtype).to(self.device)
        sample_size = sample.shape[0]
        sample_add = torch.ones([sample_size, 1],
                                dtype=self.dtype).to(self.device)
        FVS_energy = self.effective_energy(sample,
                                           FVS,
                                           tree1,
                                           tree_hierarchy)
        config_prob = torch.exp(-FVS_energy - torch.logsumexp(-FVS_energy, dim=0))

        sample_compeltion = torch.empty([sample_size, self.D],
                                        dtype=self.dtype).to(self.device)
        for i in range(self.D):
            if i not in FVS:
                frozen_nodes = list(FVS)
                frozen_nodes.append(i)
                tree_hierarchy, tree_order = self.tree_hierarchize(frozen_nodes)
                sample_positive = torch.cat((sample, sample_add), dim=1)
                sample_negative = torch.cat((sample, -sample_add), dim=1)

                energy_positive = self.effective_energy(sample_positive,
                                                        frozen_nodes,
                                                        tree_order,
                                                        tree_hierarchy)
                energy_negative = self.effective_energy(sample_negative,
                                                        frozen_nodes,
                                                        tree_order,
                                                        tree_hierarchy)
                p_positive = torch.exp(FVS_energy - energy_positive)
                p_negative = torch.exp(FVS_energy - energy_negative)
                sample_compeltion[:, i] = p_positive - p_negative
            else:
                sample_compeltion[:, i] = sample[:, FVS.index(i)]

        magnetization = config_prob @ sample_compeltion

        return magnetization

    def energy(self, sample, J, h):
        batch = sample.shape[0]
        D = sample.shape[1]
        J = J.to_sparse()
        energy = - torch.bmm(sample.view(batch, 1, D),
                             torch.sparse.mm(J, sample.t()).t().view(batch, D, 1)).reshape(batch) / 2 - sample @ h

        return energy
    
    def internal_energy(self):
        config = torch.from_numpy(exact_config(self.D)).to(self.dtype).to(self.device)
        energy = self.energy(config, self.J, self.h)
        print(energy)
        return ((energy * torch.exp(-self.beta * energy)).sum() / np.exp(self.lnZ())).item()

    def lnZ(self):
        config = torch.from_numpy(exact_config(self.D)).to(self.dtype).to(self.device)
        energy = self.energy(config, self.J, self.h)
        lnZ = torch.logsumexp(-self.beta * energy, dim=0)
        return lnZ.item()

    def lnZ_fvs(self):
        FVS, tree1, tree_hierarchy = self.FVS_decomposition()

        sample = torch.from_numpy(exact_config(len(FVS))).to(self.dtype).to(self.device)
        effective_energy = self.effective_energy(sample, FVS, tree1, tree_hierarchy)
        lnZ = torch.logsumexp(-effective_energy, dim=0)

        return lnZ.item()


def test_tree():
    from BeliefPropagation import belief_propagation, LoopyBeliefPropagation
    number_vertices, beta = 10, 1.0
    torch.manual_seed(0)
    g = nx.random_tree(number_vertices, seed=0)
    g.add_edge(3, 6)
    # g = nx.random_regular_graph(3, number_vertices, seed=0)
    print(g.edges)
    adj_matrix = torch.from_numpy(nx.adjacency_matrix(g, nodelist=list(g.nodes)).todense()).to(torch.float64)
    weights = torch.ones_like(adj_matrix)# torch.rand_like(adj_matrix)
    weights = (weights + weights.T) / 2
    coupling_matrix = weights * adj_matrix
    fields = torch.randn(number_vertices, dtype=torch.float64)

    exact_solver = Exact(g.copy(), coupling_matrix, fields, beta, 'cpu', 0)
    mag_exact = exact_solver.magnetization().cpu()
    f_exact = -exact_solver.lnZ() / beta
    e_exact = exact_solver.internal_energy()
    entropy_exact = beta * (e_exact - f_exact)
    cor_exact = exact_solver.correlation()
    print('exact magnetzation:', mag_exact)
    print((mag_exact+1)/2)
    print('exact correlation:', cor_exact)
    print('exact free energy:', f_exact)
    print('exact internal energy:', e_exact)
    print('exact entropy:', entropy_exact)

    # BP result
    f_BP, e_BP, entropy_BP, mag_BP, cor_BP, _ = belief_propagation(g.number_of_nodes(), beta, list(g.edges), [list(g.neighbors(i)) for i in g.nodes], coupling_matrix, fields)
    print('BP magnetzation:', mag_BP)
    print('BP free energy:', f_BP)
    print('BP internal energy:', e_BP)
    print('BP entropy:', entropy_BP)
    print('BP correlation:', cor_BP)

    lbp = LoopyBeliefPropagation(g, 2, coupling_matrix.numpy(), fields.numpy(), beta, np.float64)
    # for i in lbp.node_set:
    #     print(i, lbp.neighbor_r_nodes[i].node_set, lbp.neighbor_r_nodes[i].edges)
    #     for j in lbp.neighbor_r_nodes[i].node_set:
    #         print((i,j), lbp.neighbor_difference[i][j].exclude_node, lbp.neighbor_difference[i][j].node_set, lbp.neighbor_difference[i][j].edges)
    #         print((i,j), lbp.neighbor_intersection[i][j].exclude_node, lbp.neighbor_intersection[i][j].node_set, lbp.neighbor_intersection[i][j].edges)
    
    lbp.iteration()
    marginal = lbp.marginal()
    magnetzation = marginal - (1-marginal)
    e_lbp = lbp.internal_energy()
    # fe_lbps = [lbp.free_energy(node) for node in [0]]
    # fe_lbps = lbp.free_energy_path(0)
    # print(fe_lbps)
    fe_lbp = lbp.free_energy_cover()
    print('loopy BP marginal:', marginal)
    print('loopy BP magnetzation:', magnetzation)
    print('loopy BP internal energy:', e_lbp)
    print('loopy BP free energy:', fe_lbp)


if __name__ == '__main__':
    test_tree()