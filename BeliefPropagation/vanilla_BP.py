import numpy as np


def belief_propagation(number_nodes, beta, edges, neighbors, coupling_matrix, fields, stepmax=1000, epsilon=1e-4, damping_factor=0.0):
    num_edges = len(edges)

    difference_max = 10
    h = np.random.randn(number_nodes, number_nodes) / number_nodes ** 2

    # belief propagation
    for step in range(stepmax):
        for i in range(number_nodes):
            for j in range(len(neighbors[i])):
                a = neighbors[i][j]
                neighbor_i_exclude_j = list(neighbors[i])
                neighbor_i_exclude_j.remove(a)
                temp = fields[i] + (
                    np.arctanh(
                        np.tanh(beta * coupling_matrix[i, neighbor_i_exclude_j]) * np.tanh(
                            beta * h[neighbor_i_exclude_j, i])
                    ) / beta
                ).sum()
                temp = damping_factor*h[i][a] + (1-damping_factor)*temp
                difference = abs(temp - h[i][a])
                h[i][a] = temp
                if i == 0 and j == 0:
                    difference_max = difference
                elif difference > difference_max:
                    difference_max = difference
        if difference_max <= epsilon:
            break

    # calculate free energy
    fe_node = np.zeros(number_nodes)
    for i in range(number_nodes):
        neighbor_i = list(neighbors[i])
        temp1 = np.exp(beta * fields[i]) * (
            np.cosh(beta * (coupling_matrix[i, neighbor_i] +
                    h[neighbor_i, i])) / np.cosh(beta * h[neighbor_i, i])
        ).prod()
        temp2 = np.exp(-beta * fields[i]) * (
            np.cosh(beta * (-coupling_matrix[i, neighbor_i] +
                    h[neighbor_i, i])) / np.cosh(beta * h[neighbor_i, i])
        ).prod()
        fe_node[i] = -np.log(temp1 + temp2) / beta
    fe_node_sum = np.sum(fe_node)

    fe_edge = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*coupling_matrix[i, j]) * np.cosh(beta*(h[i, j]+h[j, i])) + \
            np.exp(-beta*coupling_matrix[i, j]) * \
            np.cosh(beta*(h[i, j]-h[j, i]))
        temp2 = 2 * np.cosh(beta*h[i, j]) * np.cosh(beta*h[j, i])
        fe_edge[edge_count] = -np.log(temp1/temp2) / beta
        edge_count += 1
    fe_edge_sum = np.sum(fe_edge)

    fe_sum = fe_node_sum - fe_edge_sum

    # calculate energy
    energy_edge = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = -coupling_matrix[i, j] * np.exp(beta*coupling_matrix[i, j]) * np.cosh(beta*(h[i, j]+h[j, i])) + \
            coupling_matrix[i, j] * np.exp(-beta*coupling_matrix[i, j]) * \
            np.cosh(beta*(h[i, j]-h[j, i]))
        temp2 = np.exp(beta*coupling_matrix[i, j]) * np.cosh(beta*(h[i, j]+h[j, i])) + \
            np.exp(-beta*coupling_matrix[i, j]) * \
            np.cosh(beta*(h[i, j]-h[j, i]))
        energy_edge[edge_count] = temp1 / temp2
        edge_count += 1
    energy_edge_sum = np.sum(energy_edge)

    energy_node = np.zeros(number_nodes)
    for i in range(number_nodes):
        neighbor_i = list(neighbors[i])
        energy_node[i] = fields[i] * np.tanh(
            beta * fields[i] + np.arctanh(np.tanh(
                beta*coupling_matrix[i, neighbor_i]) * np.tanh(beta*h[neighbor_i, i])).sum()
        )
    energy_node_sum = np.sum(energy_node)

    energy_BP = energy_edge_sum - energy_node_sum

    # calculate entropy
    entropy_BP = beta*(energy_BP - fe_sum)

    # calcualte magnetzation
    mag_BP = np.zeros(number_nodes)
    for i in range(number_nodes):
        neighbor_i = list(neighbors[i])
        temp = beta * fields[i] + np.arctanh(
            np.tanh(beta*coupling_matrix[i, neighbor_i]
                    ) * np.tanh(beta*h[neighbor_i, i])
        ).sum()
        mag_BP[i] = np.tanh(temp)

    # calculate connected correlation
    correlation_BP = np.empty(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*coupling_matrix[i, j]) * \
            np.cosh(beta*(h[i, j]+h[j, i]))
        temp2 = np.exp(-beta*coupling_matrix[i, j]) * \
            np.cosh(beta*(h[i, j]-h[j, i]))
        correlation_BP[edge_count] = (temp1 - temp2) / (temp1 + temp2) - \
            mag_BP[i] * mag_BP[j]
        edge_count += 1

    return fe_sum, energy_BP, entropy_BP, mag_BP, correlation_BP, step
