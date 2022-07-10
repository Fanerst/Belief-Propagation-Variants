import numpy as np

def readgraph(D, graph_dir, couplings):
    with open(graph_dir + '{}nodes.txt'.format(D), 'r') as f:
        list1 = f.readlines()
    f.close()
    num_edges = int(list1[0].split()[1])
    edges = np.zeros([len(list1)-1, 2], dtype=int)
    for i in range(len(list1)-1):
        edges[i] = list1[i+1].split()
    neighbors = {}.fromkeys(np.arange(D))
    for key in neighbors.keys():
        neighbors[key] = []
    for edge in edges:
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    '''
    for key in neighbors.keys():
        neighbors[key] = np.array(neighbors[key])
    '''
    J = np.loadtxt(graph_dir + 'energy_function{}nodes{}.txt'.format(
        D, couplings), dtype=np.float32)

    return num_edges, edges, neighbors, J


def atanh(x):
    return 0.5*np.log((1+x)/(1-x))

def BPofh(D, beta, edges, neighbors, J, stepmax=1000, epsilon=1e-4, damping_factor=0.0):
    num_edges = len(edges)

    difference_max = 10
    h = {}.fromkeys(np.arange(D))
    for key in h.keys():
        h[key] = np.random.randn(len(neighbors[key]))

    # belief propagation
    for step in range(stepmax):
        for i in h.keys():
            for j in range(len(h[i])):
                a = neighbors[i][j]
                B = list(neighbors[i])
                B.remove(a)
                temp = 0
                for b in B:
                    temp += np.arctanh(np.tanh(beta * J[i, b]) *
                            np.tanh(beta *h[b][neighbors[b].index(i)])) / beta
                temp = damping_factor*h[i][j] + (1-damping_factor)*temp
                difference = abs(temp - h[i][j])
                h[i][j] = temp
                if i == 0 and j == 0:
                    difference_max = difference
                elif difference > difference_max:
                    difference_max = difference
        if difference_max <= epsilon:
            break

    # calculate free energy
    fe_node = np.zeros(D)
    for i in range(D):
        B = list(neighbors[i])
        temp1 = 1
        temp2 = 1
        for k in range(len(B)):
            b = B[k]
            temp1 *= np.cosh(beta*(J[i, b]+h[b][neighbors[b].index(i)])) / \
                    np.cosh(beta*h[b][neighbors[b].index(i)])
            temp2 *= np.cosh(beta*(-J[i, b]+h[b][neighbors[b].index(i)])) / \
                    np.cosh(beta*h[b][neighbors[b].index(i)])
        fe_node[i] = - np.log(temp1 + temp2) / beta
    fe_node_sum = np.sum(fe_node)

    fe_edge = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]+h[j][neighbors[j].index(i)]))+ \
                np.exp(-beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]-h[j][neighbors[j].index(i)]))
        temp2 = 2*np.cosh(beta*h[i][neighbors[i].index(j)])*np.cosh(beta*h[j][neighbors[j].index(i)])
        fe_edge[edge_count] = - np.log(temp1/temp2) / beta
        edge_count += 1
    fe_edge_sum = np.sum(fe_edge)

    fe_sum = fe_node_sum - fe_edge_sum

    # calculate energy
    energy_BP = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = -J[i,j]*np.exp(beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]+h[j][neighbors[j].index(i)]))+ \
                J[i,j]*np.exp(-beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]-h[j][neighbors[j].index(i)]))
        temp2 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]+h[j][neighbors[j].index(i)]))+ \
                np.exp(-beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]-h[j][neighbors[j].index(i)]))
        energy_BP[edge_count] = temp1 / temp2
        edge_count += 1
    energy_BP = np.sum(energy_BP)

    # calculate entropy
    entropy_BP = beta*(energy_BP - fe_sum)

    # calcualte magnetzation
    mag_BP = np.zeros(D)
    for i in range(D):
        temp = 0
        for b in list(neighbors[i]):
            temp += np.arctanh(np.tanh(beta*J[i, b]) * np.tanh(beta *h[b][neighbors[b].index(i)]))
        mag_BP[i] = np.tanh(temp)
    '''
    # calculate correlation
    h_de = {}.fromkeys(np.arange(D))
    for key in h.keys():
        h_de[key] = np.zeros([len(neighbors[key]), D])
        h_de[key][:, key] = 1

    for step in range(stepmax):
        for i in h_de.keys():
            for j in range(len(neighbors[i])):
                a = neighbors[i][j]
                B = list(neighbors[i])
                B.remove(a)
                temp = np.zeros(D)
                for b in B:
                    weight = np.tanh(beta*J[i,b])  #* \
                    #(1-np.tanh(beta*h[b][neighbors[b].index(i)])**2) / \
                    #(1-np.tanh(beta*J[i,b])**2*np.tanh(beta*h[b][neighbors[b].index(i)])**2)
                    temp += weight * h_de[b][neighbors[b].index(i)]
                temp[i] += 1

                difference = abs(temp - h_de[i][j]).mean()
                h_de[i][j] = temp
                if i == 0 and j == 0:
                    difference_max = difference
                elif difference > difference_max:
                    difference_max = difference
        if difference_max <= epsilon:
            break
    print(h_de)
    correlation_BP = np.zeros([D, D])
    for i in range(D):
        temp1 = 1 - mag_BP[i]**2
        B = list(neighbors[i])
        for j in range(D):
            temp2 = 0
            for b in B:
                weight = np.tanh(beta*J[i,b]) # * \
                        # (1-np.tanh(beta*h[b][neighbors[b].index(i)]))**2 / \
                        # (1-np.tanh(beta*J[i,b])**2*np.tanh(beta*h[b][neighbors[b].index(i)])**2)
                temp2 += weight * h_de[b][neighbors[b].index(i), j]
            if i == j:
                correlation_BP[i, j] = temp1 * (1 + temp2)
            else:
                correlation_BP[i, j] = temp1 * temp2
    '''
    # calculate connected correlation
    correlation_BP = np.empty(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = \
        np.exp(beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]+h[j][neighbors[j].index(i)]))
        temp2 = \
        np.exp(-beta*J[i,j])*np.cosh(beta*(h[i][neighbors[i].index(j)]-h[j][neighbors[j].index(i)]))
        correlation_BP[edge_count] = (temp1 - temp2) / (temp1 + temp2)
        edge_count += 1

    return fe_sum, energy_BP, entropy_BP, mag_BP, correlation_BP, step

def BP_revised(D, beta, edges, neighbors, J, stepmax=1000, epsilon=1e-4, damping_factor=0.0):
    num_edges = len(edges)

    difference_max = 10
    h = np.random.randn(D, D)
    # belief propagation
    for step in range(stepmax):
        for i in range(D):
            for j in range(len(neighbors[i])):
                a = neighbors[i][j]
                B = list(neighbors[i])
                B.remove(a)
                temp = (np.arctanh(
                        np.tanh(beta * J[i, B]) * np.tanh(beta * h[B, i])
                        ) / beta).sum()
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
    fe_node = np.zeros(D)
    for i in range(D):
        B = list(neighbors[i])
        temp1 = (np.cosh(beta * (J[i, B] + h[B, i])) /
                np.cosh(beta * h[B, i])).prod()
        temp2 = (np.cosh(beta * (-J[i, B] + h[B, i])) /
                np.cosh(beta * h[B, i])).prod()
        fe_node[i] = - np.log(temp1 + temp2) / beta
    fe_node_sum = np.sum(fe_node)

    fe_edge = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*J[i,j]) * np.cosh(beta*(h[i,j]+h[j,i])) + \
                np.exp(-beta*J[i,j]) * np.cosh(beta*(h[i,j]-h[j,i]))
        temp2 = 2*np.cosh(beta*h[i,j])*np.cosh(beta*h[j,i])
        fe_edge[edge_count] = - np.log(temp1/temp2) / beta
        edge_count += 1
    fe_edge_sum = np.sum(fe_edge)

    fe_sum = fe_node_sum - fe_edge_sum

    # calculate energy
    energy_BP = np.zeros(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = -J[i,j]*np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i])) + \
                J[i,j]*np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
        temp2 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i])) + \
                np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
        energy_BP[edge_count] = temp1 / temp2
        edge_count += 1
    energy_BP = np.sum(energy_BP)

    # calculate entropy
    entropy_BP = beta*(energy_BP - fe_sum)

    # calcualte magnetzation
    mag_BP = np.zeros(D)
    for i in range(D):
        B = list(neighbors[i])
        temp = np.arctanh(
                np.tanh(beta*J[i, B]) * np.tanh(beta*h[B,i])
                ).sum()
        mag_BP[i] = np.tanh(temp)

    # calculate connected correlation
    correlation_BP = np.empty(num_edges)
    edge_count = 0
    for edge in edges:
        i, j = edge
        temp1 = np.exp(beta*J[i,j])*np.cosh(beta*(h[i,j]+h[j,i]))
        temp2 = np.exp(-beta*J[i,j])*np.cosh(beta*(h[i,j]-h[j,i]))
        correlation_BP[edge_count] = (temp1 - temp2) / (temp1 + temp2) - \
        mag_BP[i] * mag_BP[j]
        edge_count += 1

    return fe_sum, energy_BP, entropy_BP, mag_BP, correlation_BP, step
