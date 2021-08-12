import numpy as np
import networkx as nx

# put it back into a 2D symmetric array
def to_2d(vector, size):
    x = np.zeros((size,size))
    c = 0
    for i in range(1,size):
        for j in range(0,i):
            x[i][j] = vector[c]
            x[j][i] = vector[c]
            c = c + 1
    return x

def topological_measures(data, size):
    CC = np.empty((0, size), int)
    EC = np.empty((0, size), int)
    PC = np.empty((0, size), int)

    topology = []
    max_solver_iterations = 1000000000
    for i in range(data.shape[0]):
        A = to_2d(data[i], size)
        np.fill_diagonal(A, 0)

        # create a graph from similarity matrix
        G = nx.from_numpy_matrix(A)
        U = G.to_undirected()

        # # # compute closeness centrality and transform the output to vector
        cc = nx.closeness_centrality(U)
        closeness_centrality = np.array([cc[g] for g in U])

        # # compute egin centrality and transform the output to vector
        ec = nx.eigenvector_centrality(U, weight="weight", max_iter=max_solver_iterations)
        eigenvector_centrality = np.array([ec[g] for g in U])

        # compute pagerank
        pr = nx.pagerank(U, alpha=0.85, weight="weight", max_iter=max_solver_iterations)
        pagerank = np.array([pr[g] for g in U])

        CC = np.vstack((CC, closeness_centrality))
        EC = np.vstack((EC, eigenvector_centrality))
        PC = np.vstack((PC, pagerank))

    topology.append(CC)
    topology.append(EC)
    topology.append(PC)

    
    return topology