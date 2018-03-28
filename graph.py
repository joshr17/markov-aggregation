#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:06:04 2018

@author: joshua
"""
import random

import numpy as np
import scipy as sp
import networkx as nx


def matrices_to_graph(P, T):
    """ Returns a networkx DiGraph. Converts a Markov process stored as (P,T)
    into a graph format.
    """
    N = P.shape[0]

    P_nonzero_coords = P.nonzero()
    P_nonzero_vals = P.todense()[P.nonzero()].tolist()[0]
    list_of_dicts_transition_probability = [{'transition probability': prob}
                                            for prob in P_nonzero_vals]
    list_of_dicts_mean_transition_time = [{'mean transition time': T.todense()[node, 0]}
                                          for node in range(N)]

    edges_with_weights = list(zip(P_nonzero_coords[0],
                                  P_nonzero_coords[1],
                                  list_of_dicts_transition_probability))
    nodes_with_times = list(zip(list(range(N)),
                                list_of_dicts_mean_transition_time))

    graph = nx.DiGraph()
    graph.add_edges_from(edges_with_weights)
    graph.add_nodes_from(nodes_with_times)

    return graph


def graph_to_matrices(graph):
    """ Returns a tuple (P,T) of sparse matrices. Converts a Markov process
    stored as a networkx graph into a tuple (P,T).
    """
    N = len(graph.nodes())

    P = sp.sparse.lil_matrix((N, N))
    T = sp.sparse.lil_matrix((N, 1))

    for edge in graph.edges():
        P[edge] = graph.edges[edge]['transition probability']

    for node in graph.nodes():
        T[node] = graph.node[node]['mean transition time']

    return(P, T)


def relabel_nodes_of_graph(G, ith_group):
    """ Returns a networkx graph which is the same as G but with the nodes
    relabelled so to make is possible to distinguish between which group the
    node originates from. The node numer is prefixed with the group number.
    """
    mapping = {}
    for node in G.nodes():
        mapping[node] = str(ith_group) + '-' + str(node)

    relabelled_G = nx.relabel_nodes(G, mapping, copy=True)

    return relabelled_G


def union_two_graphs(G, H):
    union = nx.union(G, H)
    return union


def decision(probability):
    """ Returns True with chance probability, False otherwise.
    """
    return random.random() < probability


def build_hierarcical_graph(n_clusters, n_per_cluster, p, n_interconnections):
    """ Constructs a handcrafted hierarchical graph. There are n_clusters
    number of constituent communities, each of which has many connections
    within it and few connections to the other communities.

    Args:
        n_clusters(integer):  number of constituent communities
        n_per_cluster (integer): number of nodes per cluster
        p (float): probability that any two nodes in a community have a
        connecting edge.
        n_interconnections (integer): number of edges from a community to other
        communities.

    Returns:
        networkx.classes.digraph.DiGraph: A hierarchical graph.
    """
    graphs = []
    hierarchical_graph = nx.DiGraph()

    for i in list(range(n_clusters)):
        graph = nx.erdos_renyi_graph(n_per_cluster, p, seed=None, directed=True)
        graph = relabel_nodes_of_graph(graph, i)
        graphs.append(graph)

        hierarchical_graph = union_two_graphs(hierarchical_graph, graph)

    for i in list(range(n_clusters)):
        counter = 0
        while counter < n_interconnections + 1:
            j = random.choice(range(n_clusters))

            random_node_i = random.choice(list(graphs[i].nodes()))
            random_node_j = random.choice(list(graphs[j].nodes()))

            if random.uniform(0, 1) < 0.5:
                hierarchical_graph.add_edge(random_node_i, random_node_j)
            else:
                hierarchical_graph.add_edge(random_node_j, random_node_i)

            counter = counter + 1
    hierarchical_graph = nx.convert_node_labels_to_integers(hierarchical_graph)

    return hierarchical_graph


def add_random_weights(hierarchical_graph):
    """ Adds random weights to a graph and normalises to ensure stochasticity.

    Args:
        hierarchical_graph (networkx.classes.digraph.DiGraph): hierarchical
        graph without weights.

    Returns:
        networkx.classes.digraph.DiGraph: hierarchical graph with random weights.
    """
    #give each edge a random weight
    edge_weight_dict = {}
    for edge in hierarchical_graph.edges():
        rand = random.uniform(0, 1)
        edge_weight_dict[edge] = rand

    nx.set_edge_attributes(hierarchical_graph, edge_weight_dict, 'transition probability')

    #give each node a random weight
    node_weight_dict = {}
    for node in hierarchical_graph.nodes():
        rand = random.uniform(0, 1)
        node_weight_dict[node] = rand

    nx.set_node_attributes(hierarchical_graph, node_weight_dict, 'mean transition time')

    #normalise the sum of the weights of the out edges for each node.
    for node in hierarchical_graph.nodes():
        out_edges = hierarchical_graph.out_edges(node)
        sum_out = sum([hierarchical_graph.edges[edge]['transition probability']
                       for edge in out_edges])
        for edge in out_edges:
            hierarchical_graph.edges[edge]['transition probability'] = \
                hierarchical_graph.edges[edge]['transition probability']/sum_out


def build_linkage_for_hierarchical_graph(n_clusters, n_per_cluster):
    """ Returns the linkage matrix that encapsulates the natural hierarchy of
    the hierarchical graph.

    Args:
        n_clusters(integer):  number of constituent communities
        n_per_cluster (integer): number of nodes per cluster

    Returns:
        numpy.ndarray: lnkage matrix for clustering by the handcrafted communities.
    """
    N = n_clusters * n_per_cluster
    Z = np.empty([N-1, 2])

    first_elements = []
    for i in range(n_clusters):
        first_elements.append(i*n_per_cluster)

    counter = 0
    for i in range(N-1):
        if i+1 not in first_elements:
            if i in first_elements:
                Z[i-counter, 0] = i
                Z[i-counter, 1] = i+1
                counter = counter + 1
            else:
                Z[i-counter+1, 0] = i+1
                Z[i-counter+1, 1] = N+i-counter

    super_nodes = []
    for j in range(n_clusters):
        super_nodes.append(N + n_per_cluster - 2 + j*(n_per_cluster-1))


    Z[n_clusters*(n_per_cluster -1), 0] = super_nodes[0]
    Z[n_clusters*(n_per_cluster -1), 1] = super_nodes[1]

    largest = super_nodes[n_clusters-1]
    for i in range(2, n_clusters):
        Z[n_clusters*(n_per_cluster -1) - 1 + i, 0] = super_nodes[i]
        Z[n_clusters*(n_per_cluster -1) - 1 + i, 1] = largest + i -1

    return Z
