#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:09:06 2018

@author: joshua

This file contains tests of the functions contained in graph.py
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import elimination as elim
import graph as gp


def test_matrices_to_graph():
    """
    doctest not possible on stochastic matrix
    """
    P = elim.rand_stoch_matrix(10, 0.5)
    T = elim.rand_trans_times(10)

    graph = gp.matrices_to_graph(P, T)
    edge_attributes = nx.get_edge_attributes(graph, 'transition probability')

    dictionary = {}
    for i in range(10):
        for j in range(10):
            if P[i, j] != 0:
                dictionary[(i, j)] = P[i, j]

    print(edge_attributes == dictionary)


def test_graph_to_matrices():
    """
    doctest not possible on stochastic matrix
    """
    P = elim.rand_stoch_matrix(10, 0.5)
    T = elim.rand_trans_times(10)

    graph = gp.matrices_to_graph(P, T)

    (newP, newT) = gp.graph_to_matrices(graph)

    tol = 10e-5

    print(np.any(abs(P.todense()-newP.todense()) >= tol))
    print(any(abs(T-newT) >= tol))


def test_relabel_nodes_of_graph():
    """
    doctest not possible on stochastic matrix
    """
    P = elim.rand_stoch_matrix(10, 0.5)
    T = elim.rand_trans_times(10)

    graph = gp.matrices_to_graph(P, T)

    print(graph.edges())

    relabelled_graph = gp.relabel_nodes_of_graph(graph, 6)

    print(relabelled_graph.edges())


def test_build_hierarchical_graph():
    """
    doctest not possible on stochastic matrix
    """
    n_clusters = 5
    n_per_cluster = 10
    p = 0.8
    n_interconnections = 3

    graph = gp.build_hierarcical_graph(n_clusters, n_per_cluster, p, n_interconnections)
    gp.add_random_weights(graph)

    edge_attributes = nx.get_edge_attributes(graph, 'transition probability')

    print(sum(edge_attributes.values()))
    

    pos = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()

if __name__ == "__main__":

    LOGGER = logging.getLogger('markagg')
    LOGGER.setLevel(logging.DEBUG)

    print("Running module tests")
    #test_matrices_to_graph()
    #test_graph_to_matrices()
    #test_relabel_nodes_of_graph()
    test_build_hierarchical_graph()
