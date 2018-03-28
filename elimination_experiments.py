#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:07:57 2018

@author: joshua
"""
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import community # python-louvain

import elimination as elim
import aggregation as agg
import graph as gp


def experiment_changing_density():
    """ Comparing calc_stationary_dist with general_elimination_pi for various
    densities.
    """
    elim_times = []
    linalg_times = []

    for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sims1 = []
        sims2 = []

        for _ in range(0, 20):

            P = elim.rand_stoch_matrix(200, i)
            T = elim.rand_trans_times(200)
            order = [100]*2

            start_time = time.time()
            elim.calc_stationary_dist(P, T)
            sims1.append(time.time() - start_time)
            print("--- %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            elim.general_elimination_pi(P, T, order)
            sims2.append(time.time() - start_time)
            print("--- %s seconds ---" % (time.time() - start_time))

        linalg_times.append(np.mean(sims1))
        elim_times.append(np.mean(sims2))

    #fig = plt.figure()
    plt.plot([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], linalg_times, 'r')
    plt.plot([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], elim_times, 'b')
    plt.xlabel('density')
    plt.ylabel('computation time')
    #fig.savefig('experiment_changing_density.jpg')
    plt.show()

    return(elim_times, linalg_times)


def experiment_elimination_more_at_a_time():
    """ Plots three graphs showing the computation time for performing
    elimination on random graphs of size 100, 1000, and 3000, elimination i
    number of nodes at a time. This is figure 3 in dissertation.
    """
    n100_times = []
    n1000_times = []
    n3000_times = []

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        order = [i]*int(100/float(i))
        sims = []

        for _ in range(0, 10):

            P = elim.rand_stoch_matrix(100, 0.1)
            T = elim.rand_trans_times(100)

            start_time = time.time()
            elim.general_elimination_pi(P, T, order)
            sims.append(time.time() - start_time)
            print("--- %s seconds ---" % (time.time() - start_time))

        n100_times.append(np.mean(sims))

    for i in [1, 5, 10, 20, 30, 50, 70, 100, 200, 300, 400, 500]:
        order = [i]*int(1000/float(i))
        sims = []

        for _ in range(0, 2):

            P = elim.rand_stoch_matrix(1000, 0.01)
            T = elim.rand_trans_times(1000)

            start_time = time.time()
            elim.general_elimination_pi(P, T, order)
            sims.append(time.time() - start_time)
            print("--- %s seconds ---" % (time.time() - start_time))

        n1000_times.append(np.mean(sims))

    for i in [50, 150, 300, 600, 1000, 1500]:
        order = [i]*int(3000/float(i))

        sims = []

        for _ in range(0, 2):
            P = elim.rand_stoch_matrix(3000, 0.01)
            T = elim.rand_trans_times(3000)

            start_time = time.time()
            elim.general_elimination_pi(P, T, order)
            sims.append(time.time() - start_time)
            print("--- %s seconds ---" % (time.time() - start_time))

        n3000_times.append(np.mean(sims))

    #fig = plt.figure()
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50], n100_times, 'r')
    plt.xlabel('number eliminated per iteration')
    plt.ylabel('computation time')
    #fig.savefig('experiment_elimination_more_at_a_time_100.jpg')
    plt.show()

    #fig = plt.figure()
    plt.plot([1, 5, 10, 20, 30, 50, 70, 100, 200, 300, 400, 500], n1000_times, 'b')
    plt.xlabel('number eliminated per iteration')
    plt.ylabel('computation time')
    #fig.savefig('experiment_elimination_more_at_a_time_1000.jpg')
    plt.show()

    #fig = plt.figure()
    plt.plot([50, 150, 300, 600, 1000, 1500], n3000_times, 'g')
    plt.xlabel('number eliminated per iteration')
    plt.ylabel('computation time')
    #fig.savefig('experiment_elimination_more_at_a_time_3000.jpg')
    plt.show()


def heuristic_1_Perm(P, T):
    """ Returns a permutation that once applied to (P,T) means that elimination
    is performed eliminating first nodes who having highest transition rate
    out.
    """
    N = P.shape[0]

    T_diag = np.identity(N)

    for i in range(0, N):
        T_diag[i, i] = T[i, 0]

    T_diag = sp.sparse.csr_matrix(T_diag)
    rates = T_diag*P
    sum_rates = sum(rates.transpose()).todense()
    permutation = np.argsort(-sum_rates)

    return permutation.tolist()[0]


def heuristic_1(P, T):
    """ Returns the stationary distribution calculated by eliminating according
    to heuristic 1.
    """
    permutation = heuristic_1_Perm(P, T)

    (perm_P, perm_T) = agg.permute_P_and_T(P, T, permutation)


    order_to_eliminate = [2]*100

    start_time = time.time()
    stationary = elim.general_elimination_pi(perm_P, perm_T, order_to_eliminate)
    print("--- %s seconds ---" % (time.time() - start_time))

    #note, this returns the stationary distribution permuted by 'permutation'
    return stationary


def experiment_using_heuristic_1():
    """ Returns plots comparing the computation speed elimination according to
    heuristic 1 versus eliminating randomly. Markov processes are
    non-hierarchical Erdos-Renyi type.
    """
    random_times = []
    sorted_order_times = []

    for N in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600]:
        sims1 = []
        sims2 = []

        order = [int(N/10)]*10


        for _ in range(0, 12):
            while True:
                try:
                    P = elim.rand_stoch_matrix(N, 0.1)
                    T = elim.rand_trans_times(N)

                    start_time = time.time()
                    elim.general_elimination_pi(P, T, order)
                    sims1.append(time.time() - start_time)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    permutation = heuristic_1_Perm(P, T)
                    (perm_P, perm_T) = agg.permute_P_and_T(P, T, permutation)

                    start_time = time.time()
                    elim.general_elimination_pi(perm_P, perm_T, order)
                    sims2.append(time.time() - start_time)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    break

                except IndexError:
                    pass # Try again

        random_times.append(np.mean(sims1))
        sorted_order_times.append(np.mean(sims2))

    #fig = plt.figure()
    plt.plot([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600], random_times, 'r')
    plt.plot([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600], sorted_order_times, 'b')
    plt.xlabel('number of nodes')
    plt.ylabel('computation time')
    #fig.savefig('experiment_using_heuristic_1.jpg')
    plt.show()


def heuristic_2_Perm(P, T):
    """ Returns a permutation that once applied to (P,T) means that elimination
    is performed eliminating first nodes who having highest transition rate
    into them.
    """
    N = P.shape[0]

    T_diag = np.identity(N)

    for i in range(0, N):
        T_diag[i, i] = T[i, 0]

    T_diag = sp.sparse.csr_matrix(T_diag)
    rates = T_diag*P

    sum_rates = sum(rates).todense()
    permutation = np.argsort(-sum_rates)
    return permutation.tolist()[0]


def heuristic_2(P, T):
    """ Returns the stationary distribution calculated by eliminating according
    to heuristic 2.
    """
    permutation = heuristic_2_Perm(P, T)

    (perm_P, perm_T) = agg.permute_P_and_T(P, T, permutation)

    order_to_eliminate = [2]*100

    start_time = time.time()
    stationary = elim.general_elimination_pi(perm_P, perm_T, order_to_eliminate)
    print("--- %s seconds ---" % (time.time() - start_time))

    #note, this returns the stationary distribution permuted by 'permutation'
    return stationary


def experiment_using_heuristic_2():
    """ Returns plots comparing the computation speed elimination according to
    heuristic 2 versus eliminating randomly. Markov processes are
    non-hierarchical Erdos-Renyi type.
    """
    random_times = []
    sorted_order_times = []

    random_times = []
    sorted_order_times = []

    for N in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600]:
        sims1 = []
        sims2 = []

        order = [int(N/10)]*10

        for _ in range(0, 12):
            while True:
                try:
                    P = elim.rand_stoch_matrix(N, 0.1)
                    T = elim.rand_trans_times(N)

                    ####
                    start_time = time.time()
                    elim.general_elimination_pi(P, T, order)
                    sims1.append(time.time() - start_time)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    ####
                    permutation = heuristic_2_Perm(P, T)
                    (perm_P, perm_T) = agg.permute_P_and_T(P, T, permutation)

                    start_time = time.time()
                    elim.general_elimination_pi(perm_P, perm_T, order)
                    sims2.append(time.time() - start_time)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    break

                except IndexError:
                    pass # Try again

        random_times.append(np.mean(sims1))
        sorted_order_times.append(np.mean(sims2))

    #fig = plt.figure()
    plt.plot([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600], random_times, 'r')
    plt.plot([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600], sorted_order_times, 'b')
    plt.xlabel('number of nodes')
    plt.ylabel('computation time')
    #fig.savefig('experiment_using_heuristic_2.jpg')
    plt.show()


def get_permutation_and_order_from_partition(partition):
    """ Returns the permutation required to eliminate in the order specified by
    partition, which is the output of the community.best_partition function.
    """
    number_of_communities = len(set(partition.values()))

    permutation = []
    order = []
    for i in range(number_of_communities):
        ith_community_nodes = [node for node in partition if partition[node] == i]
        permutation = permutation + ith_community_nodes
        order.append(len(ith_community_nodes))

    return(permutation, order)


def experiment_using_louvain():
    """ Plots graphs comparing the runtime of elimination when eliminating
    according to a heirarchy found using Louvain compared with random
    elimination.
    """
    random_times = []
    sorted_order_times = []

    for N in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600]:
        sims1 = []
        sims2 = []

        for _ in range(0, 12):
            while True:
                try:
                    P = elim.rand_stoch_matrix(N, 0.1)
                    T = elim.rand_trans_times(N)

                    N = P.shape[0]
                    T_diag = np.identity(N)

                    for i in range(0, N):
                        T_diag[i, i] = T[i, 0]

                    T_diag = sp.sparse.csr_matrix(T_diag)
                    rates = T_diag*P
                    A = rates + rates.transpose()
                    graph = gp.matrices_to_graph(A, T)
                    graph = graph.to_undirected()
                    partition = community.best_partition(graph, weight='transition probability')

                    (permutation, order) = get_permutation_and_order_from_partition(partition)
                    (perm_P, perm_T) = agg.permute_P_and_T(P, T, permutation)

                    start_time = time.time()
                    elim.general_elimination_pi(perm_P, perm_T, order)
                    sims2.append(time.time() - start_time)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    start_time = time.time()
                    elim.general_elimination_pi(P, T, order)
                    sims1.append(time.time() - start_time)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    break

                except IndexError:
                    pass # Try again

        random_times.append(np.mean(sims1))
        sorted_order_times.append(np.mean(sims2))

    #fig = plt.figure()
    plt.plot([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600], random_times, 'r')
    plt.plot([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
              390, 420, 450, 480, 510, 540, 570, 600], sorted_order_times, 'b')
    plt.xlabel('number of nodes')
    plt.ylabel('computation time')
    #fig.savefig('experiment_using_louvain.jpg')
    plt.show()


def experiment_heuristics():
    """ Simply checks performance for the heuristics.
    """
    P = elim.rand_stoch_matrix(200, 0.1)
    T = elim.rand_trans_times(200)
    order = [2]*100

    start_time = time.time()
    elim.calc_stationary_dist(P, T)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    elim.general_elimination_pi(P, T, order)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    elim.elimination_pi(P, T)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    heuristic_1(P, T)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    heuristic_2(P, T)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    print("Running elimination experiments")
    experiment_changing_density()
    experiment_elimination_more_at_a_time()
    experiment_using_heuristic_1()
    experiment_using_heuristic_2()
    experiment_using_louvain()
    experiment_heuristics()
