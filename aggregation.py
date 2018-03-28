#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:44:28 2018

@author: joshua
"""

import scipy as sp
import numpy as np

import elimination as elim


def to_dictionary(P):
    """ Returns a dictionary whose keys are tuples (edges) (i,j) where i and j
    are from the integers 0,1,...,P.shape[0]-1. The value associated to the key
    (i,j) is the transition probabliity from i to j, i.e. P[i,j]. Only edges
    for which P[i,j] is non-zero are included.
    """
    arr_P = P.toarray()
    indexes = [(i, j)
               for i in range(0, arr_P.shape[0])
               for j in range(0, arr_P.shape[1])
               if not arr_P[i][j] == 0]

    dict_P = dict([index, arr_P[index]] for index in indexes)

    return dict_P


def to_line_graph(P, T):
    """ Returns the transition matrix and mean waiting time vector for the
    Markov process on the line graph constructed as described in the paper.
    """
    N = P.shape[1]

    line_P = sp.sparse.csc_matrix((N**2, N**2))
    line_T = sp.sparse.csc_matrix((N**2, 1))

    arr_P = P.toarray()
    indices_P = [(i, j)
                 for i in range(0, arr_P.shape[0])
                 for j in range(0, arr_P.shape[1])
                 if not arr_P[i][j] == 0]

    for edge1 in indices_P:
        line_T[edge1[0]*N + edge1[1]] = T[edge1[1], 0]
        for edge2 in indices_P:
            if edge2[1] == edge1[0]:
                line_P[edge2[0]*N + edge2[1], edge1[0]*N + edge1[1]] = P[edge1[0], edge1[1]]

    return [line_P, line_T]


def random_permutation(n):
    """ Returns a random 1D numpy.ndarray representing a permutation from the
    permutation group S_n. If perm is returned then perm[0] = 7 means send
    7-->0.
    """
    return np.random.permutation(np.arange(n))


def invert(perm):
    """ Returns the inverse permutation of perm.
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def perm_rows(H, permutation):
    """ Returns a matrix equal to matrix H but with its row indices permuted
    according to the permutation permutation.
    """
    if sp.sparse.issparse(H):
        x = H.tocoo()
        permutation = np.argsort(permutation)
        permutationSorted = np.asarray(permutation, dtype=x.row.dtype)
        x.row = permutationSorted[x.row]

        H = x
        H = x.tocsr()

    else:
        permutation = np.argsort(permutation)
        permutationSorted = np.asarray(permutation, dtype=H.dtype)
        H.row = permutationSorted[H.row]

    return H


def perm_matrix(H, permutation):
    """ Returns a matrix equal to matrix H but with its row and column indices
    permuted according to the permutation permutation.
    """
    x = H.tocoo()
    permutation = np.argsort(permutation)
    permutation_sorted = np.asarray(permutation, dtype=x.row.dtype)
    x.row = permutation_sorted[x.row]
    x.col = permutation_sorted[x.col]
    H = x.tocsr()
    return H


def permute_P_and_T(P, T, permutation):
    """ Returns a matrix and vector equal to matrix P and vector T but with
    P's row and column indices permuted and H's row indices permuted according
    to the permutation permutation.
    """
    perm_P = perm_matrix(P, permutation)
    perm_T = perm_rows(T, permutation)
    return (perm_P, perm_T)


def permute_Markov_flow(M, permutation):
    """ Returns an augmented matrix representing the permuted Markov process
    permuted exactly as is done in function permute_P_and_T.
    """
    P = elim.get_P(M)
    T = elim.get_T(M)
    perm_P = perm_matrix(P, permutation)
    perm_T = perm_rows(T, permutation)

    perm_M = elim.augment_matrix(perm_P, perm_T)
    return perm_M


def linkage_dict(Z):
    """ Returns a dictionary whose keys are the entries of the matrix Z and
    whose values are the subset of nodes contained in the entry value. This is
    just a translation between datatypes. See the documentation for
    scipy.cluster.hierarchy.linkage for details about the linkage matrix
    datatype for storing the information for a hierarchical clustering.
    """
    N = Z.shape[0]
    dict_Z = {}
    for i in range(0, N):
        if Z[i, 0] <= N:
            dict_Z[Z[i, 0]] = [Z[i, 0]]
        else:
            dict_Z[Z[i, 0]] = dict_Z[Z[Z[i, 0]-N-1, 0]] + dict_Z[Z[Z[i, 0]-N-1, 1]]

        if Z[i, 1] <= N:
            dict_Z[Z[i, 1]] = [Z[i, 1]]
        else:
            dict_Z[Z[i, 1]] = dict_Z[Z[Z[i, 1]-N-1, 0]] + dict_Z[Z[Z[i, 1]-N-1, 1]]

    dict_Z[2*N] = list(range(0, N+1))
    return dict_Z


def edges_in_A(dictP, A):
    """ Returns the list of edges (i,j) for which both i and j are contained in
    the collection of nodes A.
    """
    edges = [edge for edge in dictP if edge[0] in A and edge[1] in A]
    return edges


def edges_between_AB(dictP, A, B):
    """ Returns the list of edges (i,j) for which exactly one of i and j are in
    A and the other in B.
    """
    edges = [edge for edge in dictP
             if (edge[0] in A and edge[1] in B)
             or (edge[0] in B and edge[1] in A)]
    return edges


def edges_to_line(edges, N):
    """ Returns a list of integers corresponding to a relabellign of the edge
    tuples by mapping (i,j) --> to N*i + j as per my convention for labelling
    of lineP, lineT.
    """
    def relabel(edge):
        return N*edge[0] +edge[1]

    newlist = list(map(relabel, edges))
    return newlist


def translate_dendrogram(dictP, lineP, lineT, Z, N, is_binary, n_clusters, n_per_cluster):
    """ Returns the permuted matrix and vector of the line graph transition
    probability matrix and mean waiting times vector. They are permuted into
    the order so that, upon eliminating according to order (also returned) this
    corresponds to aggregation by the hierarchical clustering Z. is_binary is a
    boolean, and if is_binary = True then this means the hierarhichal
    clustering is binary and the arguments n_clusters and n_per_cluster play
    absolutely no role. If is_binary = False then the hierarchical clustering
    is not binary. Currently this situation is only supported for Markov
    processes generated by graph.build_hierarcical_graph, and the arguments
    should simply be inherited from that function.

    Args:
        dictP (dictionary): Transition probabilites for Markov process on edges
        stored as a dictionary.
        lineP (scipy.sparse.csc.csc_matrix): Transition probabilites for Markov
        process on edges stored as a matrix.
        lineT (scipy.sparse.csc.csc_matrix): Mean waiting times for Markov
        process on edges stored as a matrix.
        Z (numpy.ndarray): linkage matrix.
        N (integer): Number of nodes.
        is_binary (boolean): Says whether cluster tree (dendrogram) is binary
        or not.
        n_clusters (integer): number of clusters in hand-crafted Markov process.
        n_per_cluster (integer): nubmer of nodes per clusters in hand-crafted
        Markov process.

    Returns:
        tuple: Contains the permuted Markov process on edges, the order to
        eliminate it in to aggregae Markov process on nodes according to Z,
        and the permutation used to reach this ordering.
    """
    permutation = []
    order = []

    linkages = linkage_dict(Z)

    counter = 0

    for i in range(0, N-1):

        leaves0 = linkages[Z[i, 0]]
        leaves1 = linkages[Z[i, 1]]

        edges1 = edges_between_AB(dictP, leaves0, leaves1)

        if (len(leaves0) == 1 and (leaves0[0], leaves0[0]) in dictP):
            edges1 = edges1 + [(leaves0[0], leaves0[0])]
        if (len(leaves1) == 1 and (leaves1[0], leaves1[0]) in dictP):
            edges1 = edges1 + [(leaves1[0], leaves1[0])]

        edges2 = edges_to_line(edges1, N)

        if len(edges2) > 0:
            counter = counter + len(edges2)
            permutation.extend(edges2)
            if is_binary:
                order.append(len(edges2))

    if is_binary is False:
        linkage_dict_Z = linkage_dict(Z)
        cluster_numbers = [N-1+(n_per_cluster-1)*i for i in range(1, n_clusters+1)]

        for cluster in cluster_numbers:
            nodes_in_cluster = linkage_dict_Z[cluster]
            edges_in_cluster = edges_in_A(dictP, nodes_in_cluster)
            number_edges_in_cluster = len(edges_in_cluster)
            order.append(number_edges_in_cluster)

        edges_between_1st_and_2nd_clusters = edges_between_AB(
            dictP, linkage_dict_Z[cluster_numbers[0]], linkage_dict_Z[cluster_numbers[1]])

        number_edges_between_1st_and_2nd_clusters = len(edges_between_1st_and_2nd_clusters)
        order.append(number_edges_between_1st_and_2nd_clusters)

        for i in range(2, n_clusters):
            merging_with = 2*(N-1) - n_clusters + i

            edges_between = edges_between_AB(
                dictP, linkage_dict_Z[cluster_numbers[i]], linkage_dict_Z[merging_with])

            number_edges_between = len(edges_between)
            order.append(number_edges_between)

    remainder = [x for x in range(N**2) if x not in set(permutation)]
    permutation = permutation + remainder

    (permP, permT) = permute_P_and_T(lineP, lineT, permutation)

    return(permP, permT, order, permutation)


def aggregation_pi(P, T, Z, is_binary, n_clusters=0, n_per_cluster=0):
    """ Returns the stationary distribution of the Markov process on edges
    formed from the markov process (P,T) on nodes.

    Args:
        P (scipy.sparse.csr.csr_matrix): Transition matrix.
        T (scipy.sparse.lil.lil_matrix): Mean waiting times.
        Z (numpy.ndarray): linkage matrix.
        is_binary (boolean): specifies whether or not clustering is binary.

    Returns:
        scipy.sparse.csr.csr_matrix: Stationary distribution for the Markov
        process on edges.
    """
    N = P.shape[0]
    (line_P, line_T) = to_line_graph(P, T)
    dict_P = to_dictionary(P)

    (perm_P, perm_T, order, permutation) = translate_dendrogram(
        dict_P, line_P, line_T, Z, N, is_binary, n_clusters, n_per_cluster)

    inv_permutation = invert(permutation)
    perm_stationary = elim.general_elimination_pi(perm_P, perm_T, order)
    perm_stationary = sp.sparse.csr_matrix(perm_stationary)

    stationary = perm_rows(perm_stationary, inv_permutation)

    return stationary
