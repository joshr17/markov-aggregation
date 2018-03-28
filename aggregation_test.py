#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:44:28 2018

@author: joshua

This file contains tests of the functions contained in aggregation.py
"""
import time
import logging

import numpy as np
import scipy as sp

import elimination as elim
import aggregation as agg


def test_to_dictionary():
    """
    doctest:

    >>> test_to_dictionary()
    {(0, 2): 0.8, (0, 3): 0.2, (1, 0): 0.4, (1, 2): 0.6, (2, 1): 0.3, (2, 3): 0.7, (3, 3): 1.0}
    """
    P = sp.sparse.csr_matrix([[0, 0, 0.8, 0.2], [0.4, 0, 0.6, 0], [0, 0.3, 0, 0.7], [0, 0, 0, 1]])

    val = agg.to_dictionary(P)
    print(val)


def test_to_line_graph():
    """
    doctest:

    >>> test_to_line_graph()
    [[0.95 0.05 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.9  0.09 0.01 0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.9  0.09 0.01 0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.05 0.9  0.05 0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.8  0.
      0.05 0.15]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.9  0.09 0.01 0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.05 0.9  0.05 0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.8  0.
      0.05 0.15]
     [0.95 0.05 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.05 0.9  0.05 0.   0.
      0.   0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.8  0.
      0.05 0.15]]
    [[1.]
     [1.]
     [0.]
     [0.]
     [0.]
     [1.]
     [1.]
     [1.]
     [0.]
     [1.]
     [1.]
     [1.]
     [1.]
     [0.]
     [1.]
     [1.]]
    """
    P = sp.sparse.csr_matrix([[.95, .05, 0., 0.],
                              [0., 0.9, 0.09, 0.01],
                              [0., 0.05, 0.9, 0.05],
                              [0.8, 0., 0.05, 0.15]])

    T = sp.sparse.csr_matrix([[1], [1], [1], [1]])

    (line_P, line_T) = agg.to_line_graph(P, T)

    denseline_P = line_P.todense()
    denseline_T = line_T.todense()
    print(denseline_P)
    print(denseline_T)


def test_perm_rows():
    """
    doctest:

    >>> test_perm_rows()
    [[1 2 4]
     [6 3 4]
     [8 5 2]]
    [[8 5 2]
     [1 2 4]
     [6 3 4]]
    """
    test = sp.sparse.csr_matrix([[1, 2, 4], [6, 3, 4], [8, 5, 2]])
    print(test.toarray())

    test = agg.perm_rows(test, [2, 0, 1])
    print(test.toarray())


def test_perm_matrix():
    """
    doctest:

    >>> test_perm_matrix()
    [[2 8 5]
     [4 1 2]
     [4 6 3]]
    """
    test = sp.sparse.csr_matrix([[1, 2, 4], [6, 3, 4], [8, 5, 2]])

    test = agg.perm_matrix(test, [2, 0, 1])
    print(test.toarray())


def test_permute_Markov_flow():
    """
    doctest:

    >>> test_permute_Markov_flow()
    [[0.4 0.3 1.4 1.9]]
    [[1 0 2 3]]
    """
    P = sp.sparse.csr_matrix([[0, 0, 0.8, 0.2], [0.4, 0, 0.6, 0], [0, 0.3, 0, 0.7], [0, 0, 0, 1]])

    arr = sum(P).toarray()
    print(arr)
    sortedarr = np.argsort(arr)
    print(sortedarr)


def test_permute_Markov_flow_stochastic():
    """
    doctest not possible on stochastic matrix
    """
    P = elim.rand_stoch_matrix(100, 0.05)
    T = elim.rand_trans_times(100)
    M = elim.augment_matrix(P, T)
    perm = agg.random_permutation(100)

    start_time = time.time()

    agg.permute_Markov_flow(M, perm)
    print("--- permute_Markov_flow took %s seconds ---" % (time.time() - start_time))


def test_linkage_dict1():
    """
    doctest:

    >>> test_linkage_dict1()
    {0: [0], 1: [1], 3: [3], 4: [4], 5: [5], 6: [6], 7: [0, 1], 2: [2], 10: [0, 1, 2], 8: [3, 4], 11: [0, 1, 2, 3, 4], 9: [5, 6], 12: [0, 1, 2, 3, 4, 5, 6]}
    """
    Z = np.array([[0, 1],
                  [3, 4],
                  [5, 6],
                  [7, 2],
                  [10, 8],
                  [11, 9]])

    dict_Z = agg.linkage_dict(Z)
    print(dict_Z)


def test_linkage_dict2():
    """
    doctest:

    >>> test_linkage_dict2()
    {0: [0], 1: [1], 3: [3], 4: [4], 5: [5], 6: [6], 11: [0, 1], 2: [2], 13: [5, 6], 7: [7], 14: [0, 1, 2], 12: [3, 4], 15: [5, 6, 7], 8: [8], 17: [5, 6, 7, 8], 9: [9], 16: [0, 1, 2, 3, 4], 18: [5, 6, 7, 8, 9], 19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10: [10], 20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    """
    Z = np.array([[0, 1],
                  [3, 4],
                  [5, 6],
                  [11, 2],
                  [13, 7],
                  [14, 12],
                  [15, 8],
                  [17, 9],
                  [16, 18],
                  [19, 10]])

    dict_Z = agg.linkage_dict(Z)
    print(dict_Z)


def test_edges():
    """
    doctest:

    >>> test_edges()
    [(0, 2), (1, 0), (1, 2), (2, 1)]
    [2, 4, 6, 9]
    """
    P = sp.sparse.csr_matrix([[0, 0, 0.8, 0.2],
                              [0.4, 0, 0.6, 0],
                              [0, 0.3, 0, 0.7],
                              [0, 0, 0, 1]])

    dict_P = agg.to_dictionary(P)
    A = [0, 1, 2]

    edges = agg.edges_in_A(dict_P, A)
    print(edges)

    newedges = agg.edges_to_line(edges, 4)
    print(newedges)


def test_aggregation_pi():
    """ nonrandom test of aggregation method.
    doctest:
    >>> test_aggregation_pi()
    --- stationary1 ---
    [[0.33043478+0.j]
     [0.0173913 +0.j]
     [0.        -0.j]
     [0.        -0.j]
     [0.        -0.j]
     [0.29347826+0.j]
     [0.02934783+0.j]
     [0.00326087+0.j]
     [0.        -0.j]
     [0.01521739+0.j]
     [0.27391304+0.j]
     [0.01521739+0.j]
     [0.0173913 +0.j]
     [0.        -0.j]
     [0.00108696+0.j]
     [0.00326087+0.j]]
    -------
    --- stationary2 ---
    [[0.33043478+0.j]
     [0.0173913 +0.j]
     [0.        +0.j]
     [0.        +0.j]
     [0.        +0.j]
     [0.29347826+0.j]
     [0.02934783+0.j]
     [0.00326087+0.j]
     [0.        +0.j]
     [0.01521739+0.j]
     [0.27391304+0.j]
     [0.01521739+0.j]
     [0.0173913 +0.j]
     [0.        +0.j]
     [0.00108696+0.j]
     [0.00326087+0.j]]
    -------
    --- stationary3 ---
    [[0.33043478]
     [0.0173913 ]
     [0.        ]
     [0.        ]
     [0.        ]
     [0.29347826]
     [0.02934783]
     [0.00326087]
     [0.        ]
     [0.01521739]
     [0.27391304]
     [0.01521739]
     [0.0173913 ]
     [0.        ]
     [0.00108696]
     [0.00326087]]
    -------
    True
    True
    """
    #This is the transition matrix
    P = sp.sparse.csr_matrix([[.95, .05, 0., 0.],
                              [0., 0.9, 0.09, 0.01],
                              [0., 0.05, 0.9, 0.05],
                              [0.8, 0., 0.05, 0.15]])

    #These are the mean waiting times
    T = sp.sparse.csr_matrix([[1], [1], [1], [1]])

    (line_P, line_T) = agg.to_line_graph(P, T)

    #this just says what order to aggregate in
    test_Z = np.array([[0, 1],
                       [4, 2],
                       [5, 3]])

    stationary1 = elim.calc_stationary_dist(line_P, line_T)
    stationary2 = agg.aggregation_pi(P, T, test_Z, True, 0, 0).todense()
    stationary3 = elim.elimination_pi(line_P, line_T)

    print("--- stationary1 ---")
    print(stationary1)
    print("-------")
    print("--- stationary2 ---")
    print(stationary2)
    print("-------")
    print("--- stationary3 ---")
    print(stationary3)
    print("-------")

    tol = 1e-10
    print(np.all(abs(stationary1 - stationary2) < tol))
    print(np.all(abs(stationary2 - stationary3) < tol))


def test_aggregation_pi_stochastic():
    """ Random test of aggregation_pi. Doctest not possible on stochastic
    matrix.
    """
    test_Z = np.array([[01., 08.],
                       [21., 22.],
                       [17., 18.],
                       [15., 30.],
                       [00., 33.],
                       [02., 09.],
                       [10., 13.],
                       [03., 19.],
                       [14., 32.],
                       [06., 34.],
                       [12., 37.],
                       [11., 40.],
                       [39., 41.],
                       [20., 29.],
                       [16., 35.],
                       [25., 26.],
                       [24., 43.],
                       [27., 45.],
                       [28., 31.],
                       [36., 44.],
                       [46., 48.],
                       [23., 47.],
                       [38., 42.],
                       [49., 52.],
                       [05., 53.],
                       [50., 51.],
                       [04., 07.],
                       [54., 56.],
                       [55., 57.]])

    start_time = time.time()
    test_Z = test_Z.astype(int)
    P = elim.rand_stoch_matrix(30, 0.7)
    T = elim.rand_trans_times(30)
    (line_P, line_T) = agg.to_line_graph(P, T)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    val1 = agg.aggregation_pi(P, T, test_Z, True, 0, 0)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    val2 = elim.calc_stationary_dist(line_P, line_T)
    print("--- %s seconds ---" % (time.time() - start_time))

    tol = 1e-10
    print(np.all(abs((val1-val2)) < tol))


if __name__ == "__main__":

    LOGGER = logging.getLogger('markagg')
    LOGGER.setLevel(logging.DEBUG)

    print("Running module tests")
    test_to_dictionary()
    test_to_line_graph()
    test_perm_rows()
    test_perm_matrix()
    test_permute_Markov_flow()
    test_permute_Markov_flow_stochastic()
    test_linkage_dict1()
    test_linkage_dict2()
    test_edges()
    test_aggregation_pi()
    test_aggregation_pi_stochastic()

    print("Running doctest")
    import doctest
    doctest.testmod()
