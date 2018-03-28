#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:45:55 2018

@author: joshua

This file contains tests of the functions contained in elimination.py
"""
import logging
import time

import numpy as np
import scipy as sp

import elimination as elim


def test_LUDecomp():
    """
    doctest:
    >>> test_LUDecomp() # doctest: +NORMALIZE_WHITESPACE
    LYX =
      (0, 1)        0.5000000000000002
      (1, 0)        15.999999999999986
      (1, 1)        7.999999999999996
    new_PYY =
      (0, 0)        0.9450000000000001
      (1, 0)        0.7699999999999996
      (0, 1)        0.05500000000000001
      (1, 1)        0.22999999999999998
    new_TY =
      (0, 0)        1.5000000000000002
      (1, 0)        24.999999999999982
    LU =
    [[ True  True  True  True  True]
     [ True  True  True  True  True]
     [ True  True  True  True  True]
     [ True  True  True  True  True]]
    """
    P = sp.sparse.csr_matrix([[.95, .05, 0., 0.],\
        [0., 0.9, 0.09, 0.01],\
        [0., 0.05, 0.9, 0.05],\
        [0.8, 0., 0.05, 0.15]])

    #These are the mean waiting times
    T = sp.sparse.csr_matrix([[1], [1], [1], [1]])

    M = elim.augment_matrix(P, T)
    n = 2

    #form object
    decomp = elim.LUdecomp(M, n)

    #calculating quantities, want to check these are right.
    LYX = decomp.LYX()
    new_PYY = decomp.new_PYY(LYX)
    new_TY = decomp.new_TY(LYX)
    L = decomp.L(LYX)
    U = decomp.U(new_PYY, new_TY)
    LU = L*U

    print("LYX = ")
    print(LYX)
    print("new_PYY = ")
    print(new_PYY)
    print("new_TY = ")
    print(new_TY)
    print("LU = ")

    tol = 10e-5
    print(abs((LU - M).toarray()) < tol)


def test_calc_TAB():
    """
    doctest:
    >>> test_calc_TAB() # doctest: +NORMALIZE_WHITESPACE
      (0, 0)    2.569060773480663
      (1, 0)    3.2044198895027627
      (2, 0)    1.9613259668508287
    """
    P = sp.sparse.csr_matrix([[0.0, 0.0, 0.8, 0.2], [0.4, 0.0, 0.6, 0.0],
                              [0.0, 0.3, 0.0, 0.7], [0.0, 0.0, 0.0, 1.0]])
    T = sp.sparse.csr_matrix([[1.0], [1.0], [1.0], [1.0]])

    val = elim.calc_TAB(P, T, 1)

    print(val)


def test_general_elimination_pi():
    """
    doctest:
    >>> test_general_elimination_pi()
    [[0.25]
     [0.5 ]
     [0.25]]
    [[0.25]
     [0.5 ]
     [0.25]]
    """

    P = sp.sparse.csr_matrix([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]])
    T = sp.sparse.csr_matrix([[1.0], [1.0], [1.0]])
    order = [1, 1, 1]

    stat_dist_elim2 = elim.general_elimination_pi(P, T, order)
    print(stat_dist_elim2)

    stat_dist = elim.calc_stationary_dist(P, T)
    print(stat_dist)


def test_general_elimination_pi_stochastic():
    """
    >>> test_general_elimination_pi_stochastic()
    True
    """
    P = elim.rand_stoch_matrix(900, 0.01)
    T = elim.rand_trans_times(900).tocsr()
    order = [450]*2

    val1 = elim.general_elimination_pi(P, T, order)

    val2 = elim.calc_stationary_dist(P, T)

    equal = np.allclose(val2, val1, rtol=1e-05, atol=1e-04)
    print(equal)


def test_calc_stationary_dist_stochastic():
    """
    doctest not possible on stochastic matrix
    """
    P = elim.rand_stoch_matrix(500, 0.01)
    T = elim.rand_trans_times(500)

    start_time = time.time()
    elim.calc_stationary_dist(P, T)
    print("--- %s seconds ---" % (time.time() - start_time))


def test_elimination_pi_stochastic():
    """Tests of elimination_pi gives the right answer (i.e. the same answer as
    calc_stationary_dist).
    doctest:
    >>> test_elimination_pi_stochastic()
    True
    """
    P = elim.rand_stoch_matrix(100, 0.1)
    T = elim.rand_trans_times(100)

    statDistElim = elim.elimination_pi(P, T)
    statDistManual = elim.calc_stationary_dist(P, T)

    tol = 1e-4
    print(np.all(statDistElim - statDistManual) < tol)


if __name__ == "__main__":

    LOGGER = logging.getLogger('markagg')
    LOGGER.setLevel(logging.DEBUG)

    print("Running module tests")
    test_LUDecomp()
    test_calc_TAB()
    test_general_elimination_pi()
    test_general_elimination_pi_stochastic()
    test_calc_stationary_dist_stochastic()
    test_elimination_pi_stochastic()

    print("Running doctest")
    import doctest
    doctest.testmod()
