#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:36:17 2018

@author: joshua

This file contains the functions required to perform elimination and to use
it to calculate the stationary distribution of a Markov process or the mean
first passage times from every node in a set A to its compelement A^c.
"""
import logging

import numpy as np
import scipy as sp
from scipy.linalg import eig
from sklearn.preprocessing import normalize

LOGGER = logging.getLogger('markagg')


def rand_stoch_matrix(n, den):
    """ Returns a scipy sparse random stochastic matrix of shape nxn and
    density den.
    """
    while True:
        # Create uniformly distributed random matrix
        matrix = sp.sparse.rand(n, n, density=den, format='lil')
        if np.any(matrix.sum(axis=1) < 0.001):
            pass
            #try again - rows cannot be zero - loop unil it works
        else:
            matrix = normalize(matrix, norm='l1', axis=1)
            break

    return matrix


def rand_trans_times(n):
    """ Returns a scipy sparse random matrix of shape nx1 and density den.
    """
    transTimes = sp.sparse.rand(n, 1, density=1, format='lil')
    return transTimes


def augment_matrix(P, T):
    """ Returns the augmented matrix [I-P, T]. This object encapsulates all the
    required information about a semi-Markov process.
    """
    n = P.shape[0]

    #identity matrix as a sparse matrix
    identity = sp.sparse.eye(n, n, format='lil')

    #this matrix is the matrix [I-P T]
    augmented_matrix = sp.sparse.lil_matrix((n, n+1))
    augmented_matrix[:, 0:n] = identity - P
    augmented_matrix[:, n:n+1] = T

    return augmented_matrix.tocsc()


def get_P(M):
    """ Returns P when given [I-P, T].
    """
    n = M.shape[0]
    identity = sp.sparse.eye(n, n, format='lil')
    return(identity - M[:, range(n)])

def get_T(M):
    """ Returns T when given [I-P, T].
    """
    n = M.shape[0]
    return(M[:, n])


""" Each of the following 7 functions returns a submatrix of a given matrix M. 
"""
def block_11(M, i):
    return M[0:i, 0:i]

def block_12(M, i):
    (_, m) = M.shape
    return M[0:i, i:m-1]

def block_12var(M, i, c):
    (_, m) = M.shape
    return M[0:i, i:m-c-1]

def block_13(M, i):
    (_, m) = M.shape
    return M[0:i, m-1:m]

def block_21(M, i):
    (n, _) = M.shape
    return M[i:n, 0:i]

def block_22(M, i):
    (n, m) = M.shape
    return M[i:n, i:m-1]

def block_23(M, i):
    (n, m) = M.shape
    return M[i:n, m-1:m]


def nodes_into_A(P, A):
    """ Returns a subset of [1,n] of nodes which are connected to edges leading
    into the set A of nodes as judged by the directed adjacency matrix P.
    """
    nodes = [node for node in range(0, P.shape[0]) if  np.any(P[node, A])]
    return nodes


def nodes_out_of_A(P, A):
    """ Returns a subset of [1,n] of nodes which are connected to edges leading
    out of the set A of nodes as judged by the directed adjacency matrix P.
    """
    nodes = [node for node in range(0, P.shape[0]) if  np.any(P[A, node])]
    return nodes


class LUdecomp:
    """ A class used to compute the updated transition probabilities and
    mean waiting times after elimination via block LU decomposition as
    described on page 10 of:
    S. MacKay, R & D. Robinson, J. (2018). Aggregation of Markov flows I:
    theory. Philosophical Transactions of The Royal Society A Mathematical
    Physical and Engineering Sciences. 376. 20170232. 10.1098/rsta.2017.0232.

    Attributes:
        aug_matrix: a sparse augmented matrix of the type formed by the
        function augment_matrix.
        nx: An integer denoting the number of noes to be eliminated.
    """
    def __init__(self, aug_matrix, nx):
        """ Initialises LUdecomp class.
        """
        self.aug_matrix = aug_matrix
        self.nx = nx


    def LYX(self):
        """ Returns the matrix L_{YX} from paper.
        """
        B11 = block_11(self.aug_matrix, self.nx).tocsc()
        LOGGER.debug('B11 = ')
        LOGGER.debug(B11)

        B21 = block_21(self.aug_matrix, self.nx).tocsc()
        LOGGER.debug('B21 = ')
        LOGGER.debug(B21)

        trans_B11 = sp.sparse.csc_matrix.transpose(B11).tocsc()
        LOGGER.debug(type(trans_B11))

        minus_trans_B21 = sp.sparse.csc_matrix.transpose(-B21).tocsc()
        LOGGER.debug(type(minus_trans_B21))

        trans_LYX = sp.sparse.linalg.spsolve(trans_B11, minus_trans_B21)

        ny = self.aug_matrix.shape[0] - self.nx
        if trans_LYX.shape != (self.nx, ny):
            trans_LYX = trans_LYX.reshape(self.nx, ny)

        trans_LYX = sp.sparse.csc_matrix(trans_LYX)
        LYX = sp.sparse.csc_matrix.transpose(trans_LYX)
        return LYX #this is a csr sparse matrix


    def new_PYY(self, LYX):
        """ Returns the updated transition probabilities for the remaining
        nodes.
        """
        B22 = block_22(self.aug_matrix, self.nx)
        cB22 = sp.sparse.csc_matrix(B22)
        B12 = block_12(self.aug_matrix, self.nx)
        cB12 = sp.sparse.csc_matrix(-B12)

        new_PYY = sp.sparse.eye(*B22.shape, format='csc') - cB22 + LYX*cB12
        return new_PYY.tocsc()


    def new_TY(self, LYX):
        """ Returns the updated mean waiting times for the remaining nodes.
        """
        B13 = block_13(self.aug_matrix, self.nx)
        cB13 = sp.sparse.csc_matrix(B13)
        B23 = block_23(self.aug_matrix, self.nx)
        cB23 = sp.sparse.csc_matrix(B23)

        new_TY = cB23 + LYX*cB13
        return new_TY.tocsc()

    @staticmethod
    def new_aug_matrix(new_PYY, new_TY):
        """ Returns the augmented for the eliminated Markov process.
        """
        return augment_matrix(new_PYY, new_TY).tocsc()


    def L(self, LYX):
        """ Returns the L matrix from the block LU decomposition form.
        """
        ny = self.aug_matrix.shape[0] - self.nx
        L21 = -LYX

        L = sp.sparse.bmat([[sp.sparse.eye(self.nx, self.nx), None], [L21, sp.sparse.eye(ny, ny)]])
        return L


    def U(self, new_PYY, new_TY):
        """ Returns the U matrix from the block LU decomposition form.
        """
        U11 = block_11(self.aug_matrix, self.nx)
        U12 = block_12(self.aug_matrix, self.nx)
        U13 = block_13(self.aug_matrix, self.nx)
        U22 = sp.sparse.eye(*new_PYY.shape) -new_PYY
        U23 = new_TY

        U = sp.sparse.bmat([[U11, U12, U13], [None, U22, U23]])
        return U


def get_rhox(LYX, rhoy):
    """Returns the (unnormalised) stationary probabilities for the nodes in
    the eliminated set X, given the (unnormalised) stationary probabilities for
    nodes in Y = X^c.
    """
    rhoyT = rhoy.transpose()
    rhox = rhoyT*LYX
    crhox = sp.sparse.csr_matrix(rhox.transpose())
    return crhox


def get_PiFromRho(rho, T):
    """ Returns the actual stationary distribution by reweighting rho according
    to T.
    """
    pi = np.multiply(rho, T)
    return pi


def elimination_pi(P, T):
    """ Returns the stationary distribution of the Markov process (P,T) by
    eliminating a single node at a time in the order speficied by the index
    of the transition matrix P. Refer to paper for theory.

    Args:
        P (scipy.sparse.lil_matrix): Matrix of transition probabilities.
        T (scipy.sparse.lil_matrix): Vector of mean waiting times.

    Returns:
        numpy.ndarray: The stationary distribution of the Markov process.
    """
    M = augment_matrix(P, T)
    N = M.shape[0]
    currentAugMatrix = M.tocsc()

    seq_LYX = []

    for i in range(0, N-1):
        decomp = LUdecomp(currentAugMatrix, 1)
        new_LYX = decomp.LYX()
        new_PYY = decomp.new_PYY(new_LYX)
        new_TY = decomp.new_TY(new_LYX)
        currentAugMatrix = decomp.new_aug_matrix(new_PYY, new_TY)
        if i == N-2:
            finalAugMatrix = decomp.new_aug_matrix(new_PYY, new_TY)
            rhoy = sp.sparse.csr_matrix([1/finalAugMatrix[0, 1]])

        seq_LYX.append(new_LYX)

    for i in reversed(range(0, N-1)):
        current_LYX = seq_LYX[i]
        rhox = get_rhox(current_LYX, rhoy)
        rhoy = sp.sparse.bmat([[rhox], [rhoy]])

    rho_final = rhoy
    rho_final_arr = rho_final.toarray()
    T_arr = T.toarray()
    final = np.multiply(rho_final_arr, T_arr)
    return final


def general_elimination_pi(P, T, order_to_eliminate):
    """ Returns the stationary distribution of the Markov process (P,T) by
    eliminating an arbitrary given number of nodes at a time in the order
    speficied by the index of the transition matrix P. Refer to paper for
    theory.

    Args:
        P (scipy.sparse.lil_matrix): Matrix of transition probabilities.
        T (scipy.sparse.lil_matrix): Vector of mean waiting times.
        order_to_eliminate (list of integers): Specifies how many nodes to
        eliminate at each iteration, order_to_eliminate[0] number being
        eliminated in the first iteration. The sum of the numbers should
        amount to no more than N = P.shape[0].

    Returns:
        numpy.matrixlib.defmatrix.matrix: The stationary distribution of the
        Markov process.
    """
    M = augment_matrix(P, T)
    iterations = len(order_to_eliminate)
    currentAugMatrix = M.tocsc()

    seq_LYX = []

    for i in range(0, iterations-1):
        decomp = LUdecomp(currentAugMatrix, order_to_eliminate[i])

        new_LYX = decomp.LYX()
        new_PYY = decomp.new_PYY(new_LYX)
        new_TY = decomp.new_TY(new_LYX)

        if i == iterations-2:
            final_P = new_PYY.todense()
            LOGGER.debug(final_P.shape)
            [S, U] = eig(final_P.transpose())
            stationary = np.matrix(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
            new_TY_dense = new_TY.todense().transpose()
            unnormalised_dist = np.multiply(stationary, new_TY_dense).transpose()
            rhoy = np.matrix(stationary)/(sum(unnormalised_dist))
            rhoy = rhoy.transpose()

        currentAugMatrix = augment_matrix(new_PYY, new_TY)
        seq_LYX.append(new_LYX)

    for i in reversed(range(0, iterations-1)):
        current_LYX = seq_LYX[i]
        rhox = get_rhox(current_LYX, rhoy)
        rhoy = sp.sparse.bmat([[rhox], [rhoy]])

    rho_final = rhoy
    rho_final_arr = rho_final.toarray()
    T_arr = T.toarray()
    final = np.multiply(rho_final_arr, T_arr)

    return np.asmatrix(final)

def calc_stationary_dist(P, T):
    """ Returns the stationary distribution of a Markov processes computed
    using scipy.linalg, an interface for the LAPACK and BLAS libraries. Used
    for comparison and testing of elimination method.

    Args:
        P (scipy.sparse.lil_matrix): Matrix of transition probabilities.
        T (scipy.sparse.lil_matrix): Vector of mean waiting times.

    Returns:
        numpy.ndarray: The stationary distribution of the Markov process.
    """
    P = P.toarray()
    P = np.asmatrix(P)

    [S, U] = eig(P.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary = stationary / np.sum(stationary)

    T_arr = T.transpose().toarray()

    unnormalised_dist = np.multiply(stationary, T_arr).transpose()
    return unnormalised_dist/sum(unnormalised_dist)

def calc_TAB(P, T, c):
    """ Returns the MFPT from A = V\B to B where B is a prespecified subest
    of the nodes. It is assumed that a permutation preprocesing step has been
    performed so that B corresponds to the indexes 0,1,..., c-1 of P. Nodes
    are eliminated one at a time. Refer to paper for theory.

    Args:
        P (scipy.sparse.lil_matrix): Matrix of transition probabilities.
        T (scipy.sparse.lil_matrix): Vector of mean waiting times.
        c (int): Speficfies B as the set 0,1,...,c-1.

    Returns:
        scipy.sparse.coo.coo_matrix: A vector of length |A| of the mean time
        taken to reach B from each element of A.
    """
    M = augment_matrix(P, T)
    N = M.shape[1]

    seq_M = []
    seq_I_PXX = []
    seq_TX = []
    seq_PXC = []

    new_M = M

    for i in range(0, N-c-1):
        seq_M.append(new_M)
        seq_I_PXX.append(block_11(new_M, 1))
        seq_TX.append(block_13(new_M, 1))
        seq_PXC.append(-block_12var(new_M, 1, c))

        seq_I_PXX[i] = sp.sparse.csr_matrix(seq_I_PXX[i])
        seq_TX[i] = sp.sparse.csr_matrix(seq_TX[i])
        seq_PXC[i] = sp.sparse.csr_matrix(seq_PXC[i])

        decomp = LUdecomp(new_M, 1)
        new_LYX = decomp.LYX()
        new_PYY = decomp.new_PYY(new_LYX)
        new_TY = decomp.new_TY(new_LYX)
        new_M = decomp.new_aug_matrix(new_PYY, new_TY)

    TB = sp.sparse.csr_matrix([sp.sparse.linalg.spsolve(seq_I_PXX[N-c-2], seq_TX[N-c-2])])

    for i in reversed(range(0, N-c-2)):
        TXB = sp.sparse.linalg.spsolve(seq_I_PXX[i], seq_TX[i] + seq_PXC[i]*TB)
        TB = sp.sparse.vstack([TXB, TB])
    return TB
