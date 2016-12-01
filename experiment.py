# ----------------------------------------- #
# Experiment
# ----------------------------------------- #
# Author: Mido Assran
# Date: Nov. 10, 2016
# Description: Experiment solves the finite difference equations using
# both the ConjugateGradientFiniteDifferencePotentialSolver and the
# CholeskiDecomposition solver, and performs postprocessing to
# plot and compare the results.

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from utils import *

B = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    1.7, 1.8, 1.9])

H = np.array([0.0, 14.7, 36.5, 71.7, 121.4, 197.4, 256.2, 348.7, 540.6,
    1062.8, 2318.0, 4781.9, 8687.4, 13924.3, 22650.2])


# # F_j is the numerator of the Lagrange Polynomial
# _p_f = lambda x, r: map(lambda r, x=x: (x-r), filter(lambda r, x=x: x != r, r))
# F = lambda x, r: reduce(lambda x, y: x* y, _p_f(x,r))
# lgr_ply = lambda x, i: F(x, i)

def lagrange_polynomial(x, j, dom):
    """
    :type x: float
    :type j: int
    :type dom: np.array([float])
    :rtype: float
    """

    def F(x=x, j=j):
        val = 1.0
        for r, x_r in enumerate(dom):
            if r != j:
                val *= (x - x_r)
        return val

    # Return the evaluated lagrange polynomial
    return F(x, j) / F(dom[j], j)


def determine_parameters(dom, target):
    # One parameter per polynomial
    # One polynomial per domain point
    # Each, polynomial is (n-1)th degree

    # Create Matrix
    G = np.empty([dom.shape[0], dom.shape[0]])
    G[:] = -1
    for i, row in enumerate(G):
        for j, _ in enumerate(row):
            if G[i,j] == -1:
                G[i,j] = sum([lagrange_polynomial(x, i, dom) * lagrange_polynomial(x_k, j, dom) for x_k in dom])
                G[j,i] = G[i,j]

    # Create target
    b = np.empty([dom.shape[0]])
    b[:] = -1
    for i, _ in enumerate(b):
        b[i] = sum([target(k) * lagrange_polynomial(x_k, i, dom) for k, x_k in enumerate(dom)])

    # Solve LSE
    

def interpolate_first_n(n=6):




if __name__ == "__main__":

    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(H, B, 'b', label="Orignal BH")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('BH Interpolation')
    plt.ylabel('B')
    plt.xlabel('H')
    plt.show()
