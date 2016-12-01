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
from utils import *



if __name__ == "__main__":

    print("\n", end="\n")
    print("# --------------- TEST --------------- #", end="\n")
    print("# ------ Choleski Decomposition ------ #", end="\n")
    print("# --------- (Preconditioned) --------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    chol_d = CholeskiDecomposition()
    # Create a symmetric, real, positive definite matrix.
    A = matrix_dot_matrix(matrix_transpose(cgfdps._A), cgfdps._A)
    b = matrix_dot_matrix(matrix_transpose(cgfdps._A), cgfdps._b)

    print("A:\n", A, end="\n\n")
    print("b:\n", b, end="\n\n")
    v = chol_d.solve(A=A, b=b)
    print("result = solve(A, b):\n", v, end="\n\n")

    node_number = cgfdps.map_coordinates_to_node((0.06, 0.04))
    potential = v[node_number]
    print("Potential (0.06, 0.04):\n", potential, end="\n\n")
    print("# ------------------------------------ #", end="\n\n")


    # Perform postprocessing of ConjugateGradient residual history
    fig, ax = plt.subplots()
    norm_2 = [np.linalg.norm(v) for i, v in enumerate(residual_history)]
    norm_inf = [np.linalg.norm(v, np.inf) for i, v in enumerate(residual_history)]
    ax.plot(norm_2, 'r', label="2-norm")
    ax.plot(norm_inf, 'b', label="inf-norm")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('Residual Norm vs Iteration')
    plt.ylabel(r'$||b - Ax||$')
    plt.xlabel('iteration')
    plt.show()
