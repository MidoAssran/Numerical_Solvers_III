# ----------------------------------------- #
# Newton Raphson Solver
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: NewtonRaphsonSolver solves non-linear equations by using
# the Newton-Raphson method.

import random
import numpy as np
import matplotlib.pyplot as plt
from utils import matrix_dot_matrix, matrix_transpose, matrix_dot_vector
from choleski import CholeskiDecomposition

class NewtonRaphsonSolver:

    def solve(self, starting_guess, f, g, stopping_ratio=1e-6):
        """
        :type starting_guess: float
        :type stopping_ratio: float
        :type f: lambda(float)
        :type g: lambda(float)
        :rtype: dict('num_steps': int, 'arg_history': list(float))
        """
        x = starting_guess
        arg_history = [x]

        progress_ratio = abs(f(x) / f(starting_guess))
        itr = 0
        while progress_ratio > stopping_ratio:
            itr += 1
            x += -1.0 * f(x) / g(x)
            arg_history.append(x)
            progress_ratio = abs(-1.0 * f(x) / f(starting_guess))

        return {'num_steps': itr, 'arg_history': arg_history}


    def solve_2D(self, starting_guess, f, J, stopping_ratio=1e-6):
        """
        :type starting_guess: ndarray([float, float])
        :type stopping_ratio: float
        :type f: list(lambda(float, float))
        :type J: list(lambda(float, float))
        :rtype: dict('num_steps': int,
                     'arg_history': list([float, float]),
                     'error_history': list(float))
        """
        x = starting_guess
        arg_history = [x]
        chol = CholeskiDecomposition()
        error = np.linalg.norm([f[0](x[0], x[1]), f[1](x[0], x[1])])/200e-3
        error_history = []

        print('iteration \t voltages \t\t\t\t\t f \t\t\t\t\t error')

        itr = 0
        while error > stopping_ratio:
            itr += 1
            A = np.empty([2,2])
            A[0,0] = Jacobian[0][0](x[0], x[1])
            A[0,1] = Jacobian[0][1](x[0], x[1])
            A[1,0] = Jacobian[1][0](x[0], x[1])
            A[1,1] = Jacobian[1][1](x[0], x[1])
            b = np.empty([2])
            b[0] = -f[0](x[0], x[1])
            b[1] = -f[1](x[0], x[1])
            b = matrix_dot_vector(matrix_transpose(A), b)
            A = matrix_dot_matrix(matrix_transpose(A), A)
            temp = chol.solve(A=A, b=b)
            x += temp
            arg_history.append(x)
            error = np.linalg.norm([f[0](x[0], x[1]), f[1](x[0], x[1])])/200e-3
            error_history.append(error)
            print(itr, x, b, error, sep='\t\t')


        return {'num_steps': itr, 'arg_history': arg_history, 'error_history': error_history}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lagrange_subdomain_interpolation import LagrangeSubdomainInterpolator

    print("\n", end="\n")
    print("# ------------ Question 1 ------------ #", end="\n")
    print("# ---------- Newton-Raphson ---------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    # Piece-wise Linear Interpolation of data
    B = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2,
                  1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    H = np.array([0.0, 14.7, 36.5, 71.7, 121.4, 197.4, 256.2, 348.7, 540.6,
                  1062.8, 2318.0, 4781.9, 8687.4, 13924.3, 22650.2])
    dom, target = B, H
    lsi = LagrangeSubdomainInterpolator(dom=dom, target=target)
    y, sub_doms = lsi.interpolate()
    x_range = np.linspace(0.0, dom[-1], num=20)
    A = 1e-2 * 1e-2
    h_phi = lambda phi, lsi=lsi, sub_doms=sub_doms, A=A: y[lsi.determine_sub_domain_index(phi/A, sub_doms)](phi/A)
    g_phi = lambda phi, lsi=lsi, sub_doms=sub_doms, A=A: lsi.coefficients[lsi.determine_sub_domain_index(phi/A, sub_doms)][0]/A
    R_g = 0.5e-2 / (1e-4 * 1.25663706e-6)
    L_c = 30e-2
    M = 800.0 * 10.0
    objective = lambda phi, h_phi=h_phi: R_g * phi + L_c * h_phi(phi) - M
    derivative = lambda phi, g_phi=g_phi: R_g + L_c * g_phi(phi)
    xtst = - (objective(0.0))
    nrs = NewtonRaphsonSolver()
    starting_guess = 0.0
    print("nrs.solve(starting_guess=" + str(starting_guess), "):", sep="")
    result = nrs.solve(starting_guess=starting_guess, f=objective, g=derivative)
    print("\n   num_steps: ", result['num_steps'], "    flux: ", result['arg_history'][-1])
    print("# ----------------------------------- #", end="\n\n")


    print("\n", end="\n")
    print("# ------------ Question 1 ------------ #", end="\n")
    print("# ---------- Successive-Sub ---------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    # Piece-wise Linear Interpolation of data
    B = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2,
                  1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    H = np.array([0.0, 14.7, 36.5, 71.7, 121.4, 197.4, 256.2, 348.7, 540.6,
                  1062.8, 2318.0, 4781.9, 8687.4, 13924.3, 22650.2])
    dom, target = B, H
    lsi = LagrangeSubdomainInterpolator(dom=dom, target=target)
    y, sub_doms = lsi.interpolate()
    x_range = np.linspace(0.0, dom[-1], num=20)
    A = 1e-2 * 1e-2
    h_phi = lambda phi, lsi=lsi, sub_doms=sub_doms, A=A: y[lsi.determine_sub_domain_index(phi/A, sub_doms)](phi/A)
    R_g = 0.5e-2 / (1e-4 * 1.25663706e-6)
    L_a = 30e-2
    M = 800.0 * 10.0
    objective = lambda phi: 1e-10 * (R_g * phi + (L_a * h_phi(phi) - M))
    derivative = lambda phi: 1.0
    nrs = NewtonRaphsonSolver()
    starting_guess = 0.0
    print("nrs.solve(starting_guess=" + str(starting_guess), "):", sep="")
    result = nrs.solve(starting_guess=starting_guess, f=objective, g=derivative)
    print("\n   num_steps: ", result['num_steps'], "    flux: ", result['arg_history'][-1])
    print("# ----------------------------------- #", end="\n\n")


    print("\n", end="\n")
    print("# ------------ Question 2 ------------ #", end="\n")
    print("# ---------- Newton-Raphson ---------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    E = 200e-3; kT_q = 25e-3; R = 512; I_sa = 0.8e-6; I_sb = 1.1e-6

    Jacobian = [[0, 0],[0, 0]]
    Jacobian[0][0] = lambda v1, v2: 1.0 + (R * I_sa / kT_q) * np.exp((v1-v2) / kT_q)
    Jacobian[1][0] = lambda v1, v2: (I_sa / kT_q) * np.exp((v1-v2) / kT_q)
    Jacobian[0][1] = lambda v1, v2: - (R * I_sa / kT_q) * np.exp((v1-v2) / kT_q)
    Jacobian[1][1] = lambda v1, v2: - (I_sa / kT_q) * np.exp((v1-v2) / kT_q) - (I_sb / kT_q) * np.exp(v2 / kT_q)

    objective = [0, 0]
    objective[0] = lambda v1, v2: v1 - E + (R * I_sa) * (np.exp((v1-v2) / kT_q) - 1.0)
    objective[1] = lambda v1, v2: I_sa * (np.exp((v1-v2) / kT_q) - 1.0) - I_sb * (np.exp(v2/ kT_q) - 1.0)

    nrs = NewtonRaphsonSolver()
    starting_guess = np.array([0.0, 0.0])
    print("nrs.solve(starting_guess=" + str(starting_guess), "):", sep="", end="\n\n")
    result = nrs.solve_2D(starting_guess=starting_guess, f=objective, J=Jacobian)
    print("\nnum_steps: ", result['num_steps'], "    voltages: ", result['arg_history'][-1])

    n1_error = [1.0 / (2 ** (2 ** (i+1 / 0.9))) for i, e in enumerate(result['error_history'])]
    n_error = [e for i, e in enumerate(result['error_history'])]
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.semilogy(n_error, 'r', label="Error")
    ax.semilogy(n1_error, '--b', label="Ideal Quadratic Convergence")
    legend = ax.legend(loc='best', fontsize='small')
    plt.show()
    print("# ----------------------------------- #", end="\n\n")
