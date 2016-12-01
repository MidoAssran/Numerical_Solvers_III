# ----------------------------------------- #
# Lagrange Interpolation
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: LagrangeInterpolator is a class that uses Lagrange
# polynomials to interpolate a data set by minimizing the least
# squares error with respect to the (domain, target) points.

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from choleski import CholeskiDecomposition
from polynomials import LagrangePolynomial

class LagrangeInterpolator:

    def __init__(self):
        self.polynomial = LagrangePolynomial()

    def interpolate(self, dom, target):
        """
        :type dom: ndarray([float])
        :type target: ndarray([float])
        :rtype: lambda(float x)
        """
        poly = self.polynomial
        n = len(dom)    # Number of domain coordinates
        a = self.determine_model_parameters(dom=dom, target=target)
        y = lambda x: sum([a[j] * poly.evaluate(x, j, dom) for j in range(n)])
        return y

    def determine_model_parameters(self, dom, target):
        """
        <Minimize least squares error with respect to the (domain, target)>
        :type dom: ndarray([float])
        :type target: ndarray([float])
        :rtype: ndarray([float])
        """
        poly = self.polynomial
        # Create Matrix
        G = np.empty([dom.shape[0], dom.shape[0]])
        G[:] = -1
        for i, row in enumerate(G):
            for j, _ in enumerate(row):
                if G[i,j] == -1:
                    G[i,j] = sum([poly.evaluate(x_k, i, dom) * poly.evaluate(x_k, j, dom) for x_k in dom])
                    G[j,i] = G[i,j]

        # Create target
        b = np.empty([dom.shape[0]])
        b[:] = -1
        for i, _ in enumerate(b):
            b[i] = sum([target[k] * poly.evaluate(x_k, i, dom) for k, x_k in enumerate(dom)])


        # Solve LSE using Choleski Decomposition: solve Ga = b
        A = matrix_dot_matrix(matrix_transpose(G), G)   # Make positive definite
        y = matrix_dot_vector(matrix_transpose(G), b)
        chol_d = CholeskiDecomposition()

        return chol_d.solve(A=A, b=y)


if __name__ == "__main__":
    print("\n", end="\n")
    print("# ---------- Interpolation ---------- #", end="\n")
    print("# ---------- First 6 Points --------- #", end="\n")
    print("# ----------------------------------- #", end="\n\n")
    B = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    H = np.array([0.0, 14.7, 36.5, 71.7, 121.4, 197.4])
    dom = H; target = B
    li = LagrangeInterpolator()
    y = li.interpolate(dom, target)
    x_range = np.linspace(0.0, H[-1], num=100000)
    interpolation = [y(x) for x in x_range]
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(H, B, 'bo', label="Orignal BH")
    ax.plot(x_range, interpolation, 'r', label="Lagrange BH")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('BH Interpolation')
    plt.ylabel('B')
    plt.xlabel('H')
    plt.show()
    print("# ----------------------------------- #", end="\n\n")


    print("\n", end="\n")
    print("# ---------- Interpolation ---------- #", end="\n")
    print("# -------- 6 Separate Points -------- #", end="\n")
    print("# ----------------------------------- #", end="\n\n")
    B = np.array([0.0, 1.3, 1.4, 1.7, 1.8, 1.9])
    H = np.array([0.0, 540.6, 1062.8, 8687.4, 13924.3, 22650.2])
    dom = H; target = B
    li = LagrangeInterpolator()
    y = li.interpolate(dom, target)
    x_range = np.linspace(0.0, H[-1], num=100000)
    interpolation = [y(x) for x in x_range]
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(H, B, 'bo', label="Orignal BH")
    ax.plot(x_range, interpolation, 'r', label="Lagrange BH")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('BH Interpolation')
    plt.ylabel('B')
    plt.xlabel('H')
    plt.show()
    print("# ----------------------------------- #", end="\n\n")
