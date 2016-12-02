# ----------------------------------------- #
# Lagrange Interpolation
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: LagrangeInterpolator is a class that uses Lagrange
# polynomials to interpolate a data set by minimizing the least
# squares error with respect to the (domain, target) points.

import numpy as np
from utils import *
from choleski import CholeskiDecomposition
from polynomial_collective import LagrangePolynomial

class LagrangeInterpolator:

    def __init__(self, dom, target):
        """
        :type dom: ndarray([float])
        :type target: ndarray([float])
        """
        self.dom, self.target = dom, target

        self.polynomials = []
        for j in range(len(dom)):
            self.polynomials.append(LagrangePolynomial(j, dom))

        self.degree = self.polynomials[0].degree

         # Least squares curve fitting
        self.params = self.determine_model_parameters()

        self.coefficients = [
            sum([self.params[j] * lp.coefficients[k]
                for j, lp in enumerate(self.polynomials)])
            for k in range(self.degree + 1)]

        print(self)

    def interpolate(self):
        """
        :rtype: lambda(float x)
        """
        a, polys, dom, target = self.params, self.polynomials, self.dom, self.target
        n = len(dom)    # Number of domain coordinates
        y = lambda x: sum([a[j] * polys[j].evaluate(x) for j in range(n)])
        return y

    def determine_model_parameters(self):
        """
        < Least Squares curve fitting >
        :rtype: ndarray([float])
        """
        polys, dom, target = self.polynomials, self.dom, self.target

        # Create Matrix
        G = np.empty([dom.shape[0], dom.shape[0]])
        G[:] = -1
        for i, row in enumerate(G):
            for j, _ in enumerate(row):
                if G[i,j] == -1:
                    G[i,j] = sum([polys[i].evaluate(x_k) * polys[j].evaluate(x_k) for x_k in dom])
                    G[j,i] = G[i,j]

        # Create target
        b = np.empty([dom.shape[0]])
        b[:] = -1
        for i, _ in enumerate(b):
            b[i] = sum([target[k] * polys[i].evaluate(x_k) for k, x_k in enumerate(dom)])


        # Solve LSE using Choleski Decomposition: solve Ga = b
        A = matrix_dot_matrix(matrix_transpose(G), G)   # Make positive definite
        y = matrix_dot_vector(matrix_transpose(G), b)
        chol_d = CholeskiDecomposition()

        return chol_d.solve(A=A, b=y)

    def __str__(self):
        l = ["({:.5})*x^{}".format(c, self.degree-i) for i, c in enumerate(self.coefficients)]
        return "Polynomial:\n   " + " + ".join(l) + "\n"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("\n", end="\n")
    print("# ---------- Interpolation ---------- #", end="\n")
    print("# ---------- First 6 Points --------- #", end="\n")
    print("# ----------------------------------- #", end="\n\n")
    B = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    H = np.array([0.0, 14.7, 36.5, 71.7, 121.4, 197.4])
    dom, target = H, B
    li = LagrangeInterpolator(dom=dom, target=target)
    y = li.interpolate()
    x_range = np.linspace(0.0, dom[-1], num=10000)
    interpolation = [y(x) for x in x_range]
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(dom, target, 'bo', label="Orignal BH")
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
    dom, target = H, B
    li = LagrangeInterpolator(dom=dom, target=target)
    y = li.interpolate()
    x_range = np.linspace(0.0, dom[-1], num=10000)
    interpolation = [y(x) for x in x_range]
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(dom, target, 'bo', label="Orignal BH")
    ax.plot(x_range, interpolation, 'r', label="Lagrange BH")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('BH Interpolation')
    plt.ylabel('B')
    plt.xlabel('H')
    plt.show()
    print("# ----------------------------------- #", end="\n\n")
