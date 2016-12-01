# ----------------------------------------- #
# Choleski Decomposition
# ----------------------------------------- #
# Author: Mido Assran
# Date: 30, September, 2016
# Description: CholeskiDecomposition solves the linear system of equations:
# Ax = b by decomposing matrix A using Choleski factorization and using
# forward and backward substitution to determine x. Matrix A must
# be symmetric, real, and positive definite.

import random
import timeit
import numpy as np
from utils import matrix_transpose

DEBUG = True

class CholeskiDecomposition(object):

    def __init__(self):
        if DEBUG:
            np.core.arrayprint._line_width = 200

    def solve(self, A, b, band=None):
        """
        :type A: np.array([float])
        :type b: np.array([float])
        :type band: int
        :rtype: np.array([float])
        """

        start_time = timeit.default_timer()

        # If the matrix, A, is banded, leverage that!
        if band is not None:
            self._band = band

        # If the matrix, A, is not square, exit
        if A.shape[0] != A.shape[1]:
            return "Matrix 'A' is not square!"

        n = A.shape[1]


        # -------------------------------------------------------------- #
        # Simultaneous Choleski factorization of A and chol-elimination
        # -------------------------------------------------------------- #
        # Choleski factorization & forward substitution
        for j in range(n):

            # If the matrix A is not positive definite, exit
            if A[j,j] <= 0:
                return "Matrix 'A' is not positive definite!"

            A[j,j] = A[j,j] ** 0.5    # Compute the j,j entry of chol(A)
            b[j] /= A[j,j]            # Compute the j entry of forward-sub

            for i in range(j+1, n-1):

                # Banded matrix optimization
                if (band is not None) and (i == self._band):
                    self._band += 1
                    break

                A[i,j] /= A[j,j]      # Compute the i,j entry of chol(A)
                b[i] -= A[i,j] * b[j] # Look ahead modification of b

                if A[i,j] == 0:       # Optimization for matrix sparsity
                    continue

                # Look ahead moidification of A
                for k in range(j+1, i+1):
                    A[i,k] -= A[i,j] * A[k,j]

            # Perform computation for the test source
            if (j != n-1):
                A[n-1,j] /= A[j,j]        # Compute source entry of chol(A)
                b[n-1] -= A[n-1,j] * b[j] # Look ahead modification of b
                # Look ahead moidification of A
                for k in range(j+1, n):
                    A[n-1,k] -= A[n-1,j] * A[k,j]
        # -------------------------------------------------------------- #


        # -------------------------------------------------------------- #
        # Now solve the upper traingular system
        # -------------------------------------------------------------- #
        # Transpose(A) is the upper-tiangular matrix of chol(A)
        A[:] = matrix_transpose(A)

        # Backward substitution
        for j in range(n - 1, -1, -1):
            b[j] /= A[j,j]

            for i in range(j):
                b[i] -= A[i,j] * b[j]
        # -------------------------------------------------------------- #

        elapsed_time = timeit.default_timer() - start_time

        if DEBUG:
            print("Execution time:\n", elapsed_time, end="\n\n")

        # The solution was overwritten in the vector b
        return b

if __name__ == "__main__":
    from utils import generate_positive_semidef, matrix_dot_vector

    order = 10
    seed = 5

    print("\n", end="\n")
    print("# --------------- TEST --------------- #", end="\n")
    print("# ------ Choleski Decomposition ------ #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    chol_d = CholeskiDecomposition()
    # Create a symmetric, real, positive definite matrix.
    A = generate_positive_semidef(order=order, seed=seed)
    x = np.random.randn(order)
    b = matrix_dot_vector(A=A, b=x)
    print("A:\n", A, end="\n\n")
    print("x:\n", x, end="\n\n")
    print("b (=Ax):\n", b, end="\n\n")
    v = chol_d.solve(A=A, b=b)
    print("result = solve(A, b):\n", v, end="\n\n")
    print("2-norm error:\n", np.linalg.norm(v - x), end="\n\n")
    print("# ------------------------------------ #", end="\n\n")
