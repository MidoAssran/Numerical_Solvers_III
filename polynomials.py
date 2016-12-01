# ----------------------------------------- #
# Polynomials
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: Collection of polynomial classes / data structures.

class LagrangePolynomial:

    def evaluate(self, x, j, dom):
        """
        :type x: float
        :type j: int
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

class HermitePolynomial:

    def evaluate(self, x, j, dom):
        """
        :type x: float
        :type j: int
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
