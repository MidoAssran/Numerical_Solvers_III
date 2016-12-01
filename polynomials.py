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

class HermiteSubdomainPolynomial:

    def _lagrange(self, x, j, dom):
        if j == 0:
            return (x - dom[j+1]) / (dom[j] - dom[j+1])
        elif j == 1:
            return (x - dom[j-1]) / (dom[j] - dom[j-1])

    def _lagrange_p(self, j, dom):
        if j== 0:
            return 1.0 / float(dom[j] - dom[j+1])
        elif j == 1:
            return 1.0 / float(dom[j] - dom[j-1])

    def evaluate_U(self, x, j, dom):
        return (1 - 2 * self._lagrange_p(j, dom) * (x - dom[j])) * (self._lagrange(x, j, dom)**2)

    def evaluate_V(self, x, j, dom):
        return (x - dom[j]) * (self._lagrange(x, j, dom) ** 2)
