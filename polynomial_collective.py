# ----------------------------------------- #
# Polynomials
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: Collection of polynomial classes / data structures.
from polynomial import Polynomial

class LagrangePolynomial:

    def __init__(self, j, dom):
        polynomial = Polynomial()
        for r, x_r in enumerate(dom):
            if r != j:
                polynomial.multiply_binomial(-1.0 * x_r)
        polynomial.divide_scalar(polynomial.evaluate(dom[j]))
        self.polynomial = polynomial
        self.coefficients = polynomial.coefficients
        self.degree = polynomial.degree

    def evaluate(self, x):
        return self.polynomial.evaluate(x)

    def __str__(self):
        l = ["{:.5}x^{}".format(c, self.degree-i) for i, c in enumerate(self.coefficients)]
        return " + ".join(l)


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
