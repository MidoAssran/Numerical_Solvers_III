# ----------------------------------------- #
# Lagrange Subdomain Interpolation
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: LagrangeSubdomainInterpolator is a class that uses Lagrange
# polynomials to interpolate a data set by minimizing the least
# squares error with respect to the (domain, target) points.

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from choleski import CholeskiDecomposition
from polynomial_collective import LagrangePolynomial

class LagrangeSubdomainInterpolator:

    def __init__(self, dom, target):
        """
        :type dom: ndarray([float])
        :type target: ndarray([float])
        """
        self.dom, self.target = dom, target

        self.polynomials = []
        self.sub_doms = []
        for i, v in enumerate(dom[:-1]):
            sub_dom = dom[i:i+2]
            self.sub_doms.append(sub_dom)
            self.polynomials.append(
                (LagrangePolynomial(0, sub_dom), LagrangePolynomial(1, sub_dom))
            )

        self.degree = self.polynomials[0][0].degree

        self.params = self.determine_model_parameters()

        polys = self.polynomials
        self.coefficients = [
                [self.params[j] * polys[j][0].coefficients[k]
                + self.params[j+1] * polys[j][1].coefficients[k]
                for k in range(self.degree + 1)]
            for j in range(len(self.sub_doms))]

        # print(self.coefficients)
        print(self)

    def interpolate(self):
        """
        :rtype: tuple(ndarray([float]), ndarray([float]))
        """
        a, polys, sub_doms = self.params, self.polynomials, self.sub_doms
        y = []
        for i, sub_dom in enumerate(sub_doms):
            y.append(lambda x, i=i: a[i] * polys[i][0].evaluate(x) + a[i+1] * polys[i][1].evaluate(x))
        return y, sub_doms

    def determine_model_parameters(self):
        """
        :rtype: ndarray([float])
        """
        return self.target

    def determine_sub_domain_index(self, x, sub_doms):
        """
        :type x: float
        :type sub_doms: ndarray([float])
        :rtype: int
        """
        for i, rng in enumerate(sub_doms):
            x_min, x_max = rng[0], rng[1]
            if (x >= x_min) and (x <= x_max):
                return i
        return -1


    def __str__(self):

        r_str = "Polynomial:\n"
        for i, poly_coeff in enumerate(self.coefficients):
            r_str += "  Subdomain " + str(i) + ":\n   "
            l = ["({:.5})*x^{}".format(c, self.degree-i) for i, c in enumerate(poly_coeff)]
            r_str += " + ".join(l) + "\n"


        return r_str

if __name__ == "__main__":
    print("\n", end="\n")
    print("# ---------- Interpolation ---------- #", end="\n")
    print("# -------- 6 Separate Points -------- #", end="\n")
    print("# ----------------------------------- #", end="\n\n")
    B = np.array([0.0, 1.3, 1.4, 1.7, 1.8, 1.9])
    H = np.array([0.0, 540.6, 1062.8, 8687.4, 13924.3, 22650.2])
    dom, target = H, B
    lsi = LagrangeSubdomainInterpolator(dom=dom, target=target)
    y, sub_doms = lsi.interpolate()
    x_range = np.linspace(0.0, dom[-1], num=40000)
    interpolation = []
    for x in x_range:
        indx = lsi.determine_sub_domain_index(x, sub_doms)
        interpolation.append(y[indx](x))
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(dom, target, 'bo', label="Orignal BH")
    ax.plot(x_range, interpolation, 'r', label="Hermite BH")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('BH Interpolation')
    plt.ylabel('B')
    plt.xlabel('H')
    plt.show()
    print("# ----------------------------------- #", end="\n\n")
