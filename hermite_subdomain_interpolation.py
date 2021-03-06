# ----------------------------------------- #
# Hermite Subdomain Interpolation
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: HermiteSubdomainInterpolator is a class that uses Hermite
# polynomials to interpolate a data set by minimizing the least
# squares error with respect to the (domain, target) points.

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from choleski import CholeskiDecomposition
from polynomial_collective import HermiteSubdomainPolynomial

class HermiteSubdomainInterpolator:

    def __init__(self, dom, target):
        """
        :type dom: ndarray([float])
        :type target: ndarray([float])
        """
        self.dom, self.target = dom, target
        self.polynomial = HermiteSubdomainPolynomial()

    def interpolate(self):
        """
        :rtype: tuple(list(lambda(float x)), list([float, float]))
        """
        n = 2    # Number of points in subdomain
        sub_doms = []
        poly, dom, target = self.polynomial, self.dom, self.target
        a, b = self.determine_model_parameters()
        y = []
        for i, v in enumerate(dom[:-1]):
            sub_doms.append(dom[i:i+2])
            y.append(lambda x, i=i: \
                    sum([a[i+j] * poly.evaluate_U(x=x, j=j, dom=dom[i:i+2]) \
                    + b[i+j] * poly.evaluate_V(x=x, j=j, dom=dom[i:i+2]) \
                    for j in range(n)]))


        return y, sub_doms

    def determine_model_parameters(self):
        """
        :type dom: ndarray([float])
        :type target: ndarray([float])
        :rtype: tuple(ndarray([float]), ndarray([float]))
        """
        dom, target = self.dom, self.target
        a = target
        b = []
        for i, _ in enumerate(target):
            if i == 0:
                b.append((target[i+1] - target[i]) / (dom[i+1] - dom[i]))
            elif i == len(target) - 1:
                b.append((target[i] - target[i-1]) / (dom[i] - dom[i-1]))
            else:
                s1 = (target[i+1] - target[i]) / (dom[i+1] - dom[i])
                s2 = (target[i] - target[i-1]) / (dom[i] - dom[i-1])
                w = s2 / (s1 + s2)
                b.append(w * s1 + (1.0 - w) * s2)

        return a, b

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


if __name__ == "__main__":
    print("\n", end="\n")
    print("# ------------ Question 1 ----------- #", end="\n")
    print("# ---------- Interpolation ---------- #", end="\n")
    print("# -------- 6 Separate Points -------- #", end="\n")
    print("# ----------------------------------- #", end="\n\n")
    B = np.array([0.0, 1.3, 1.4, 1.7, 1.8, 1.9])
    H = np.array([0.0, 540.6, 1062.8, 8687.4, 13924.3, 22650.2])
    dom, target = B, H
    hsi = HermiteSubdomainInterpolator(dom=dom, target=target)
    y, sub_doms = hsi.interpolate()
    x_range = np.linspace(0.0, dom[-1], num=40000)
    interpolation = []
    for x in x_range:
        indx = hsi.determine_sub_domain_index(x, sub_doms)
        interpolation.append(y[indx](x))
    # Perform postprocessing
    fig, ax = plt.subplots()
    ax.plot(dom, target, 'bo', label="Orignal BH")
    ax.plot(x_range, interpolation, 'r', label="Hermite BH")
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('HB Interpolation')
    plt.ylabel('H')
    plt.xlabel('B')
    plt.show()
    print("# ----------------------------------- #", end="\n\n")
