# ----------------------------------------- #
# Polynomials
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 1, 2016
# Description: Polynomial data structure and utils class.

class Polynomial:

    def __init__(self):
        self.degree = 0
        self.coefficients = []

    def multiply_binomial(self, binomial_arg):
        # Lazy instantiation
        if self.degree == 0:
            self.degree += 1
            self.coefficients.append(1.0)
            self.coefficients.append(binomial_arg)

        else:
            self.degree += 1
            self.coefficients.append(0.0)

            temp = []
            for i, v in enumerate(self.coefficients):
                if i == 0:
                    temp.append(self.coefficients[i])
                else:
                    temp.append(v +  binomial_arg * self.coefficients[i-1])
            self.coefficients = temp


    def divide_scalar(self, scalar):
        self.coefficients = [v / scalar for v in self.coefficients]

    def evaluate(self, x):
        val = 0
        for i, c in enumerate(self.coefficients):
            d = self.degree - i
            val += c * (x ** d)
        return val

    def __str__(self):
        l = ["{:.5}x^{}".format(c, self.degree-i) for i, c in enumerate(self.coefficients)]
        return " + ".join(l)
