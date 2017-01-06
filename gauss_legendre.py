# ----------------------------------------- #
# One Point Gauss-Legendre
# ----------------------------------------- #
# Author: Mido Assran
# Date: Dec. 8, 2016
# Description: One Point Gauss-Legendre is a class that uses the one point
# Gauss-Legendre method to perform integration of a function over an
# arbitrary interval by splitting up into sub-intervals, and
# performing a coordinate mapping for compatability
# with the Gauss-Legendre interval.

import matplotlib.pyplot as plt


class OnePointGaussLegendre:

    def integrate(self, function, interval, num_segments):
        """
        :type function: lambda
        :type interval: tuple(float, float)
        :type num_segments: int
        :rtype: float
        """

        sub_interval_width = (interval[1] - interval[0]) / num_segments
        sub_intervals = [(interval[0] + i * sub_interval_width,
                          interval[0] + (i+1) * sub_interval_width)
                          for i in range(num_segments)]

        weights = [sub_interval_width for i in range(num_segments)]
        abscissas = [(x[0] + x[1]) * 0.5 for x in sub_intervals]

        integral = sum([weights[i] * function(abscissas[i])
                        for i in range(num_segments)])

        return integral

    def integrate_uneven(self, function, interval, num_segments):
        """
        :type function: lambda
        :type interval: tuple(float, float)
        :type num_segments: int
        :rtype: float
        """

        # Create uneqal spacings
        scalings = [i for i in range(1, num_segments+1)]
        scalings = [p / sum(scalings) for p in scalings]
        sub_interval_width = (interval[1] - interval[0]) / num_segments
        widths = [sub_interval_width / scalings[i] for i in range(num_segments)][::-1]
        widths = [widths[i] / sum(widths) for i in range(num_segments)]

        # Create sub_intervals
        sub_intervals = []
        running_lower = interval[0]
        for i in range(num_segments):
            l = running_lower
            u = l + widths[i]
            running_lower += widths[i]
            sub_intervals.append((l,u))

        weights = [widths[i] for i in range(num_segments)]
        abscissas = [(x[0] + x[1]) * 0.5 for x in sub_intervals]

        integral = sum([weights[i] * function(abscissas[i])
                        for i in range(num_segments)])

        return integral



if __name__ == "__main__":
    import math

    # print("\n", end="\n")
    # print("# ------------ Question 3 ------------ #", end="\n")
    # print("# ---------- Gauss-Legendre ---------- #", end="\n")
    # print("# ------------------------------------ #", end="\n\n")
    # opgl = OnePointGaussLegendre()
    # f = lambda x: math.sin(x)
    # print("N \t integrate(sin(x)) \t\t error")
    # rng = (0.0, 1.0)
    # truth = 0.45970
    # errors = []
    # n_vector = [i for i in range(1,21)]
    # for N in n_vector:
    #     result = opgl.integrate(function=f, interval=rng, num_segments=N)
    #     error = abs(truth-result)
    #     print(N, result, error, sep='\t')
    #     errors.append(error)
    # n_vector = [math.log(N, 10) for N in n_vector]
    # errors = [math.log(e, 10) for e in errors]
    # fig, ax = plt.subplots()
    # ax.plot(n_vector, errors, 'r', label="Error")
    # # ax.set_yscale('log')
    # # ax.set_xscale('log')
    # # ax.set_xlim([n_vector[0], n_vector[-1]])
    # # ax.set_ylim([errors[-1], errors[0]])
    # legend = ax.legend(loc='best', fontsize='small')
    # plt.title('Gauss-Legendre Error vs Resolution')
    # plt.ylabel('Error')
    # plt.xlabel('N (number of subintervals)')
    # plt.show()
    # print("# ------------------------------------ #", end="\n\n")
    #
    #
    # print("\n", end="\n")
    # print("# ------------ Question 3 ------------ #", end="\n")
    # print("# ---------- Gauss-Legendre ---------- #", end="\n")
    # print("# ------------------------------------ #", end="\n\n")
    # opgl = OnePointGaussLegendre()
    # f = lambda x: math.log(x)
    # print("N \t integrate(log(x)) \t\t error")
    # rng = (0.0, 1.0)
    # truth = -1.0
    # errors = []
    # n_vector = [i for i in range(1,21)]
    # for N in n_vector:
    #     result = opgl.integrate(function=f, interval=rng, num_segments=N)
    #     error = abs(truth-result)
    #     print(N, result, error, sep='\t')
    #     errors.append(error)
    # n_vector = [math.log(N, 10) for N in n_vector]
    # errors = [math.log(e, 10) for e in errors]
    # fig, ax = plt.subplots()
    # ax.plot(n_vector, errors, 'r', label="Error")
    # # ax.set_yscale('log')
    # # ax.set_xscale('log')
    # # ax.set_xlim([n_vector[0], n_vector[-1]])
    # # ax.set_ylim([errors[-1], errors[0]])
    # legend = ax.legend(loc='best', fontsize='small')
    # plt.title('Gauss-Legendre Error vs Resolution')
    # plt.ylabel('Error')
    # plt.xlabel('N (number of subintervals)')
    # plt.show()
    # print("# ------------------------------------ #", end="\n\n")


    print("\n", end="\n")
    print("# ------------ Question 3 ------------ #", end="\n")
    print("# ---------- Gauss-Legendre ---------- #", end="\n")
    print("# ------------------------------------ #", end="\n\n")
    opgl = OnePointGaussLegendre()
    f = lambda x: math.log(0.2 * abs(math.sin(x)))
    print("N \t integrate(log(0.2 |sin(x)|)) \t\t error")
    rng = (0.0, 1.0)
    truth = -2.662
    errors = []
    n_vector = [i for i in range(1,21)]
    for N in n_vector:
        result = opgl.integrate(function=f, interval=rng, num_segments=N)
        error = abs(truth-result)
        print(N, result, error, sep='\t')
        errors.append(error)
    n_vector = [math.log(N, 10) for N in n_vector]
    errors = [math.log(e, 10) for e in errors]
    test = [3*math.log(0.948 / ((N + 1)), 10) for N in n_vector]
    fig, ax = plt.subplots()
    ax.plot(n_vector, errors, 'r', label="Error")
    ax.plot(n_vector, test, 'b', label="Ideal")
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlim([n_vector[0], n_vector[-1]])
    # ax.set_ylim([errors[-1], errors[0]])
    legend = ax.legend(loc='best', fontsize='small')
    plt.title('Gauss-Legendre Error vs Resolution')
    plt.ylabel('Error')
    plt.xlabel('N (number of subintervals)')
    plt.show()
    print("# ------------------------------------ #", end="\n\n")


    # print("\n", end="\n")
    # print("# ------------ Question 3 ------------ #", end="\n")
    # print("# ---------- Gauss-Legendre ---------- #", end="\n")
    # print("# ------------------------------------ #", end="\n\n")
    # opgl = OnePointGaussLegendre()
    # f = lambda x: math.log(x)
    # print("\t\t\t \033[1m  integrate(log(x)) \t\t error")
    # print("\033[0m")
    # rng = (0.0, 1.0)
    # truth = -1.0
    # N = 10
    # result_even = opgl.integrate(function=f, interval=rng, num_segments=N)
    # error_even = abs(truth-result_even)
    # print("Even Spacing:", result_even, error_even, sep="\t\t")
    # result_uneven = opgl.integrate_uneven(function=f, interval=rng, num_segments=N)
    # error_uneven = abs(truth-result_uneven)
    # print("Uneven Spacing:", result_uneven, error_uneven, sep="\t\t")
    # print("# ------------------------------------ #", end="\n\n")
    #
    #
    #
    # print("\n", end="\n")
    # print("# ------------ Question 3 ------------ #", end="\n")
    # print("# ---------- Gauss-Legendre ---------- #", end="\n")
    # print("# ------------------------------------ #", end="\n\n")
    # opgl = OnePointGaussLegendre()
    # f = lambda x: math.log(0.2 * abs(math.sin(x)))
    # print("\t\t \033[1m  integrate(log(0.2 |sin(x)|)) \t\t error")
    # print("\033[0m")
    # rng = (0.0, 1.0)
    # truth = -2.662
    # N = 10
    # result_even = opgl.integrate(function=f, interval=rng, num_segments=N)
    # error_even = abs(truth-result_even)
    # print("Even Spacing:", result_even, error_even, sep="\t\t")
    # result_uneven = opgl.integrate_uneven(function=f, interval=rng, num_segments=N)
    # error_uneven = abs(truth-result_uneven)
    # print("Uneven Spacing:", result_uneven, error_uneven, sep="\t\t")
    # print("# ------------------------------------ #", end="\n\n")
