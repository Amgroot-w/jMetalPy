
from math import pi, sin, sqrt, cos
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
"""
.. module:: UF
   :platform: Unix, Windows
   :synopsis: Problems of the CEC2009 multi-objective competition

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class UF1(FloatProblem):
    """ Problem UF1.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super(UF1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        sum1 = 0
        sum2 = 0
        count1 = 0
        count2 = 0

        x = solution.variables

        for i in range(2, self.number_of_variables):
            y = x[i-1] - sin(6.0 * pi * x[0] + i * pi/solution.number_of_variables)
            y = y*y

            if i % 2 is 0:
                sum2 += y
                count2 +=1
            else:
                sum1 +=y
                count1 += 1

        solution.objectives[0] = x[0] + 2.0 * sum1 /(1.0 * count1)
        solution.objectives[1] = 1.0 - sqrt(x[0]) + 2.0 * sum2 / (1.0 * count2)

        return solution

    def get_name(self):
        return 'UF1'


class UF8(FloatProblem):
    """ Problem UF8.

    .. note:: Unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super(UF8, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self.number_of_objectives)]

        self.lower_bound = self.number_of_variables * [-2.0]
        self.upper_bound = self.number_of_variables * [2.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0
        self.lower_bound[1] = 0.0
        self.upper_bound[1] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        count1 = 0  #
        count2 = 0
        count3 = 0

        x = solution.variables

        for i in range(3, self.number_of_variables):

            y = x[i - 1] - 2 * x[1] * (sin(2 * pi * x[0] + (i * pi / solution.number_of_variables)))
            y = y * y

            if (i - 1) % 3 is 0:
                sum1 += y
                count1 += 1
            if (i - 2) % 3 is 0:
                sum2 += y
                count2 += 1
            if (i - 3) % 3 is 0:
                sum3 += y
                count3 += 1

        solution.objectives[0] = (cos(0.5 * x[0] * pi)) * (cos(0.5 * x[1] * pi)) + 2.0 * sum1 / (1.0 * count1)
        solution.objectives[1] = (cos(0.5 * x[0] * pi)) * (sin(0.5 * x[1] * pi)) + 2.0 * sum2 / (1.0 * count2)
        solution.objectives[2] = (sin(0.5 * x[0] * pi)) + 2.0 * sum3 / (1.0 * count3)

        return solution

    def get_name(self):
        return 'UF8'
