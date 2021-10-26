import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution, Solution, FloatSolution, IntegerSolution, PermutationSolution, \
    CompositeSolution
from jmetal.util.ckecking import Check

"""
.. module:: mutation
   :platform: Unix, Windows
   :synopsis: Module implementing mutation operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullMutation(Mutation[Solution]):
    # 不变异，直接返回父代
    def __init__(self):
        super(NullMutation, self).__init__(probability=0)

    def execute(self, solution: Solution) -> Solution:
        return solution

    def get_name(self):
        return 'Null mutation'


class BitFlipMutation(Mutation[BinarySolution]):
    # 二进制编码：直接翻转
    def __init__(self, probability: float):
        super(BitFlipMutation, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        Check.that(type(solution) is BinarySolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            for j in range(len(solution.variables[i])):
                rand = random.random()  # 注意是对每一位都判断一下概率
                if rand <= self.probability:
                    solution.variables[i][j] = True if solution.variables[i][j] is False else False

        return solution

    def get_name(self):
        return 'BitFlip mutation'


class PolynomialMutation(Mutation[FloatSolution]):
    # 多项式变异（实数编码）
    def __init__(self, probability: float, distribution_index: float = 0.20):
        super(PolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        for i in range(solution.number_of_variables):
            rand = random.random()  # 对每一位都进行变异概率判断

            if rand <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl  # 当前位置的上下界值相等，直接赋值为该值
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)

                    # 修复超过边界的解
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = y

        return solution

    def get_name(self):
        return 'Polynomial mutation'


class IntegerPolynomialMutation(Mutation[IntegerSolution]):
    # （int型）多项式变异，适用于整数编码：在上一个的基础上直接取了个整而已
    def __init__(self, probability: float, distribution_index: float = 0.20):
        super(IntegerPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    rnd = random.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                        deltaq = 1.0 - val ** mut_pow

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = int(round(y))  # 取整
        return solution

    def get_name(self):
        return 'Polynomial mutation (Integer)'


class SimpleRandomMutation(Mutation[FloatSolution]):
    # 简单随机变异：在下界和上界之间随机产生一个数作为变异值
    def __init__(self, probability: float):
        super(SimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = solution.lower_bound[i] + \
                                        (solution.upper_bound[i] - solution.lower_bound[i]) * random.random()
        return solution

    def get_name(self):
        return 'Simple random_search mutation'


class UniformMutation(Mutation[FloatSolution]):
    # 均匀变异
    def __init__(self, probability: float, perturbation: float = 0.5):
        super(UniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                # random.random(): 产生一个[0,1)范围内的随机数, 减去0.5是为了将中心移动到0处，即产生[-0.5,0.5)范围内的随机数
                tmp = (random.random() - 0.5) * self.perturbation
                tmp += solution.variables[i]  # 当前值 + 因子*产生的随机数
                # 修正解
                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def get_name(self):
        return 'Uniform mutation'


class NonUniformMutation(Mutation[FloatSolution]):
    # 非均匀变异
    # 该变异方法将进化代数考虑在内，在进化过程的不同时期，产生变异值的策略不同（服从的分布不同），由当前进化代数决定
    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 0.5):
        super(NonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                rand = random.random()

                if rand <= 0.5:
                    tmp = self.__delta(solution.upper_bound[i] - solution.variables[i], self.perturbation)
                else:
                    tmp = self.__delta(solution.lower_bound[i] - solution.variables[i], self.perturbation)

                tmp += solution.variables[i]
                # 修正解
                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def set_current_iteration(self, current_iteration: int):
        # 获取当前的进化代数
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        # 该变异方法将进化代数考虑在内
        return (y * (1.0 - pow(random.random(),
                               pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter))))

    def get_name(self):
        return 'Non-Uniform mutation'


class PermutationSwapMutation(Mutation[PermutationSolution]):
    # 互换突变：适用于排列编码
    # 随机产生两个互换位点，然后交换各自的值
    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        Check.that(type(solution) is PermutationSolution, "Solution type invalid")

        rand = random.random()

        if rand <= self.probability:
            # 随机产生互换位点
            pos_one, pos_two = random.sample(range(solution.number_of_variables - 1), 2)
            # 互换各自的值
            solution.variables[pos_one], solution.variables[pos_two] = \
                solution.variables[pos_two], solution.variables[pos_one]

        return solution

    def get_name(self):
        return 'Permutation Swap mutation'


class CompositeMutation(Mutation[Solution]):
    # 混合型编码，调用各自的变异算子进行变异
    def __init__(self, mutation_operator_list:[Mutation]):
        super(CompositeMutation,self).__init__(probability=1.0)

        Check.is_not_none(mutation_operator_list)
        Check.collection_is_not_empty(mutation_operator_list)

        self.mutation_operators_list = []
        for operator in mutation_operator_list:
            Check.that(issubclass(operator.__class__, Mutation), "Object is not a subclass of Mutation")
            self.mutation_operators_list.append(operator)

    def execute(self, solution: CompositeSolution) -> CompositeSolution:
        Check.is_not_none(solution)
        # 依次变异，最后再组装到一起
        mutated_solution_components = []
        for i in range(solution.number_of_variables):
            mutated_solution_components.append(self.mutation_operators_list[i].execute(solution.variables[i]))

        return CompositeSolution(mutated_solution_components)

    def get_name(self) -> str:
        return "Composite mutation operator"


class ScrambleMutation(Mutation[PermutationSolution]):
    # 在随机选择的染色体片段之间，重新排列
    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        rand = random.random()

        if rand <= self.probability:
            point1 = random.randint(0, len(solution.variables))
            point2 = random.randint(0, len(solution.variables) - 1)

            # 保证point1 < point2
            if point2 >= point1:
                point2 += 1
            else:
                point1, point2 = point2, point1
            # 保证两个位点之间距离不超过20
            if point2 - point1 >= 20:
                point2 = point1 + 20

            values = solution.variables[point1:point2]
            solution.variables[point1:point2] = random.sample(values, len(values))

        return solution

    def get_name(self):
        return "Scramble"



