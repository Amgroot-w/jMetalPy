import copy
import random
from typing import List

from jmetal.core.operator import Crossover
from jmetal.core.solution import Solution, FloatSolution, BinarySolution, PermutationSolution, IntegerSolution, \
    CompositeSolution
from jmetal.util.ckecking import Check

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""

# 1.不交叉：直接返回两个parents
class NullCrossover(Crossover[Solution, Solution]):
    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Null crossover'


class PMXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    """
    部分匹配交叉，适用于组合优化
    """
    def __init__(self, probability: float):
        super(PMXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        permutation_length = offspring[0].number_of_variables

        rand = random.random()
        if rand <= self.probability:
            cross_points = sorted([random.randint(0, permutation_length) for _ in range(2)])

            def _repeated(element, collection):
                # 检查element在collection中是否有重复（即出现次数大于1）
                c = 0
                for e in collection:
                    if e == element:
                        c += 1
                return c > 1

            def _swap(data_a, data_b, cross_points):
                # 对传入的两条染色体data_a和data_b，在给定的cross_points中互换片段
                c1, c2 = cross_points
                new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
                new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
                return new_a, new_b

            def _map(swapped, cross_points):
                n = len(swapped[0])
                c1, c2 = cross_points
                s1, s2 = swapped
                map_ = s1[c1:c2], s2[c1:c2]  # 获取swapped的一个copy
                for i_chromosome in range(n):
                    # 只对互换片段之外的片段部分进行检查
                    if not c1 < i_chromosome < c2:
                        # 分别对两个互换后的染色体进行操作
                        for i_son in range(2):
                            # 检查第i_chromosome个变量在该染色体中是否重复
                            while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                                # 如果重复了，将该变量的index提取出来
                                map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                                # 将重复位置的元素用另一个染色体对应位置的元素替换
                                swapped[i_son][i_chromosome] = map_[1 - i_son][map_index]
                return s1, s2

            swapped = _swap(parents[0].variables, parents[1].variables, cross_points)  # 交换片段
            mapped = _map(swapped, cross_points)  # 对交换片段之外的重复元素进行互换

            offspring[0].variables, offspring[1].variables = mapped

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Partially Matched crossover'


class CXCrossover(Crossover[PermutationSolution, PermutationSolution]):
    """
    循环交叉，适用于组合优化
        2021.10.26：这个bug已经在jmetalpy的develop分支里面被修复了。。。只不过一年多了都没没有合并到master分支上，也就是说这一年以来新学
    jmetalpy的用户都会遇到这个bug。。。
    """
    def __init__(self, probability: float):
        super(CXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        # 注意这里offspring在初始化的时候，直接反转了父代的顺序
        offspring = [copy.deepcopy(parents[1]), copy.deepcopy(parents[0])]
        rand = random.random()

        if rand <= self.probability:
            # idx = random.randint(0, len(parents[0].variables[i]) - 1)  # 此处有个小bug
            idx = random.randint(0, len(parents[0].variables) - 1)
            curr_idx = idx
            cycle = []

            while True:
                cycle.append(curr_idx)
                # curr_idx = parents[0].variables[i].index(parents[1].variables[i][curr_idx])
                curr_idx = parents[1].variables.index(parents[0].variables[curr_idx])

                if curr_idx == idx:
                    break

            # for j in range(len(parents[0].variables[i])):
            for j in range(len(parents[0].variables)):
                if j in cycle:
                    offspring[0].variables[j] = parents[0].variables[j]
                    offspring[1].variables[j] = parents[1].variables[j]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Cycle crossover'


class SBXCrossover(Crossover[FloatSolution, FloatSolution]):
    """
    模拟二进制交叉
    """
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index  # 公式中的n值，该值越大，交叉生成的子代与父代越接近
        if distribution_index < 0:
            raise Exception("The distribution index is negative: " + str(distribution_index))

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        # parents要求：必须为两个，且均为float型变量
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]  # 初始化子代
        rand = random.random()

        # SBX算法涉及概率1（染色体重组概率）和概率2（条件重组概率），且强调概率1，把概率2设为定值0.5
        if rand <= self.probability:  # 概率1：染色体重组概率（满足该概率，执行交叉操作）
            for i in range(parents[0].number_of_variables):  # 逐个位点进行交叉操作
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]
                if random.random() <= 0.5:  # 概率2：条件重组概率（满足该概率，在染色体最小片段上执行交叉操作）
                    # 如果两个父代的差距大于阈值，我们才认为它们是不同的，此时再进行后续交叉
                    if abs(value_x1 - value_x2) > self.__EPS:
                        # 保证y1<y2
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))  # 计算子代个体1

                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))  # 计算子代个体2

                        # 修复超过边界的解
                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[i] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[i] = c2
                    # 如果两个父代的差距小于阈值，就认为它是相同的，此时不进行交叉，直接赋值给子代
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'SBX crossover'


class IntegerSBXCrossover(Crossover[IntegerSolution, IntegerSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(IntegerSBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        # 多了一个修正环节：float型 -> int型
                        if random.random() <= 0.5:
                            offspring[0].variables[i] = int(c2)
                            offspring[1].variables[i] = int(c1)
                        else:
                            offspring[0].variables[i] = int(c1)
                            offspring[1].variables[i] = int(c2)
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Integer SBX crossover'


class SPXCrossover(Crossover[BinarySolution, BinarySolution]):
    """
    单点交叉：适用于二进制编码
    """
    def __init__(self, probability: float):
        super(SPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(type(parents[1]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = parents[0].get_total_number_of_bits()

            # 2. Calculate the point to make the crossover
            crossover_point = random.randrange(0, total_number_of_bits)

            # 3. Compute the variable containing the crossover bit
            variable_to_cut = 0
            bits_count = len(parents[1].variables[variable_to_cut])
            while bits_count < (crossover_point + 1):
                variable_to_cut += 1
                bits_count += len(parents[1].variables[variable_to_cut])

            # 4. Compute the bit into the selected variable（找到被切的variable的index）
            diff = bits_count - crossover_point
            crossover_point_in_variable = len(parents[1].variables[variable_to_cut]) - diff

            # 5. Apply the crossover to the variable（刚好被切的那个染色体，对切点以后的染色体片段进行互换）
            bitset1 = copy.copy(parents[0].variables[variable_to_cut])
            bitset2 = copy.copy(parents[1].variables[variable_to_cut])

            for i in range(crossover_point_in_variable, len(bitset1)):
                swap = bitset1[i]
                bitset1[i] = bitset2[i]
                bitset2[i] = swap

            offspring[0].variables[variable_to_cut] = bitset1
            offspring[1].variables[variable_to_cut] = bitset2

            # 6. Apply the crossover to the other variables（后面的整段variables直接替换）
            for i in range(variable_to_cut + 1, parents[0].number_of_variables):
                offspring[0].variables[i] = copy.deepcopy(parents[1].variables[i])
                offspring[1].variables[i] = copy.deepcopy(parents[0].variables[i])

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Single point crossover'


class DifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """ This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random_search one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.

    翻译翻译：best and rand variants取决于传入的3个父代中的第三个，取决于它是best个体还是random个体。而这两种对应的实现方法是一致的，因为传入
            什么父代由外部决定，与本class无关。
    """

    # ***注意：在jMetalPy中，差分进化(DE)算法的实现包括两个类：
    #           -- 差分进化选择（DifferentialEvolutionSelection）
    #           -- 差分进化交叉（DifferentialEvolutionCrossover）
    #         没有差分进化变异！！！而且以上两个类必须配套使用！！！
    #
    # 在这里，jMetalPy只实现了 DE/rand/1/bin 和 DE/best/1/bin 两个DE算法，具体调用的是哪个，由传入的三个父代个体中的第三个父代个体决定，如
    # 果第三个个体时rand个体，就是DE/rand/1/bin算法，如果是best个体，就是DE/best/1/bin算法。
    # 其他的DE变体，如DE/rand/2/bin, DE/best/2/bin, DE/rand-to-best/bin, DE/current-to-rand/bin, DE/current-to-best/bin, 需要
    # 自己实现，直接集成crossover类，然后重写execute()方法即可。

    def __init__(self, CR: float, F: float, K: float = 0.5):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR  # 交叉概率
        self.F = F  # 缩放因子
        self.K = K  # 这个变量是干嘛用的？

        self.current_individual: FloatSolution = None  # “current_individual”这个变量会在外部被赋值！！！

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """ Execute the differential evolution crossover ('best/1/bin' variant in jMetal).
        """
        if len(parents) != self.get_number_of_parents():  # parents数目必须为3
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)

        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)

        # DE算法中，对pop中的每一个个体逐个进行进化操作，而不是像别的算法那样随机选择，导致有的个体会从未被选择上
        for i in range(number_of_variables):
            # 当i=rand时不用判断直接交叉，当i为其他值时以self.CR为概率进行交叉
            if random.random() < self.CR or i == rand:
                # 执行差分进化交叉操作
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])
                # 修复超过边界的解
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            # 不满足交叉概率，不进行交叉，直接赋值
            else:
                value = child.variables[i]  # 直接将当前个体的在该位点的值赋给该位点

            child.variables[i] = value  # 将经过交叉操作后的实数值赋值给该位点

        # 注意：标准DE算法中的“选择”算子，没有在此处实现，而是在算法类中实现（算法GDE3类的replacement()方法中，实现了这个“选择”过程）

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'Differential Evolution crossover'


class CompositeCrossover(Crossover[CompositeSolution, CompositeSolution]):
    __EPS = 1.0e-14

    def __init__(self, crossover_operator_list:[Crossover]):
        super(CompositeCrossover, self).__init__(probability=1.0)

        Check.is_not_none(crossover_operator_list)
        Check.collection_is_not_empty(crossover_operator_list)

        self.crossover_operators_list = []
        for operator in crossover_operator_list:
            Check.that(issubclass(operator.__class__, Crossover), "Object is not a subclass of Crossover")
            self.crossover_operators_list.append(operator)

    def execute(self, solutions: List[CompositeSolution]) -> List[CompositeSolution]:
        Check.is_not_none(solutions)
        Check.that(len(solutions) == 2, "The number of parents is not two: " + str(len(solutions)))

        offspring1 = []
        offspring2 = []

        number_of_solutions_in_composite_solution = solutions[0].number_of_variables
        # 依次交叉并产生子代，最后再组装到一起
        for i in range(number_of_solutions_in_composite_solution):
            parents = [solutions[0].variables[i], solutions[1].variables[i]]
            children = self.crossover_operators_list[i].execute(parents)
            offspring1.append(children[0])
            offspring2.append(children[1])

        return [CompositeSolution(offspring1), CompositeSolution(offspring2)]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Composite crossover'
