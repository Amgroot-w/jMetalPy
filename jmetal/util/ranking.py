from abc import ABC, abstractmethod
from typing import TypeVar, List

from jmetal.util.comparator import DominanceComparator, Comparator, SolutionAttributeComparator

S = TypeVar('S')


class Ranking(List[S], ABC):

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(Ranking, self).__init__()
        self.number_of_comparisons = 0
        self.ranked_sublists = []
        self.comparator = comparator

    @abstractmethod
    def compute_ranking(self, solutions: List[S], k: int = None):
        pass

    def get_nondominated(self):
        return self.ranked_sublists[0]

    def get_subfront(self, rank: int):
        # 得到分层结果中的指定层数上的所有个体集合
        if rank >= len(self.ranked_sublists):
            raise Exception('Invalid rank: {0}. Max rank: {1}'.format(rank, len(self.ranked_sublists) - 1))
        return self.ranked_sublists[rank]

    def get_number_of_subfronts(self):
        return len(self.ranked_sublists)

    @classmethod
    def get_comparator(cls) -> Comparator:
        pass


class FastNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]_ """

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(FastNonDominatedRanking, self).__init__(comparator)

    # 这个函数的作用：得到分层的排序结果F（即第一层为NDSet，第二层为次之，以此类推）
    def compute_ranking(self, solutions: List[S], k: int = None):
        """ Compute ranking of solutions.

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]

        for p in range(len(solutions) - 1):
            for q in range(p + 1, len(solutions)):
                # 调用compare方法，比较两个个体；
                # 默认的比较方法是：先比较约束条件的违反程度，若相同，再比较目标函数值；
                # 注意：算法认为目标函数值小的个体更好，因此在jmetal底层算法实现中，统一认为算法的进化方向是：最小化目标函数值！！！
                # 如果问题需要求最大值，那么就在重写类problem的evaluate()方法时，将该维目标加上负号即可！
                dominance_test_result = self.comparator.compare(solutions[p], solutions[q])
                self.number_of_comparisons += 1

                if dominance_test_result == -1:   # 结果为-1，表示p支配q
                    ith_dominated[p].append(q)    # 在被p支配的个体清单中加上q
                    dominating_ith[q] += 1        # 能够支配q的个体数目加1
                elif dominance_test_result is 1:  # 结果为1，表示q支配p
                    ith_dominated[q].append(p)    # 在被q支配的个体清单中加上p
                    dominating_ith[p] += 1        # 能够支配p的个体数目加1

        # 下面代码作用：得到front[0]，即NDSet，即不被其他任何个体支配的个体的集合
        for i in range(len(solutions)):
            if dominating_ith[i] is 0:
                front[0].append(i)
                solutions[i].attributes['dominance_ranking'] = 0

        # 下面代码作用：得到分层之后的其他层中的个体，并为其他所有个体赋予一个层数
        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] is 0:
                            front[i].append(q)
                            solutions[q].attributes['dominance_ranking'] = i

        # 下面代码作用：得到分层结果（上面得到的分层结果中只是存储了个体的index，现在要得到存储了相应个体地分层结果）
        self.ranked_sublists = [[]] * i
        for j in range(i):
            q = [0] * len(front[j])
            for m in range(len(front[j])):
                q[m] = solutions[front[j][m]]
            self.ranked_sublists[j] = q

        # 下面代码作用：如果设置了个体的数量上限，则取分层结果的一部分作为结果返回
        if k:
            count = 0
            for i, front in enumerate(self.ranked_sublists):
                count += len(front)
                if count >= k:
                    self.ranked_sublists = self.ranked_sublists[:i + 1]
                    break

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator('dominance_ranking')


class StrengthRanking(Ranking[List[S]]):
    """ Class implementing a ranking scheme based on the strength ranking used in SPEA2. """

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(StrengthRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: int = None):
        """
        Compute ranking of solutions.

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        strength: [int] = [0 for _ in range(len(solutions))]
        raw_fitness: [int] = [0 for _ in range(len(solutions))]

        # strength(i) = | {j | j < - SolutionSet and i dominate j} |
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) < 0:
                    strength[i] += 1

        # Calculate the raw fitness:
        # rawFitness(i) = |{sum strength(j) | j <- SolutionSet and j dominate i}|
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) == 1:
                    raw_fitness[i] += strength[j]

        max_fitness_value: int = 0
        for i in range(len(solutions)):
            solutions[i].attributes['strength_ranking'] = raw_fitness[i]
            if raw_fitness[i] > max_fitness_value:
                max_fitness_value = raw_fitness[i]

        # Initialize the ranked sublists. In the worst case will be max_fitness_value + 1 different sublists
        self.ranked_sublists = [[] for _ in range(max_fitness_value + 1)]

        # Assign each solution to its corresponding front
        for solution in solutions:
            self.ranked_sublists[int(solution.attributes['strength_ranking'])].append(solution)

        # Remove empty fronts
        counter = 0
        while counter < len(self.ranked_sublists):
            if len(self.ranked_sublists[counter]) == 0:
                del self.ranked_sublists[counter]
            else:
                counter += 1

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator('strength_ranking')
