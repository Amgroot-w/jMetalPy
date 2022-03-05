import time
from typing import TypeVar, List

from jmetal.config import store
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.comparator import MultiComparator


S = TypeVar('S')
R = TypeVar('R')

class NSGAII:

    def __init__(self,
                 problem,
                 population_size,
                 offspring_population_size,
                 mutation,
                 crossover,
                 selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])),
                 termination_criterion = store.default_termination_criteria,
                 population_generator = store.default_generator,
                 population_evaluator = store.default_evaluator,
                 dominance_comparator = store.default_comparator):

        super().__init__()

        self.solutions: List[S] = []
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.observable = store.default_observable


        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.dominance_comparator = dominance_comparator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.mating_pool_size = \
            self.offspring_population_size * \
            self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem)
                for _ in range(self.population_size)]

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def init_progress(self) -> None:
        # 这个变量表示总评价次数，初始化为population_size是因为初始化的时候就已经评价了population_size次，因为初始化的时候
        # 种群中的每个个体都要计算其初始的目标函数值。
        self.evaluations = self.population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        mating_population = self.selection(self.solutions)  # 选择
        offspring_population = self.reproduction(mating_population)  # 交叉、变异
        offspring_population = self.evaluate(offspring_population)  # 评价子代个体

        self.solutions = self.replacement(self.solutions, offspring_population)  # 进化选择：确定最终的子代

    def update_progress(self) -> None:
        """
        ！！！注意：在jMetalpy中，StoppingByEvaluations指的是达到最大的评价次数就终止进化，注意是评价次数而不是进化代数！！！！！！！！
            * 在jMetalpy中，每一次进化迭代，都会累加上这次迭代所进行的评价次数，并判断是否满足最大评价次数，若满足则停止；
            * 在每一次的进化迭代中，评价次数 = 生成的子种群个数；（因为新生成1个个体，就需要评价它的目标函数1次）
            * 有时候1个进化迭代过程，可能会产生很多次评价，而具体有多少次评价，是由算法本身决定的，例如MOEA/D中，一次迭代只产生一个个体（即对
              其中一个权重向量进行更新），那么每次进化都只增加一次评价次数；而在NSGA-II中，每次进化生成的个体数由输入参数offspring_population_size
              决定，因此每1次进化迭代，都会增加很多很多次评价次数，也就是说，不需要几次进化，整个算法就结束了！！！！！
        """
        self.evaluations += self.offspring_population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    # jMetalPy的crossover和mutation均包含在了reproduction中
    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        # 注意这里的parents数设置是有要求的！它必须能够整除种群大小！
        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []  # 初始化子代种群
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            # 1.先产生parents
            parents = []
            for j in range(number_of_parents_to_combine):
                # 这里的根据pop选取parents的方法是：假如设定的parents个数为2，则在pop中从前往后按顺序每次取2个个体作为parents
                parents.append(mating_population[i + j])

            # 2.交叉算子：对每一对parents做交叉，产生对应的一对offspring个体
            offspring = self.crossover_operator.execute(parents)

            # 3.变异算子：对交叉操作产生的offspring逐个进行变异操作
            for solution in offspring:
                self.mutation_operator.execute(solution)  # 变异
                offspring_population.append(solution)  # 将完成变异的offspring加入子代种群中
                if len(offspring_population) >= self.offspring_population_size:
                    break  # 若子代种群的个体数满足要求，则break

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:

        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'NSGAII'

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'