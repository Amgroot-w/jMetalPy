from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: genetic_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Genetic Algorithms.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class GeneticAlgorithm(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        super(GeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size)
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

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

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

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

    # 这个函数一般会被具体的算法重写！因此此处的排序法算法几乎没有用到（也有点简单粗暴，只是按照第一维目标从小到大排序而已）
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        # 将父代种群和子代种群合并
        population.extend(offspring_population)
        # 将合并后的种群按照第一维目标函数值从小到大排序（也就是说算法的进化方向为：目标函数最小化！）
        population.sort(key=lambda s: s.objectives[0])
        # 取排序后靠前的population_size个个体，剩余的个体丢弃
        return population[:self.population_size]

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Genetic algorithm'
