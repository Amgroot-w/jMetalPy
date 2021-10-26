import logging
import threading
import time
from abc import abstractmethod, ABC
from typing import TypeVar, Generic, List

from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :platform: Unix, Windows
   :synopsis: Templates for algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Algorithm(Generic[S, R], threading.Thread, ABC):

    def __init__(self):
        threading.Thread.__init__(self)

        self.solutions: List[S] = []
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.observable = store.default_observable

    @abstractmethod
    def create_initial_solutions(self) -> List[S]:
        """ Creates the initial list of solutions of a metaheuristic. """
        pass

    @abstractmethod
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """ Evaluates a solution list. """
        pass

    @abstractmethod
    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        pass

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        pass

    @abstractmethod
    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        pass

    @abstractmethod
    def get_observable_data(self) -> dict:
        """ Get observable data, with the information that will be send to all observers each time. """
        pass

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        LOGGER.debug('Initializing progress')
        self.init_progress()

        LOGGER.debug('Running main loop until termination criteria is met')
        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    @abstractmethod
    def get_result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class DynamicAlgorithm(Algorithm[S, R], ABC):

    @abstractmethod
    def restart(self) -> None:
        pass


class EvolutionaryAlgorithm(Algorithm[S, R], ABC):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int):
        super(EvolutionaryAlgorithm, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

    @abstractmethod
    def selection(self, population: List[S]) -> List[S]:
        """ Select the best-fit individuals for reproduction (parents). """
        pass

    @abstractmethod
    def reproduction(self, population: List[S]) -> List[S]:
        """ Breed new individuals through crossover and mutation operations to give birth to offspring. """
        pass

    @abstractmethod
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ Replace least-fit population with new individuals. """
        pass

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def init_progress(self) -> None:
        # 这个变量表示总评价次数，初始化为population_size是因为初始化的时候就已经评价了population_size次，因为初始化的时候
        # 种群中的每个个体都要计算其初始的目标函数值。
        self.evaluations = self.population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

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

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'


class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]], ABC):

    def __init__(self,
                 problem: Problem[S],
                 swarm_size: int):
        super(ParticleSwarmOptimization, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'
