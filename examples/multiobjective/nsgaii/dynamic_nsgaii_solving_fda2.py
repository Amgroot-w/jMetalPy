from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.observable import TimeCounter
from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver, ProgressBarObserver, VisualizerObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = FDA2()

    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )
    # 添加observer，观测进化过程
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.observable.register(observer=PlotFrontToFileObserver('dynamic_front_vis'))
    algorithm.observable.register(observer=WriteFrontToFileObserver('dynamic_front'))

    algorithm.run()
