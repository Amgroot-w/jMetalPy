from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
from jmetal.problem import DTLZ2
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.solution import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem import ZDT1, DTLZ1, UF8
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.lab.visualization import Plot, InteractivePlot

if __name__ == '__main__':

    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
    print('当前运行路径更改为：', os.getcwd())

    problem = ZDT1()

    max_evaluations = 2000

    algorithm = MOEAD(
        problem=problem,
        population_size=100,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=20,
        weight_files_path='resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    hypervolume = HyperVolume([1.0, 1.0, 1.0])
    print("Hypervolume: " + str(hypervolume.compute([front[i].objectives for i in range(len(front))])))

    # 2.静态图：static
    plot_front = Plot(title='Pareto front approximation. Problem: ' + problem.get_name(), reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

