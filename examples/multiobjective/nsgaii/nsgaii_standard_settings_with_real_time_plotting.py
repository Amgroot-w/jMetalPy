from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1, DTLZ1, UF8
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution import read_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
Program to  configure and run the NSGA-II algorithm configured with standard settings.
"""

if __name__ == '__main__':

    # # 更改工作路径，若不更改则无法读取参考PF面
    # # 无参考PF面不会影响算法运行结果，只是最后画图的时候没有画出参考PF面而已！
    # # **见源码jmetal\lab\visualization\streaming.py中StreamingPlot类
    # import os
    # os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    max_evaluations = 2000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    # 1.实时图：streaming
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    # 运行算法
    algorithm.run()
    front = algorithm.get_result()

    # 2.静态图：static
    plot_front = Plot(title='Pareto front approximation. Problem: ' + problem.get_name(), reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # 3.交互图：interactive
    # 3.1 基于plotly框架的HTML交互
    plot_front = InteractivePlot(title='Pareto front approximation. Problem: ' + problem.get_name(), reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())
    # 3.2 基于matplotlib的和弦图
    from jmetal.lab.visualization.chord_plot import chord_diagram
    chord_diagram(solutions=front)  # chord_diagram是个函数，不是一个类，因此直接运行此函数即可

    # # Save results to file
    # print_function_values_to_file(front, 'FUN.' + algorithm.label)
    # print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))