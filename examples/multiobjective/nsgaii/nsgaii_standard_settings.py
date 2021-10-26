from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.solution import get_non_dominated_solutions, read_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations


"""  
Program to  configure and run the NSGA-II algorithm configured with standard settings.
"""

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())

    # *****************************************************************************************************************
    from jmetal.core.quality_indicator import InvertedGenerationalDistance
    reference_front = [s.objectives for s in problem.reference_front]  # 从solution变量中提取np数组形式的reference_front
    solutions_arr = [s.objectives for s in algorithm.get_result()]  # 从solution变量中提取np数组形式的solutions_arr
    igd = InvertedGenerationalDistance(reference_front)  # igd指标实例化，传入参考PF
    igd_value = igd.compute(solutions_arr)  # 调用compute函数，传入算法得到的PF
    # *****************************************************************************************************************

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))

