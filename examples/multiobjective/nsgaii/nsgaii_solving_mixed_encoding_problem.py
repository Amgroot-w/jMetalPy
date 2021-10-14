from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation, IntegerPolynomialMutation
from jmetal.operator.crossover import CompositeCrossover, IntegerSBXCrossover
from jmetal.operator.mutation import CompositeMutation
from jmetal.problem.multiobjective.unconstrained import MixedIntegerFloatProblem
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = MixedIntegerFloatProblem(10, 10, 100, -100, -1000, 1000)

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        # 这里的变异算子虽然是composite变异，但实现的时候还是调用的单个个体变异，不同类型的全部变异完毕之后，最后组装到一起
        mutation=CompositeMutation([IntegerPolynomialMutation(0.01, 20), PolynomialMutation(0.01, 20.0)]),
        # 这里的交叉算子虽然是composite交叉，但实现的时候还是调用的同类型个体交叉，不同类型的全部交叉完毕之后，最后组装到一起
        crossover=CompositeCrossover([IntegerSBXCrossover(probability=1.0, distribution_index=20),
                                      SBXCrossover(probability=1.0, distribution_index=20)]),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
