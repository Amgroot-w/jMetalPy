## MOEA/D算法整体流程

**进入algotrithm.run():**

0. time.time()计时开始；

1. **产生初始解：self.creat_initial_solutions()**

   调用的是default_generator，即：RandomGenerator， 即：problem.create_solution()。

2. **评价初始解：self.evaluate(self.solutions)**

   调用的是default_evaluator，即：SequentialEvaluator，即：problem.evaluate()，运行完毕后，每个solution的objectives就被计算好了。其实算法类的evaluate方法就是调用的problem内定义的evaluate方法。

3. **初始化进程：self.init_progress()**  --> 在MOEAD算法中被重构
   + 初始化种群大小；
   + 根据所有solution的目标函数值objectives来初始化理想点，调用的是self.fitness_function.update()函数，其中fitness_function是由算法初始化时的传入参数决定，即WeightedSum、Tschebycheff；
   + 初始化self.permutation，这个类变量存储的是种群中所有个体index的一个随机排列，具有get_next_value()方法，一次能返回一个个体的index；
   + 初始化observable，这个主要用来观测进化过程。

4. **执行进化过程（直到停止条件被满足）**
   + 4.1 进入self.step()：--> 在EvolutionaryAlgorithm中定义
     + 4.1.1 选择：self.selection()  --> **在MOEAD中定义**
       + 随机选择哪种方法产生子种群（通过生成一个随机数）：①在整个pop中选择；②在当前个体的领域中选择。选择算子是NaryRandomSolutionSelection(2)：产生的结果包含两个个体。
       + 将当前个体append到选择算子产生的子种群中，得到含有3个个体的种群。
     + 4.1.2 交叉&变异：self.reproduction()  --> **在MOEAD中定义**
       + 差分进化交叉：输入含3个个体的种群，输出一个交叉结果（list型，含一个元素）；
       + 变异：对这一个个体执行变异操作（变异算子没有默认值，需要用户传入相应算子）。
     + 4.1.3 评价生成的子种群：self.evaluate()  --> 沿用EvolutionaryAlgorithm中的定义
     + 4.1.4 进化选择：self.replacement()  --> **在MOEAD中定义**
       + 获取新生成的解，并通过其目标函数值去更新适应度函数的理想点，调用的是：self.fitness_function.update(new_solution.objectives)；
       + 更新当前个体的领域，调用并进入：self.update_current_subproblem_neighborhood()
         + 调用self.generate_permutation_of_neighbors()获取当前个体的领域，用个体对应的index的集合来表示该领域；有两种方法：①调用self.neighbourhood.get_neighborhood()方法获取当前个体的领域；②调用Permutation().get_permutation()在整个pop上获取指定数目的领域，具体用哪种方法是由前面过程产生的随机数决定的。
         + 调用self.fitness_function.compute()函数，依次计算领域中所有个体的适应度，并与当前个体进行比较，如果当前个体要优于领域内其他个体，则用当前个体去替换其他个体；同时，对替换的次数进行计数，若达到设置的最大替换次数，则break；
   + 4.2 进入self.update_progress()：--> 在EvolutionaryAlgorithm中定义
     + 调用observable更新进程。

5. time.time()计时结束。





