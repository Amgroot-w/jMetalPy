from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, List

import numpy

from jmetal.core.solution import Solution
from jmetal.util.ckecking import Check

"""
.. module:: neighborhood
   :platform: Unix, Windows
   :synopsis: implementation of neighborhoods in the context of list of solutions. The goal is,
   given the index of an element of the list, to find its neighbour solutions according to a criterion.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

S = TypeVar('S')


class Neighborhood(Generic[S], ABC):

    @abstractmethod
    def get_neighbors(self, index: int, solution_list: List[S]) -> List[S]:
        pass


class WeightNeighborhood(Neighborhood[Solution], ABC):

    def __init__(self,
                 number_of_weight_vectors: int,
                 neighborhood_size: int,
                 weight_vector_size: int = 2,
                 weights_path: str = None):
        self.number_of_weight_vectors = number_of_weight_vectors
        self.neighborhood_size = neighborhood_size
        self.weight_vector_size = weight_vector_size
        self.weights_path = weights_path

        self.neighborhood = numpy.zeros((number_of_weight_vectors, neighborhood_size), dtype=int)
        self.weight_vectors = numpy.zeros((number_of_weight_vectors, weight_vector_size))


class WeightVectorNeighborhood(WeightNeighborhood):
    """ 用来产生权重向量、初始化领域、获取领域等操作 """
    def __init__(self,
                 number_of_weight_vectors: int,
                 neighborhood_size: int,
                 weight_vector_size: int = 2,
                 weights_path: str = None):
        super(WeightVectorNeighborhood, self).__init__(number_of_weight_vectors, neighborhood_size, weight_vector_size,
                                                       weights_path)
        self.__initialize_uniform_weight(weight_vector_size, number_of_weight_vectors)  # 初始化均匀分布的权重
        self.__initialize_neighborhood()  # 初始化权重向量的领域

    """
    自己的算法可以定义一个类，集成这个类并重写下面这个函数,,,,,,重写不了啊啊？？因为这玩意儿已经在__init__()里面调用了！
    """
    # 初始化均匀分布的权重向量（在构造函数时即调用）
    def __initialize_uniform_weight(self, weight_vector_size: int, number_of_weight_vectors: int) -> None:
        """ Precomputed weights from

        * Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II

        Downloaded from:

        * http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar
        """
        # 二维的权重向量，通过下面的程序产生
        if weight_vector_size == 2:
            for i in range(0, number_of_weight_vectors):
                v = 1.0 * i / (number_of_weight_vectors - 1)
                self.weight_vectors[i, 0] = v
                self.weight_vectors[i, 1] = 1 - v

        # 三维及以上的权重向量，通过读取文件产生
        else:
            file_name = 'W{}D_{}.dat'.format(weight_vector_size, number_of_weight_vectors)
            file_path = self.weights_path + '/' + file_name

            if Path(file_path).is_file():
                with open(file_path) as file:
                    for index, line in enumerate(file):
                        vector = [float(x) for x in line.split()]
                        self.weight_vectors[index][:] = vector
            else:
                raise FileNotFoundError('Failed to initialize weights: {} not found'.format(file_path))

    # 初始化邻域（在构造函数时即调用）
    def __initialize_neighborhood(self) -> None:
        distance = numpy.zeros((len(self.weight_vectors), len(self.weight_vectors)))  # 初始化距离矩阵

        for i in range(len(self.weight_vectors)):
            for j in range(len(self.weight_vectors)):
                distance[i][j] = numpy.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])  # 计算权重间的欧氏距离

            indexes = numpy.argsort(distance[i, :])  # 根据欧氏距离从小到大排序
            self.neighborhood[i, :] = indexes[0:self.neighborhood_size]  # 找到各自的邻域，即欧氏距离最小的一系列权重向量的index

    # 获取指定index解的邻域解集（一个index对应一个解，也对应着一个权重向量）
    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        neighbors_indexes = self.neighborhood[index]  # 获取指定index的邻居的index集合

        if any(i > len(solution_list) for i in neighbors_indexes):
            raise IndexError('Neighbor index out of range')

        return [solution_list[i] for i in neighbors_indexes]  # 通过index获取对应的solution，即返回邻居解集

    # 调取总的邻居index集合
    def get_neighborhood(self):
        return self.neighborhood


class TwoDimensionalMesh(Neighborhood):
    """
    Class defining a bi-mensional mesh.
    """

    def __init__(self, rows: int, columns: int, neighborhood: [[]]):
        self.rows = rows
        self.columns = columns
        self.neighborhood = neighborhood
        self.mesh = None
        self.__create_mesh()

    def __create_mesh(self):
        """ Example:
        if rows = 5, and columns=3, we need to fill the mesh as follows
        ----------
        |00-01-02|
        |03-04-05|
        |06-07-08|
        |09-10-11|
        |12-13-14|
        ----------
        """
        self.mesh = numpy.zeros((self.rows, self.columns), dtype=int)
        next_value = 0
        for i in range(self.rows):
            for j in range(self.columns):
                self.mesh[i][j] = next_value
                next_value += 1

    def __get_row(self, index: int) -> int:
        """
        Returns the row in the mesh where the index is local
        :param index:
        :return:
        """
        return index // self.columns

    def __get_column(self, index: int) -> int:
        """
        Returns the column in the mesh where the index is local
        :param index:
        :return:
        """
        return index % self.columns

    def __get_neighbor(self, index: int, neighbor: []) -> int:
        """
        Returns the neighbor of the index
        :param index:
        :param neighbor:
        :return:
        """

        row = self.__get_row(index)

        r = (row + neighbor[0]) % self.rows
        if r < 0:
            r = self.rows - 1

        column = self.__get_column(index)
        c = (column + neighbor[1]) % self.columns
        if c < 0:
            c = self.columns - 1

        return self.mesh[r][c]

    def __find_neighbors(self, solution_list: [], solution_index: int, neighborhood: [[]]):
        """
        Returns a list containing the neighbors of a given solution belongin to a solution list
        :param solution_list:
        :param solution_index:
        :param neighborhood:
        :return:
        """
        neighbors = []

        for neighbor in neighborhood:
            index = self.__get_neighbor(solution_index, neighbor=neighbor)
            neighbors.append(solution_list[index])

        return neighbors

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        Check.is_not_none(solution_list)
        Check.that(len(solution_list) != 0, "The list of solutions is empty")

        return self.__find_neighbors(solution_list, index, self.neighborhood)


class C9(TwoDimensionalMesh):
    """
    Class defining an C9 neighborhood of a solution belonging to a list of solutions which is
    structured as a bi-dimensional mesh. The neighbors are those solutions that are in 1-hop distance

   Shape:
           * * *
           * o *
           * * *

   Topology:
            north      = {-1,  0}
            south      = { 1 , 0}
            east       = { 0 , 1}
            west       = { 0 ,-1}
            north_east = {-1,  1}
            north_west = {-1, -1}
            south_east = { 1 , 1}
            south_west = { 1 ,-1}
    """

    def __init__(self, rows: int, columns: int):
        super(C9, self).__init__(rows, columns, [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1]])


class L5(TwoDimensionalMesh):
    """
    L5 neighborhood.
    Shape:
            *
          * o *
            *

    Topology:
        north = -1,  0
        south =  1,  0
        east  =  0,  1
        west  =  0, -1
    """

    def __init__(self, rows: int, columns: int):
        super(L5, self).__init__(rows, columns, [[-1, 0], [1, 0], [0, 1], [0, -1]])
