import numpy as np
from copy import deepcopy
from utils import *
import time


def h_function_null(state, goal_state):
    return 0


class Node(object):  # Represents a node in a search tree
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def child_node(self, problem, action):
        next_state = problem.move(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.g(self.depth, self.state, action, next_state) +
                         problem.h(next_state, problem.goal_state.state),)
        return next_node

    def path(self):
        """
        Returns list of nodes from this node to the root node
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return f"#########\n#{self.state[0][0]} {self.state[0][1]} {self.state[0][2]} {self.state[0][3]}#\n#{self.state[1][0]} {self.state[1][1]} {self.state[1][2]} {self.state[1][3]}#\n#{self.state[2][0]} {self.state[2][1]} {self.state[2][2]} {self.state[2][3]}#\n#{self.state[3][0]} {self.state[3][1]} {self.state[3][2]} {self.state[3][3]}#\n#########"


    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


class Problem(object):
    def __init__(self, init_state=None, goal_state=None, h_function=None):
        self.init_state = Node(init_state)
        self.goal_state = Node(goal_state)
        self.h = h_function

    def actions(self, state):
        """
        Given the current state, return valid actions.
        :param state:
        :return: valid actions
        """
        pass

    def move(self, state, action):
        pass

    def is_goal(self, state):
        pass

    def g(self, cost, from_state, action, to_state):
        return cost + 1

    def solution(self, goal):
        """
        Returns actions from this node to the root node
        """
        if goal.state is None:
            return None
        return [node.action for node in goal.path()[1:]]

    def expand(self, node):  # Returns a list of child nodes
        return [node.child_node(self, action) for action in self.actions(node.state)]
    

class GridsProblem(Problem):
    def __init__(self,
                 n,
                 init_state=[[1, 3, 1, 1], 
                             [1, 1, 0, 1], 
                             [1, 2, 1, 1], 
                             [4, 1, 5, 6]],
                 goal_state=[[1, 1, 1, 1], 
                             [2, 1, 1, 1], 
                             [3, 5, 1, 1], 
                             [4, 6, 1, 0]],
                 h_function=h_function_null):
        super().__init__(init_state, goal_state, h_function)
        self.n = n

    def is_valid(self, loc):
        return -1 < loc[0] < self.n and -1 < loc[1] < self.n

    def actions(self, state):
        empty_row, empty_col = np.where(np.array(state) == 0)[0][0], np.where(np.array(state) == 0)[1][0]
        candidates = [[empty_row-1, empty_col], [empty_row+1, empty_col],
                      [empty_row, empty_col-1], [empty_row, empty_col+1],]
        valid_candidates = [item for item in candidates if self.is_valid(item)]
        return valid_candidates

    def move(self, state, action):
        empty_row, empty_col = np.where(np.array(state) == 0)[0][0], np.where(np.array(state) == 0)[1][0]
        new_state = deepcopy(state)
        new_state[empty_row][empty_col] = state[action[0]][action[1]]
        new_state[action[0]][action[1]] = 0
        return new_state

    def is_goal(self, state):
        return state == self.goal_state.state

    def g(self, cost, from_state, action, to_state):
        return cost + 1
    
    
def search_with_info(problem: GridsProblem):
    print("有信息搜索。")

def search_without_info(problem: GridsProblem):
    print("无信息搜索")



if __name__ == "__main__":
    # 无信息搜索
    problem = GridsProblem(4, h_function=h_function_null)
    search_without_info(problem)

    # A*搜索算法
    problem = GridsProblem(4, h_function=h_function_null)
    search_with_info(problem)

    # problem = GridsProblem(4, h_function=h_function_metohd2)
    # search_with_info(problem)

    # problem = GridsProblem(4, h_function=h_function_method3)
    # search_with_info(problem)
