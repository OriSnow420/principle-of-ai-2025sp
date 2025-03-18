"""Contains main class and functions for problem."""

from copy import deepcopy
from typing import Tuple, Optional, List
import numpy as np
import utils
import time

def h_function_null(state: List[List[int]], goal_state: List[List[int]]):
    """Null h function

    Args:
        state (List[List[int]]): The current state
        goal_state (List[List[int]]): The goal state

    Returns:
        int: 0
    """
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

    def child_node(self, the_problem, action):
        next_state = the_problem.move(self.state, action)
        next_node = Node(
            next_state,
            self,
            action,
            the_problem.g(self.depth, self.state, action, next_state)
            + the_problem.h(next_state, the_problem.goal_state.state),
        )
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
        return f"\n#########\n#{self.state[0][0]} {self.state[0][1]} {self.state[0][2]} {self.state[0][3]}#\n#{self.state[1][0]} {self.state[1][1]} {self.state[1][2]} {self.state[1][3]}#\n#{self.state[2][0]} {self.state[2][1]} {self.state[2][2]} {self.state[2][3]}#\n#{self.state[3][0]} {self.state[3][1]} {self.state[3][2]} {self.state[3][3]}#\n#########"

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        tupled_state = ()
        for rows in self.state:
            tupled_state += tuple(rows)
        return hash(tupled_state)


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
    def __init__(
        self,
        n,
        init_state=[[1, 3, 1, 1], [1, 1, 0, 1], [1, 2, 1, 1], [4, 1, 5, 6]],
        goal_state=[[1, 1, 1, 1], [2, 1, 1, 1], [3, 5, 1, 1], [4, 6, 1, 0]],
        h_function=h_function_null,
    ):
        super().__init__(init_state, goal_state, h_function)
        self.n = n

    def get_pos(self, state, num) -> Tuple[int, int]:
        """Find the position of an element in a state.

        Args:
            state (List[List[int]]): The state
            num (int): the element to be found

        Returns:
            Tuple[int, int]: The (x, y) Position
        """
        return (
            np.where(np.array(state) == num)[0][0],
            np.where(np.array(state) == num)[1][0],
        )

    def actions(self, state):
        empty_row, empty_col = self.get_pos(state, 0)
        aobing_row, aobing_col = self.get_pos(state, 5)

        candidates = [
            [empty_row + 1, empty_col],
            [empty_row - 1, empty_col],
            [empty_row, empty_col + 1],
            [empty_row, empty_col - 1],
            [aobing_row, aobing_col],
        ]

        # Deduplication
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        # Not out of board
        for candidate in unique_candidates.copy():
            if (
                candidate[0] < 0
                or candidate[0] >= self.n
                or candidate[1] < 0
                or candidate[1] >= self.n
            ):
                unique_candidates.remove(candidate)

        # Nezha position valid
        for candidate in unique_candidates.copy():
            check = False
            move_head = False
            if state[candidate[0]][candidate[1]] == 3: # Move Head
                check = True
                move_head = True
            elif state[candidate[0]][candidate[1]] == 4: # Move Body
                check = True
                move_head = False

            if check:
                pos_1_row, pos_1_col = empty_row, empty_col
                pos_2_row, pos_2_col = self.get_pos(state, 4 if move_head else 3)
                if abs(pos_1_col - pos_2_col) + abs(pos_1_row - pos_2_row) > 4:
                    unique_candidates.remove(candidate)

        return unique_candidates

    def move(self, state, action):
        empty_row, empty_col = (
            np.where(np.array(state) == 0)[0][0],
            np.where(np.array(state) == 0)[1][0],
        )
        new_state = deepcopy(state)
        new_state[empty_row][empty_col] = state[action[0]][action[1]]
        new_state[action[0]][action[1]] = 0
        return new_state

    def is_goal(self, state):
        return state == self.goal_state.state

    def g(self, cost, from_state, action, to_state):
        return cost + 1

def report(node: Optional[Node]) -> None:
    if node is None:
        print("Search failed!")
    else:
        print(f"Found path with length {len(node.path())}")

# A star
def search_with_info(problem: GridsProblem):
    count = 0
    node = Node(problem.init_state.state)
    if problem.is_goal(node.state):
        report(node)
        return
    open_list = utils.PriorityQueue(node)
    open_list_set = utils.Set()

    close_list = utils.Set()

    # open_list.push(node)
    open_list_set.add(node)

    while not open_list.empty():
        node: Node = open_list.pop()
        open_list_set.remove(node)
        # if count % 10000 == 0:
        print(f"Visiting Node #{count}. Open list size: {len(open_list._queue)}. Close list size: {len(close_list._items)}{node}")
        count += 1
        close_list.add(node)
        for child in problem.expand(node):
            if not open_list_set.include(child) and not close_list.include(child):
                if problem.is_goal(child.state):
                    report(child)
                    return
                open_list.push(child)
                open_list_set.add(child)
    report(None)

# Breadth-First
def search_without_info(problem: GridsProblem):
    count = 0
    node = Node(problem.init_state.state)
    if problem.is_goal(node.state):
        report(node)
        return
    open_list = utils.Queue()
    open_list_set = utils.Set()

    close_list = utils.Set()

    open_list.push(node)
    open_list_set.add(node)

    while not open_list.empty():
        node: Node = open_list.pop()
        open_list_set.remove(node)
        # if count % 10000 == 0:
        print(f"Visiting Node #{count}. Open list size: {len(open_list._items)}. Close list size: {len(close_list._items)}{node}")
        count += 1
        close_list.add(node)
        for child in problem.expand(node):
            if not open_list_set.include(child) and not close_list.include(child):
                if problem.is_goal(child.state):
                    report(child)
                    return
                open_list.push(child)
                open_list_set.add(child)
    report(None)


def h_function_method2(state: List[List[int]], goal_state: List[List[int]]):
    """An useful h function

    Args:
        state (List[List[int]]): The current state
        goal_state (List[List[int]]): The goal state

    Returns:
        int: 16 - #Numbers with correct position
    """
    result = 16
    for i, rows in enumerate(state):
        for j, element in enumerate(rows):
            if element == goal_state[i][j]:
                result -= 1
    return result

def h_function_method3(state: List[List[int]], goal_state: List[List[int]]):
    """More h function

    Args:
        state (List[List[int]]): The current state
        goal_state (List[List[int]]): The goal state

    Returns:
        int: ?
    """
    result = 0
    for i, rows in enumerate(state):
        for j, element in enumerate(rows):
            if element != goal_state[i][j] and element != 5:
                goal_pos_x, goal_pos_y = (
                    np.where(np.array(goal_state) == element)[0][0],
                    np.where(np.array(goal_state) == element)[1][0],
                )
                result += abs(goal_pos_x - i) + abs(goal_pos_y - j)
    return result / 1.095 # Best: [1.090909091, 1.1)

if __name__ == "__main__":

    sw = utils.Stopwatch()

    # 无信息搜索
    # problem = GridsProblem(4)
    # search_without_info(problem)

    # A*搜索算法
    # problem = GridsProblem(4, h_function=h_function_null)
    # search_with_info(problem)

    # problem = GridsProblem(4, h_function=h_function_method2)
    # search_with_info(problem)

    problem = GridsProblem(4, h_function=h_function_method3)
    search_with_info(problem)

    print(f"Search takes {sw.elapsed_time()} seconds!")
