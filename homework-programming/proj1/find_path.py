"""Contains find path functions"""

from typing import Optional
from utils import PriorityQueue, Set
from problem import IProblem
from node import INode


def find_path_a_star(problem: IProblem) -> Optional[INode]:
    """A star algorithm for finding path

    Args:
        problem (IProblem): The problem

    Returns:
        Optional[INode]: The result node; None if no path found
    """
    node = problem.init_node()
    open_list = PriorityQueue(node)
    closed_list_set = Set()

    while not open_list.empty():
        node: INode = open_list.pop()
        # For debug(Will be printed in the executable):
        # print(
        #     f"Current state: {node.get_state().values}, {node.get_state().movable_points}"
        # )
        if problem.is_goal_state(node.get_state()):
            return node
        closed_list_set.add(node.get_state())

        for child in problem.actions(node):
            if not closed_list_set.include(child.get_state()) and not open_list.include(
                child
            ):
                open_list.push(child)
            elif open_list.include(child):
                open_list.compare_and_replace(open_list.find(child), child)
    return None
