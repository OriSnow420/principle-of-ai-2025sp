"""Implement Model Classes for Cross-Line Problem"""

from typing import List, Optional, Tuple, Callable

from node import INode
from problem import IProblem
from state import IState


class CrossLineState(IState):
    """The implementation for State interface

    Args:
        _values (List[List[int]]): The current state. Different numbers stands for
            different colors, and 0 stands for empty block.

    """

    def __init__(self, state: List[List[int]] = List()):
        """Initialization

        Args:
            _values (List[List[int]]): The 2-dimensional list of int to represent the state
        """
        self._values: List[List[int]] = state

    def __hash__(self):
        tuple_self = ()
        for rows in self._values:
            tuple_self += tuple(rows)
        return hash(tuple_self)

    def __eq__(self, value):
        if isinstance(value, CrossLineState):
            return self._values == value._values
        return False


class CrossLineNode(INode):
    """The implementation for Node interface

    Args:
        _state (CrossLineState):
    """

    def __init__(
        self, state: CrossLineState, cost: int, parent: Optional["CrossLineNode"]
    ):
        self._state: CrossLineState = state
        self._parent: Optional["CrossLineNode"] = parent
        self._cost: int = cost

    def get_state(self) -> CrossLineState:
        """Get the corresponding state

        Returns:
            IState: The state
        """
        return self._state

    def get_cost(self) -> int:
        """Get the cost

        Returns:
            int: The cost
        """
        return self._cost

    def get_parent(self) -> Optional["CrossLineNode"]:
        """Returns the parent on the searching tree

        Returns:
            Optional[INode]: The parent if exists else Node.
        """
        return self._parent


class CrossLineProblem(IProblem):
    """The implementation for Problem interface

    Args:
        _row (int): Count of rows
        _col (int): Count of columns
        _init_colors (List[Tuple[int, int, int, int]]): Init color positions;
            each tuple means a pair of positions: (x1, y1, x2, y2).
        _h_function (Callable[[CrossLineState], int]): Value function, accepting a
            'CrossLineState' and returning int. Defaults to 0 function.
        _turning_penalty (bool): If the problem has the cost of turning direction.
            Defaults to False.
    """

    def __init__(
        self,
        row: int,
        col: int,
        init_colors: List[Tuple[int, int, int, int]],
        h_function: Callable[[CrossLineState], int] = lambda _: 0,
        turning_penalty: bool = False,
    ):
        self._row: int = row
        self._col: int = col
        self._init_colors: List[Tuple[int, int, int, int]] = init_colors
        self._h_function: Callable[[CrossLineState], int] = h_function
        self._turning_penalty = turning_penalty

    def init_state(self) -> CrossLineState:
        """Returns the initial state"""
        result_arr = [[0] * self._col] * self._row
        for i, value in enumerate(self._init_colors):
            result_arr[value[0]][value[1]] = i + 1
            result_arr[value[2]][value[3]] = i + 1
        return CrossLineState(result_arr)

    def is_goal_state(self, state: CrossLineState) -> bool:
        """Tells if the state is goal

        Args:
            state (IState): The state to be judged

        Returns:
            bool: True if state is the goal state
        """
        raise NotImplementedError

    def actions(self, node: CrossLineNode) -> List[CrossLineNode]:
        """Returns available child nodes with costs

        Args:
            node (INode): Current node

        Returns:
            List[INode]: Next step nodes
        """
        raise NotImplementedError
