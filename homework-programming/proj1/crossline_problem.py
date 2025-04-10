"""Implement Model Classes for Cross-Line Problem"""

from copy import deepcopy
from typing import List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass

from node import INode
from problem import IProblem
from state import IState


class Direction(Enum):
    """Used to represent 2D-direction"""

    LEFT = 1
    RIGHT = 2
    DOWN = 3
    UP = 4


@dataclass
class Slot:
    """Represents a slot's attributes

    Args:
        color (int): The slot's color
        direction (Optional[Direction]): The direction on the line, used to judge turning
    """

    color: int
    direction: Optional[Direction] = None
    movable_id: int = 0

    def __hash__(self):
        return hash((self.color, self.direction))

    def __repr__(self):
        return str(self.color)


class CrossLineState(IState):
    """The implementation for State interface

    Args:
        _values (List[List[int]]): The current state. Different numbers stands for
            different colors, and 0 stands for empty block.

    """

    def __init__(self, state: List[List[Slot]], movable_points: List[Tuple[int, int]]):
        """Initialization

        Args:
            _values (List[List[int]]): The 2-dimensional list of int to represent the state
            movable_points: List[Tuple[int, int]]: The points that can go forward
        """
        self.values: List[List[Slot]] = state
        self.movable_points: List[Tuple[int, int]] = movable_points

    def get_color(self, row: int, col: int) -> int:
        """Get the color of a position

        Args:
            row (int): The row coordinate
            col (int): The col coordinate

        Returns:
            int: The color
        """
        return self.values[row][col].color

    def modify_color(self, row: int, col: int, new_color: int) -> None:
        """Modify color of a position

        Args:
            row (int): The row coordinate
            col (int): The col coordinate
            new_color (int): The new color
        """
        self.values[row][col].color = new_color

    def get_direction(self, row: int, col: int) -> Direction:
        """Get the direction of a position

        Args:
            row (int): The row coordinate
            col (int): The column coordinate

        Returns:
            Direction: The direction
        """
        return self.values[row][col].direction

    def __hash__(self):
        tuple_self = ()
        for rows in self.values:
            tuple_self += tuple(rows)
        return hash(tuple_self)

    def __eq__(self, value):
        if isinstance(value, CrossLineState):
            return self.values == value.values
        return False


class CrossLineNode(INode):
    """The implementation for Node interface

    Args:
        _state (CrossLineState):
    """

    def __init__(
        self,
        state: CrossLineState,
        cost: int,
        parent: Optional["CrossLineNode"],
        h_function: Callable[[CrossLineState], int],
    ):
        self._state: CrossLineState = state
        self._parent: Optional["CrossLineNode"] = parent
        self._cost: int = cost
        self._h_function: Callable[[CrossLineState], int] = h_function

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
        result_arr = [[Slot(0) for _ in range(self._col)] for _ in range(self._row)]
        movables = []
        movable_id = 1
        for i, value in enumerate(self._init_colors):
            result_arr[value[0]][value[1]].color = i + 1
            result_arr[value[2]][value[3]].color = i + 1
            result_arr[value[0]][value[1]].movable_id = movable_id
            movable_id += 1
            result_arr[value[2]][value[3]].movable_id = movable_id
            movable_id += 1
            movables.append((value[0], value[1]))
            movables.append((value[2], value[3]))
        return CrossLineState(result_arr, movables)

    def init_node(self) -> CrossLineNode:
        """Returns the initial node

        Returns:
            CrossLineNode: The initial node
        """
        state = self.init_state()
        return CrossLineNode(
            state, len(state.movable_points) // 2, None, self._h_function
        )

    def perform_action(
        self, state: CrossLineState, action: Tuple[int, int, int, Direction]
    ) -> CrossLineState | None:
        """Calculates the new state according to the action

        Args:
            state (CrossLineState): Current state
            action (Tuple[int, int, int]): The action

        Returns:
            CrossLineState | None: The new state
        """
        result = deepcopy(state)
        row, col, color, direction = action
        move_row, move_col = row, col

        if direction == Direction.UP:
            move_row -= 1
        elif direction == Direction.DOWN:
            move_row += 1
        elif direction == Direction.LEFT:
            move_col -= 1
        else:  # direction == Direction.RIGHT
            move_col += 1

        # Out of bound!
        if not -1 < move_row < self._row:
            return None
        if not -1 < move_col < self._col:
            return None

        if (
            state.get_color(row, col) == state.get_color(move_row, move_col)
            and (move_row, move_col) in result.movable_points
        ):
            # Connected Lines! Hooray! Then remove the pair from movable points
            result.movable_points.remove((row, col))
            result.movable_points.remove((move_row, move_col))
            result.values[row][col].movable_id = 0
            result.values[move_row][move_col].movable_id = 0
            return result
        if state.get_color(move_row, move_col) != 0:
            # Occupied Slots! Give Up
            return None

        result.modify_color(move_row, move_col, color)
        result.values[move_row][move_col].direction = direction
        result.movable_points.remove((row, col))
        result.movable_points.append((move_row, move_col))
        result.values[move_row][move_col].movable_id = result.values[row][
            col
        ].movable_id
        result.values[row][col].movable_id = 0

        return result

    def is_goal_state(self, state: CrossLineState) -> bool:
        """Tells if the state is goal

        Args:
            state (CrossLineState): The state to be judged

        Returns:
            bool: True if state is the goal state
        """
        return state.movable_points == []

    def actions(self, node: CrossLineNode) -> List[CrossLineNode]:
        """Returns available child nodes with costs

        Args:
            node (CrossLineNode): Current node

        Returns:
            List[CrossLineNode]: Next step nodes
        """
        state = node.get_state()
        result: List[CrossLineNode] = []
        # (Row_coord, Col_coord, Color_num, direction num)
        # Dir_num: 0-Left 1-Right 2-Down 3-Up
        actions_candidate: List[Tuple[int, int, int, Direction]] = []
        for movable in node.get_state().movable_points:
            color_num = state.values[movable[0]][movable[1]].color
            actions_candidate.append(
                (movable[0], movable[1], color_num, Direction.LEFT)
            )
            actions_candidate.append(
                (movable[0], movable[1], color_num, Direction.RIGHT)
            )
            actions_candidate.append(
                (movable[0], movable[1], color_num, Direction.DOWN)
            )
            actions_candidate.append((movable[0], movable[1], color_num, Direction.UP))

        for action in actions_candidate:
            row, col, _, direction = action
            move_row, move_col = row, col

            if direction == Direction.UP:
                move_row -= 1
            elif direction == Direction.DOWN:
                move_row += 1
            elif direction == Direction.LEFT:
                move_col -= 1
            else:  # direction == Direction.RIGHT
                move_col += 1

            new_state = self.perform_action(state, action)
            if new_state is None:
                continue
            result.append(
                CrossLineNode(
                    new_state,
                    (  # Here actually is the G function
                        node.get_cost()
                        + (
                            1
                            if not self._turning_penalty
                            else (
                                3
                                if new_state.values[row][col].direction
                                != new_state.values[move_row][move_col].direction
                                and new_state.values[row][col].direction is not None
                                else 1
                            )
                        )
                    ),
                    node,
                    self._h_function,
                )
            )
        return result
