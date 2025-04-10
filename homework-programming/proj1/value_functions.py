"""Defines Value functions"""

from crossline_problem import CrossLineState


def value_function_null(_: CrossLineState) -> int:
    """Null Value Function

    Args:
        _ (CrossLineState): Not used

    Returns:
        int: 0
    """
    return 0


def value_function_manhattan_distance(state: CrossLineState) -> int:
    """A value function, calculating the sum of the manhattan distance between
    unpaired movable points

    Args:
        state (CrossLineState): The state

    Returns:
        int: Value function result
    """
    result = 0
    for row, col in state.movable_points:
        color = state.get_color(row, col)
        for row_2, col_2 in state.movable_points:
            if row == row_2 and col == col_2:
                continue
            if color == state.get_color(row_2, col_2):
                result += abs(row_2 - row) + abs(col_2 - col)
                break
    return result


def value_function_manhattan_with_turning_penalty(state: CrossLineState) -> int:
    """A value function, calculating the sum of the manhattan distance between
    unpaired movable points. Besides, if the pair is not on a row or on a column,
    an additional `2` penalty is added into result.

    Args:
        state (CrossLineState): The state

    Returns:
        int: Value function result
    """
    result = 0
    for row, col in state.movable_points:
        color = state.get_color(row, col)
        for row_2, col_2 in state.movable_points:
            if row == row_2 and col == col_2:
                continue
            if color == state.get_color(row_2, col_2):
                result += abs(row_2 - row) + abs(col_2 - col)
                if row_2 != row and col_2 != col:
                    result += 2
                break
    return result
