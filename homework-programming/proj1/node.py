"""Contains class Node interface"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
from state import IState


class INode(ABC):
    """Node Interface, which is a state with cost on a searching tree."""

    _h_function: Callable[[IState], int]

    @abstractmethod
    def get_state(self) -> IState:
        """Get the corresponding state

        Returns:
            IState: The state
        """

    @abstractmethod
    def get_cost(self) -> int:
        """Get the cost

        Returns:
            int: The cost
        """

    @abstractmethod
    def get_parent(self) -> Optional["INode"]:
        """Returns the parent on the searching tree

        Returns:
            Optional[INode]: The parent if exists else Node.
        """

    def __hash__(self):
        return self.get_state().__hash__()

    def __eq__(self, value):
        if isinstance(value, INode):
            return self.get_state() == value.get_state()
        raise TypeError("Equals to only compares between INodes")

    def __lt__(self, other):
        if isinstance(other, INode):
            return self.get_cost() + self._h_function(
                self.get_state()
            ) < other.get_cost() + other._h_function(other.get_state())
        raise TypeError("Less than only compares between INodes")
