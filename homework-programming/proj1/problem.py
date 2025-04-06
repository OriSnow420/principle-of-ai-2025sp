"""Contains Problem interface"""

from abc import ABC, abstractmethod
from typing import List

from state import IState
from node import INode


class IProblem(ABC):
    """Problem Abstract Class"""

    @abstractmethod
    def init_state(self) -> IState:
        """Returns the initial state"""

    @abstractmethod
    def is_goal_state(self, state: IState) -> bool:
        """Tells if the state is goal

        Args:
            state (IState): The state to be judged

        Returns:
            bool: True if state is the goal state
        """

    @abstractmethod
    def actions(self, node: INode) -> List[INode]:
        """Returns available child nodes with costs

        Args:
            node (INode): Current node

        Returns:
            List[INode]: Next step nodes
        """
