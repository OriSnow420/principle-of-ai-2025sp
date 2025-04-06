"""Contains State Interface"""

from abc import ABC, abstractmethod


class IState(ABC):
    """A hashable, comparable(equal only) abstract state"""

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, value):
        pass
