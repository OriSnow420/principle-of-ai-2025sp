from collections import deque
import time
import sortedcontainers


class Queue(object):
    def __init__(self):
        self._items = deque([])

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.popleft() if not self.empty() else None

    def empty(self):
        return len(self._items) == 0

    def find(self, item):
        return self._items.index(item) if item in self._items else None

    def top(self):
        return self._items[0]

    def include(self, item):
        return item in self._items


class Stack(object):
    def __init__(self):
        self._items = list()

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop() if not self.empty() else None

    def empty(self):
        return len(self) == 0

    def include(self, item):
        return item in self._items

    def __len__(self):
        return len(self._items)


class PriorityQueue(object):

    def __init__(self, node):
        self._queue = sortedcontainers.SortedList([node])

    def push(self, node):
        self._queue.add(node)

    def pop(self):
        return self._queue.pop(index=0)

    def empty(self):
        return len(self._queue) == 0

    def compare_and_replace(self, i, node):
        if node < self._queue[i]:
            self._queue.pop(index=i)
            self._queue.add(node)

    def find(self, node):
        try:
            loc = self._queue.index(node)
            return loc
        except ValueError:
            return None

    def include(self, node):
        return self.find(node) != None


class Set(object):
    def __init__(self, ip_set=None):
        self._items = set()
        if ip_set is not None:
            self._items = ip_set

    def add(self, item):
        self._items.add(item)

    def remove(self, item):
        self._items.remove(item)

    def pop(self):
        return self._items.pop()

    def include(self, item):
        return item in self._items

    def find_by_state(self, item):
        for a_node in self._items:
            if a_node.state == item:
                return a_node
        print("cannot find")
        return None

    def intersection(self, item):
        return Set(self._items.intersection(item._items))

    def empty(self):
        return not self._items


class Dict(object):
    def __init__(self):
        self._items = dict()

    def add(self, key, value):
        self._items.update({key: value})

    def remove(self, key):
        self._items.pop(key, None)

    def find(self, key):
        return self._items[key] if key in self._items else None


class Stopwatch:
    def __init__(self):
        self.begin = time.time()

    def elapsed_time(self):
        return time.time() - self.begin
