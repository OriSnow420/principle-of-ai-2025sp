"""Contains code for pa3"""
import time
import tkinter as tk
from tkinter import PhotoImage
from dataclasses import dataclass
from typing import Dict, Literal, List, Tuple
import random
import json

import numpy as np

seed = random.randint(0, 2147483647)
random.seed(seed)

UNIT = 100  # 迷宫中每个格子的像素大小
MAZE_H = 5  # 迷宫的高度（格子数）
MAZE_W = 5  # 迷宫的宽度（格子数）
INIT_POS = [0, 0]  # 哈利波特的起始位置
GOAL_POS = [2, 2]  # 火焰杯的位置
TRAP_POS = [[1, 2], [2, 1], [1, 3]]  # 陷阱的位置

SLEEP = True

@dataclass
class State():
    """Represent a state - The place of Potter"""
    x: int
    y: int

    def __init__(self, s):
        self.x = int((s[0] - 50.0) // 100)
        self.y = int((s[1] - 50.0) // 100)

    def __hash__(self):
        return hash((self.x, self.y))

    def is_goal(self):
        """Tells if the state is goal"""
        return self.x == GOAL_POS[0] and self.y == GOAL_POS[1]

    def is_trap(self):
        """Tells if the state is trap"""
        for trap in TRAP_POS:
            if self.x == trap[0] and self.y == trap[1]:
                return True
        return False

    def __str__(self):
        return f"({self.x},{self.y})"

Action = Literal[0, 1, 2, 3] # 0 - U, 1 - D, 2 - R, 3 - L.
Q_table: Dict[State, List[Tuple[Action, float]]] = {}
ALPHA = 0.2 # Learning-Rate
GAMMA = 0.01 # Discount Factor
EPSILON = 0.01
path_lens = []

class Maze(tk.Tk):
    """The maze class"""
    def __init__(self):
        super().__init__()
        # Note that "l" is actually Right, vice versa.
        self.action_space = ["u", "d", "l", "r"]  # 决策空间
        self.n_actions = len(self.action_space)
        self.title("Q-learning")
        # self.geometry("{0}x{1}".format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.geometry(f"{MAZE_H * UNIT}x{MAZE_H * UNIT}")
        self._build_maze()

    def _build_maze(self):
        """
        迷宫初始化
        """
        self.canvas = tk.Canvas(
            self, bg="white", height=MAZE_H * UNIT, width=MAZE_W * UNIT
        )

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1, fill="black")
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1, fill="black")

        origin = np.array([UNIT / 2, UNIT / 2])

        self.bm_trap = PhotoImage(file="trap.png")
        self.trap_list = []

        for trap in TRAP_POS:
            self.trap_list.append(
                self.canvas.create_image(
                    origin[0] + UNIT * trap[0],
                    origin[1] + UNIT * trap[1],
                    image=self.bm_trap,
                )
            )

        self.bm_potter = PhotoImage(file="potter.png")
        self.potter = self.canvas.create_image(
            origin[0] + INIT_POS[0] * UNIT,
            origin[1] + INIT_POS[1] * UNIT,
            image=self.bm_potter,
        )

        self.bm_goal = PhotoImage(file="cup.png")
        self.goal = self.canvas.create_image(
            origin[0] + GOAL_POS[0] * UNIT,
            origin[1] + GOAL_POS[1] * UNIT,
            image=self.bm_goal,
        )

        self.canvas.pack()

    def reset(self):
        """Reset"""
        self.update()  # tk.Tk().update 强制画面更新
        if SLEEP:
            time.sleep(0.5)
        self.canvas.delete(self.potter)
        origin = np.array([UNIT / 2, UNIT / 2])

        self.potter = self.canvas.create_image( # pylint: disable=attribute-defined-outside-init
            origin[0], origin[1], image=self.bm_potter
        )
        # 返回当前potter所在的位置
        return self.canvas.coords(self.potter)

    def step(self, action):
        """Do the step"""
        s = self.canvas.coords(self.potter)
        base_action = np.array([0, 0])
        if action == 0:  # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.potter, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.potter)

        # 回报函数
        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
            # s_ = "terminal"
        elif s_ in [self.canvas.coords(a_trap) for a_trap in self.trap_list]:
            reward = -1
            done = True
            # s_ = "terminal"
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        """Render"""
        if SLEEP:
            time.sleep(0.1)
        self.update()

def available_actions(state: State) -> List[Tuple[Action, float]]:
    """Returns an available action list for initializing

    Args:
        state (State): The state

    Returns:
        List[Tuple[Action, float]]: The available (action, value) pair list
    """
    value = 0.0
    if state.is_goal():
        value = 1.0
    elif state.is_trap():
        value = -1.0
    result = []
    if state.x > 0:
        result.append((3, value))
    if state.x < MAZE_W - 1:
        result.append((2, value))
    if state.y > 0:
        result.append((0, value))
    if state.y < MAZE_H - 1:
        result.append((1, value))
    return result

def policy(state: State) -> Action:
    """Decide the next move, which has the maximum Q_value

    Args:
        state (State): the current state

    Returns:
        Action: The action to move
    """
    if state not in Q_table:
        Q_table[state] = available_actions(state)
    if random.randrange(1000) < EPSILON * 1000:
        return random.choice(Q_table[state])[0]
    max_value = -100 # To Represent -Infinity
    action_list: List[Action] = []
    for action, q_value in Q_table[state]:
        if q_value > max_value:
            max_value = q_value
            action_list.clear()
            action_list.append(action)
        elif q_value == max_value:
            action_list.append(action)
    return random.choice(action_list)

def query_q_value_state(state: State) -> float:
    """Query for a state's Q_value, which is the highest Q_value among all (state, action) pair

    Args:
        state (State): the current state

    Returns:
        float: The value
    """
    if state not in Q_table:
        Q_table[state] = available_actions(state)
    max_value = -100 # To Represent -Infinity
    for _, q_value in Q_table[state]:
        max_value = max(max_value, q_value)
    return max_value

def query_q_value_action(state: State, action: Action) -> float:
    """Query for a (state, action)'s Q_value

    Args:
        state (State): The current state
        action (Action): The action

    Returns:
        float: The value
    """
    if state not in Q_table:
        Q_table[state] = available_actions(state)
    for act, q_value in Q_table[state]:
        if act == action:
            return q_value
    return 0.0 # Unreachable

def update_q_table_with_sample(state: State, action: Action, sample: float):
    """Update Q_table with given sample

    Args:
        state (State): The current state
        action (Action): The action
        sample (float): The sample
    """
    if state not in Q_table:
        Q_table[state] = available_actions(state)
    current_value = query_q_value_action(state, action)
    new_value = (1 - ALPHA) * current_value + ALPHA * sample
    for i, act_value in enumerate(Q_table[state]):
        if action == act_value[0]:
            Q_table[state][i] = (action, new_value)

def update():
    """Update the maze"""
    # 更新图形化界面
    for t in range(300):
        s = env.reset() # pylint: disable=possibly-used-before-assignment
        path_len = 0

        while True:
            env.render()
            current_state = State(s)

            # Decision-Making
            step = policy(current_state)
            path_len += 1

            s, r, done = env.step(step)
            next_state = State(s)

            # Learning
            sample = r + GAMMA * query_q_value_state(next_state)
            update_q_table_with_sample(current_state, step, sample)

            if done:
                print(f"Try {t + 1}: Got final reward {r}, length of path {path_len}.")
                path_lens.append(path_len)
                break

    serializable_q_table = {}
    for key, value in Q_table.items():
        serializable_q_table[str(key)] = value
    with open("q_table.json", 'w', encoding="utf-8") as f:
        json.dump(serializable_q_table, f, indent=4)
    with open("path_len.json", 'w', encoding="utf-8") as f:
        json.dump(path_lens, f, indent=4)
    print(f"Random Seed: {seed}")

if __name__ == "__main__":
    env = Maze()
    env.after(100, update)
    env.mainloop()
