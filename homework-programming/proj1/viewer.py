"""Contains Viewer class for a GUI software interface"""

import tkinter as tk
from tkinter import ttk
from crossline_problem import CrossLineProblem, Direction, CrossLineState
from find_path import find_path_a_star
from value_functions import (
    value_function_null,
    value_function_manhattan_distance,
    value_function_manhattan_with_turning_penalty,
)


class CrossLineGUI(tk.Tk):
    """The GUI class using Tk"""

    def __init__(self):
        super().__init__()
        self.title("CrossLine Solver")
        self.colors = [
            "white",
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "brown",
            "gray",
        ]
        self.cell_size = 40
        self.solution_path = []
        self.current_step = 0
        self.value_functions = {
            "Null Value Function": value_function_null,
            "Manhattan Distance": value_function_manhattan_distance,
            "Manhattan with Penalty": value_function_manhattan_with_turning_penalty,
        }
        self.create_widgets()

    def create_widgets(self):
        """Basic GUI"""
        # 输入区域
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="行数:").grid(row=0, column=0)
        self.rows_entry = ttk.Entry(input_frame, width=5)
        self.rows_entry.grid(row=0, column=1)

        ttk.Label(input_frame, text="列数:").grid(row=0, column=2)
        self.cols_entry = ttk.Entry(input_frame, width=5)
        self.cols_entry.grid(row=0, column=3)

        ttk.Label(input_frame, text="颜色数:").grid(row=0, column=4)
        self.color_count_entry = ttk.Entry(input_frame, width=5)
        self.color_count_entry.grid(row=0, column=5)

        ttk.Button(
            input_frame, text="生成输入", command=self.generate_color_inputs
        ).grid(row=0, column=6)

        # 价值函数选择
        ttk.Label(input_frame, text="价值函数:").grid(row=1, column=0)
        self.value_function_var = tk.StringVar()
        self.value_function_var.set("Manhattan Distance")
        value_function_menu = ttk.Combobox(
            input_frame,
            textvariable=self.value_function_var,
            values=list(self.value_functions.keys()),
        )
        value_function_menu.grid(row=1, column=1)

        self.turning_penalty_var = tk.IntVar()
        turning_penalty_checkbox = ttk.Checkbutton(
            input_frame, text="开启转弯惩罚", variable=self.turning_penalty_var
        )
        turning_penalty_checkbox.grid(row=1, column=2)

        self.color_inputs_frame = ttk.Frame(self)
        self.color_inputs_frame.pack(pady=10)

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10)

        ttk.Button(control_frame, text="开始求解", command=self.start_solving).pack(
            side=tk.LEFT
        )
        ttk.Button(control_frame, text="重置", command=self.reset).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, bg="white", width=600, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.path_length_label = ttk.Label(self, text="路径长度: ")
        self.path_length_label.pack(pady=10)

    def generate_color_inputs(self):
        """Labels for colors"""
        for widget in self.color_inputs_frame.winfo_children():
            widget.destroy()

        try:
            color_count = int(self.color_count_entry.get())
        except ValueError:
            return

        for i in range(color_count):
            color = self.colors[(i + 1) % len(self.colors)]
            frame = ttk.Frame(self.color_inputs_frame)
            frame.pack(pady=5)

            ttk.Label(frame, text=f"颜色{i+1} 起点:", foreground=color).pack(
                side=tk.LEFT
            )
            ttk.Entry(frame, width=3).pack(side=tk.LEFT)
            ttk.Label(frame, text="x").pack(side=tk.LEFT)
            ttk.Entry(frame, width=3).pack(side=tk.LEFT)
            ttk.Label(frame, text="y").pack(side=tk.LEFT)

            ttk.Label(frame, text="终点:").pack(side=tk.LEFT)
            ttk.Entry(frame, width=3).pack(side=tk.LEFT)
            ttk.Label(frame, text="x").pack(side=tk.LEFT)
            ttk.Entry(frame, width=3).pack(side=tk.LEFT)
            ttk.Label(frame, text="y").pack(side=tk.LEFT)

    def get_init_colors(self):
        """Input boxes for each colors"""
        init_colors = []

        for frame in self.color_inputs_frame.winfo_children():

            entries = frame.winfo_children()
            try:

                x1 = int(entries[1].get())
                y1 = int(entries[3].get())
                x2 = int(entries[6].get())
                y2 = int(entries[8].get())
                init_colors.append((x1, y1, x2, y2))
            except (ValueError, IndexError, tk.TclError):
                return None
        return init_colors

    def start_solving(self):
        """Solve the problem"""
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
        except ValueError:
            return

        init_colors = self.get_init_colors()
        if not init_colors:
            return

        selected_function_name = self.value_function_var.get()
        selected_function = self.value_functions[selected_function_name]
        turning_penalty = bool(self.turning_penalty_var.get())

        problem = CrossLineProblem(
            rows, cols, init_colors, selected_function, turning_penalty
        )
        result_node = find_path_a_star(problem)

        if result_node:
            self.solution_path = []
            current = result_node
            while current:
                self.solution_path.append(current.get_state())
                current = current.get_parent()
            self.solution_path.reverse()
            self.current_step = 0
            path_length = result_node.get_cost()
            self.path_length_label.config(text=f"路径长度: {path_length}")
            self.animate_solution()
        else:
            self.path_length_label.config(text="未找到路径")

    def animate_solution(self):
        """Show the solution"""
        if self.current_step < len(self.solution_path):
            state = self.solution_path[self.current_step]
            self.draw_state(state)
            self.current_step += 1
            self.after(500, self.animate_solution)

    def draw_state(self, state: CrossLineState):
        """Draw one state

        Args:
            state (CrossLineState): The state
        """
        self.canvas.delete("all")
        rows = len(state.values)
        cols = len(state.values[0]) if rows > 0 else 0

        for row in range(rows):
            for col in range(cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color_idx = state.get_color(row, col)
                color = self.colors[color_idx % len(self.colors)]
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="black"
                )

                direction = state.get_direction(row, col)
                if direction:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if direction == Direction.LEFT:
                        self.canvas.create_line(cx, cy, x1, cy, arrow=tk.LAST)
                    elif direction == Direction.RIGHT:
                        self.canvas.create_line(cx, cy, x2, cy, arrow=tk.LAST)
                    elif direction == Direction.UP:
                        self.canvas.create_line(cx, cy, cx, y1, arrow=tk.LAST)
                    elif direction == Direction.DOWN:
                        self.canvas.create_line(cx, cy, cx, y2, arrow=tk.LAST)

                if (row, col) in state.movable_points:
                    self.canvas.create_rectangle(
                        x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="yellow", width=2
                    )

    def reset(self):
        """Reset the status"""
        self.canvas.delete("all")
        self.rows_entry.delete(0, tk.END)
        self.cols_entry.delete(0, tk.END)
        self.color_count_entry.delete(0, tk.END)
        for widget in self.color_inputs_frame.winfo_children():
            widget.destroy()
        self.path_length_label.config(text="路径长度: ")
