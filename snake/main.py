import sys
import random
import time
import numpy as np
from collections import deque
from multiprocessing import Process, Queue, Event, Value

# 导入各类寻路算法
from bfs import bfs, safe_move
from dfs import dfs, safe_move as dfs_safe_move
from A import a_star
from dijkstra import dijkstra_snake_path
from greedy import greedy_snake_path
from double_bfs import bidirectional_bfs_snake_path

# 导入UI模块
from ui_interface import SnakeGameWindow

from PyQt5 import QtWidgets, QtCore, QtGui

# ===== 游戏参数 =====
GAME_WIDTH, HEIGHT = 600, 400
BLOCK = 20
SPEED = Value('i', 10)  # int类型共享变量，初始值10,支持跨进程修改
GRID_W = GAME_WIDTH // BLOCK
GRID_H = HEIGHT // BLOCK

# ===== 颜色定义 =====
WHITE = QtGui.QColor(255, 255, 255)
HEAD_COLOR = QtGui.QColor(0, 0, 255)
GREEN = QtGui.QColor(0, 200, 0)
RED = QtGui.QColor(200, 0, 0)
BLACK = QtGui.QColor(0, 0, 0)

# ===== 随机生成食物位置 =====
def random_food(snake):
    while True:
        fx = random.randint(0, GRID_W-1)
        fy = random.randint(0, GRID_H-1)
        if (fx, fy) not in snake:
            return (fx, fy)

# ===== 子进程函数,游戏部分 =====
def game_process_main(snake_queue, fruit_queue, stop_event, start_event, speed, record_queue, algorithm):
    """
    游戏主进程
    
    参数说明:
    - snake_queue: 传递蛇、食物及分数信息的队列
    - fruit_queue: 传递每次吃到果实的时间信息的队列
    - stop_event: 停止事件
    - start_event: 开始事件
    - speed: 速度共享变量
    - record_queue: 传递游戏记录的队列
    - algorithm: 选择的寻路算法名称
    """
    snake = [(5, 5)]
    food = random_food(snake)
    snake_queue.put({"snake": list(snake), "food": food, "score": 0, "update_snake": True})
    
    last_time = time.time()
    score = 0
    snake_id = algorithm
    start_time = time.time()
    fruit_times = []
    
    # 等待开始信号
    while not start_event.is_set() and not stop_event.is_set():
        time.sleep(0.1)
        snake_queue.put({"snake": list(snake), "food": food, "score": 0, "update_snake": True})
    
    # 游戏主循环
    while not stop_event.is_set():
        head = snake[-1]
        
        # 根据选中的算法计算路径
        if algorithm == "BFS":
            path = bfs(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        elif algorithm == "DFS":
            path = dfs(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = dfs_safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        elif algorithm == "A*":
            path = a_star(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        elif algorithm == "Dijkstra":
            path = dijkstra_snake_path(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        elif algorithm == "Greedy":
            path = greedy_snake_path(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        elif algorithm == "Double_BFS":
            path = bidirectional_bfs_snake_path(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        else:
            path = bfs(head, food, snake[:-1], GRID_W, GRID_H)
            next_cell = safe_move(head, snake, GRID_W, GRID_H) if not path else path[0]
        
        if next_cell is None:
            # 游戏结束
            end_time = time.time()
            total_time = round(end_time - start_time, 2)
            avg_time = round(np.mean(fruit_times), 2) if fruit_times else 0.0
            record_queue.put({
                "snake_id": snake_id,
                "score": score,
                "total_time": total_time,
                "avg_time": avg_time
            })
            print(f"Game Over! Score: {score}")
            break

        snake.append(next_cell)
        if next_cell == food:
            score += 1
            now = time.time()
            delta = now - last_time
            last_time = now
            fruit_times.append(delta)
            fruit_queue.put({"fruit": score, "time": delta})
            food = random_food(snake)
        else:
            snake.pop(0)

        snake_queue.put({"snake": list(snake), "food": food, "score": score, "update_snake": True})
        time.sleep(1 / speed.value)


# ===== 启动程序 =====
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    snake_queue = Queue()
    fruit_queue = Queue()
    record_queue = Queue()
    stop_event = Event()
    start_event = Event()
    speed = Value('i', 10)

    # 启动游戏进程
    p_game = Process(
        target=game_process_main,
        args=(snake_queue, fruit_queue, stop_event, start_event, speed, record_queue, "BFS")
    )
    p_game.start()

    app = QtWidgets.QApplication(sys.argv)
    window = SnakeGameWindow(snake_queue, fruit_queue, record_queue, stop_event, start_event, speed, p_game)
    window.show()
    sys.exit(app.exec_())
