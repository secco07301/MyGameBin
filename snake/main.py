import sys
import random
import time
from collections import deque
from multiprocessing import Process, Queue, Event, Value

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ===== 新增：配置matplotlib中文显示 =====
import matplotlib
# 设置字体（优先使用系统自带的中文字体，避免找不到字体的问题）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 黑体/备用字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ===== 新增：游戏记录存储 =====
import uuid  # 需在import区新增该导入
game_records = []  # 存储格式：{"snake_id": 蛇唯一编号, "score": 得分, "total_time": 总耗时, "avg_time": 平均耗时}
# ===== 游戏参数 =====
GAME_WIDTH, HEIGHT = 600, 400
BLOCK = 20
# 改用共享变量存储速度，支持跨进程修改
SPEED = Value('i', 10)  # int类型共享变量，初始值10
GRID_W = GAME_WIDTH // BLOCK
GRID_H = HEIGHT // BLOCK

WHITE = QtGui.QColor(255, 255, 255)
HEAD_COLOR = QtGui.QColor(0, 0, 255)
GREEN = QtGui.QColor(0, 200, 0)
RED = QtGui.QColor(200, 0, 0)
BLACK = QtGui.QColor(0, 0, 0)

# ===== BFS 寻路 =====
def bfs(start, goal, snake):
    queue = deque([start])
    visited = {start: None}
    snake_set = set(snake)
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            nxt = (nx, ny)
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and nxt not in visited and nxt not in snake_set:
                visited[nxt] = (x, y)
                queue.append(nxt)
    if goal not in visited:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = visited[cur]
    path.reverse()
    return path

# ===== 安全移动函数 =====
def safe_move(head, snake):
    possible = []
    snake_set = set(snake)
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = head[0]+dx, head[1]+dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in snake_set:
            tail = snake[0]
            dist = abs(nx-tail[0]) + abs(ny-tail[1])
            possible.append(((nx, ny), dist))
    if possible:
        possible.sort(key=lambda x: -x[1])
        return possible[0][0]
    else:
        return None

def random_food(snake):
    while True:
        fx = random.randint(0, GRID_W-1)
        fy = random.randint(0, GRID_H-1)
        if (fx, fy) not in snake:
            return (fx, fy)

# ===== 子进程函数 =====
def game_process_main(snake_queue, fruit_queue, stop_event, speed, record_queue):  # 新增：record_queue
    snake = [(5,5)]
    food = random_food(snake)
    last_time = time.time()
    score = 0
    #snake_id = str(uuid.uuid4())[:8]  # 生成8位短编号
    snake_id = "BFS"  # 固定编号为BFS
    start_time = time.time()  # 单局开始时间
    fruit_times = []  # 存储每个果实的耗时
    
    while not stop_event.is_set():
        head = snake[-1]
        path = bfs(head, food, snake[:-1])
        if path:
            next_cell = path[0]
        else:
            next_cell = safe_move(head, snake)
            if next_cell is None:
                # 游戏结束，计算统计数据
                end_time = time.time()
                total_time = round(end_time - start_time, 2)   # 总耗时
                avg_time = round(np.mean(fruit_times), 2) if fruit_times else 0.0  # 平均耗时
                # 发送记录到主进程（替代全局变量）
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
            fruit_times.append(delta)  # 新增：记录单次耗时
            fruit_queue.put({"fruit": score, "time": delta})
            food = random_food(snake)
        else:
            snake.pop(0)

        snake_queue.put({"snake": list(snake), "food": food, "score": score, "update_snake": True})
        time.sleep(1/speed.value)

# ===== matplotlib 绘图 =====
class SnakePlotCanvas(FigureCanvas):
    def __init__(self,parent=None):
        # 调整图表尺寸，适配新界面
        self.fig=Figure(figsize=(5, 3.5), tight_layout=False)
        self.ax=self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.times=[]
        # 新增：存储平均时间、低于/高于平均值的个数
        self.average_time = 0.0
        self.below_average = 0  # 低于平均的个数
        self.above_average = 0  # 高于平均的个数
        
        # 固定子图边距
        self.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9, wspace=0, hspace=0)
        
        # 固定Canvas尺寸
        self.setFixedSize(400, 280)
        
        # 初始化坐标轴样式
        self.ax.set_xlabel("Fruit Number", fontsize=9)
        self.ax.set_ylabel("Time (s)", fontsize=9)
        self.ax.set_title("Time per Fruit (Smoothed)", fontsize=10)

    def update_plot(self):
        self.ax.clear()
        if len(self.times)>0:
            x=np.arange(1,len(self.times)+1)
            y=np.array(self.times)
            # 关键修改1：最小窗口设为1，保证任意数量点都能生成平滑数据
            window=min(3, len(y)) if len(y) >=1 else 1
            # 关键修改2：使用mode='same'，让平滑后的数据长度和原始数据一致
            y_smooth=np.convolve(y,np.ones(window)/window,mode='same')
            
            # 绘制平滑后的连线（1个点显示圆点，2个点就有连线）
            self.ax.plot(x, y_smooth, marker='o', color='blue', label='单次耗时')
            
            # 计算平均时间
            self.average_time = np.mean(y)
            # 新增：统计低于/高于平均时间的个数
            # 严格低于平均值的个数
            self.below_average = np.sum(y < self.average_time)
            # 严格高于平均值的个数
            self.above_average = np.sum(y > self.average_time)
            # 等于平均值的情况（可选：可计入任意一方，这里单独标注）
            equal_average = np.sum(y == self.average_time)
            
            # 绘制平均时间红线
            self.ax.axhline(y=self.average_time, color='red', linestyle='--', linewidth=2, 
                           label=f'平均耗时: {self.average_time:.2f}s')
            
            # 保留原始数值的文本标注
            for i,t in enumerate(y):
                self.ax.text(x[i], t, f"{t:.2f}", fontsize=8, ha='center', va='bottom')
            
            # 新增：在图表上标注统计结果（可选，增强可读性）
            stats_text = f"低于平均: {self.below_average}个\n高于平均: {self.above_average}个"
            self.ax.text(0.05, 0.85, stats_text, transform=self.ax.transAxes, 
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            # 显示图例
            self.ax.legend(fontsize=8, loc='upper right')
            
            self.ax.set_xlabel("Fruit Number", fontsize=9)
            self.ax.set_ylabel("Time (s)", fontsize=9)
            self.ax.set_title("Time per Fruit (Smoothed)", fontsize=10)
            self.ax.set_ylim(0, max(y)*1.2)
            self.ax.set_xlim(0, max(x)*1.1 if max(x) > 0 else 10)
        else:
            # 无数据时重置统计值
            self.average_time = 0.0
            self.below_average = 0
            self.above_average = 0
            self.ax.set_xlabel("Fruit Number", fontsize=9)
            self.ax.set_ylabel("Time (s)", fontsize=9)
            self.ax.set_title("Time per Fruit (Smoothed)", fontsize=10)
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 1)
        
        self.draw()

# ===== 游戏 Widget =====
class SnakeGameWidget(QtWidgets.QWidget):
    def __init__(self, q, parent=None):
        super().__init__(parent)
        self.setFixedSize(GAME_WIDTH, HEIGHT)
        self.queue = q
        self.snake = [(5,5)]
        self.food = (10,10)
        self.score = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_state)
        self.timer.start(30)

    def update_state(self):
        while not self.queue.empty():
            data = self.queue.get()
            if data.get("update_snake"):
                self.snake = data["snake"]
                self.food = data["food"]
                self.score = data["score"]
        self.update()

    def paintEvent(self,event):
        painter = QtGui.QPainter(self)
        painter.fillRect(0,0,self.width(),self.height(),WHITE)
        for x,y in self.snake[:-1]:
            painter.fillRect(x*BLOCK,y*BLOCK,BLOCK,BLOCK,GREEN)
        hx,hy = self.snake[-1]
        painter.fillRect(hx*BLOCK,hy*BLOCK,BLOCK,BLOCK,HEAD_COLOR)
        fx,fy = self.food
        painter.fillRect(fx*BLOCK,fy*BLOCK,BLOCK,BLOCK,RED)
        painter.setPen(BLACK)
        painter.setFont(QtGui.QFont('Arial',12))
        painter.drawText(10,20,f"Score: {self.score}")

# ===== 主窗口 =====
class SnakeMainWindow(QtWidgets.QWidget):
    def __init__(self, snake_queue, fruit_queue, record_queue, stop_event, speed, p_game):  # 新增：record_queue
        super().__init__()
        self.setWindowTitle("AI 贪吃蛇 + 折线统计图 (多进程版)")
        self.setGeometry(100,100,GAME_WIDTH+480,HEIGHT+50)
        self.snake_queue = snake_queue
        self.fruit_queue = fruit_queue
        self.record_queue = record_queue  # 新增：保存记录队列
        self.stop_event = stop_event
        self.speed = speed
        self.p_game = p_game
        
        # 新增：主进程维护游戏记录（替代全局变量）
        self.game_records = []

        # ===== 主布局 =====
        main_layout = QtWidgets.QHBoxLayout(self)

        # 游戏 Widget
        self.game_widget = SnakeGameWidget(snake_queue, self)
        main_layout.addWidget(self.game_widget, 3)

        # 右侧布局（垂直）
        right_layout = QtWidgets.QVBoxLayout()

        # 折线图
        self.plot_canvas = SnakePlotCanvas(self)
        right_layout.addWidget(self.plot_canvas, alignment=QtCore.Qt.AlignCenter)

        # 排名表格
        self.rank_table = QtWidgets.QTableWidget()
        self.rank_table.setColumnCount(5)
        self.rank_table.setHorizontalHeaderLabels(["排名", "蛇的编号", "得分", "总耗时(s)", "平均耗时(s)"])
        self.rank_table.verticalHeader().setVisible(False)
        self.rank_table.setFixedSize(450, 180)
        self.rank_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.rank_table.setColumnWidth(0, 60)
        self.rank_table.setColumnWidth(1, 100)
        self.rank_table.setColumnWidth(2, 60)
        self.rank_table.setColumnWidth(3, 80)
        self.rank_table.setColumnWidth(4, 80)
        self.rank_table.setAlternatingRowColors(True)
        self.rank_table.setStyleSheet("alternate-background-color: #f5f5f5;")
        right_layout.addWidget(self.rank_table, alignment=QtCore.Qt.AlignCenter)

        # 平均时间显示标签
        self.average_label = QtWidgets.QLabel("平均耗时：0.00 秒")
        self.average_label.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.average_label.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.average_label)

        # 统计结果显示标签
        self.stats_label = QtWidgets.QLabel("低于平均：0 个 | 高于平均：0 个")
        self.stats_label.setFont(QtGui.QFont('Arial', 10))
        self.stats_label.setStyleSheet("color: #333333;")
        self.stats_label.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.stats_label)

        # 速度控制区域
        speed_layout = QtWidgets.QHBoxLayout()
        speed_label = QtWidgets.QLabel("蛇移动速度：")
        speed_label.setFont(QtGui.QFont('Arial', 10))
        speed_layout.addWidget(speed_label)
        self.speed_input = QtWidgets.QLineEdit()
        self.speed_input.setPlaceholderText("输入数字（如100）")
        self.speed_input.setFixedWidth(80)
        self.speed_input.setText(str(self.speed.value))
        speed_layout.addWidget(self.speed_input)
        self.confirm_btn = QtWidgets.QPushButton("确认")
        self.confirm_btn.clicked.connect(self.update_speed)
        speed_layout.addWidget(self.confirm_btn)
        self.tip_label = QtWidgets.QLabel("（数值越大速度越快）")
        self.tip_label.setFont(QtGui.QFont('Arial', 8))
        self.tip_label.setStyleSheet("color: gray;")
        speed_layout.addWidget(self.tip_label)
        right_layout.addLayout(speed_layout)

        # 原有按钮布局
        btn_layout = QtWidgets.QHBoxLayout()
        self.restart_btn = QtWidgets.QPushButton("重新开始")
        self.exit_btn = QtWidgets.QPushButton("退出")
        btn_layout.addWidget(self.restart_btn)
        btn_layout.addWidget(self.exit_btn)
        right_layout.addLayout(btn_layout)

        main_layout.addLayout(right_layout, 2)

        # 定时更新折线图、统计信息、排名表格
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)  # 合并更新逻辑
        self.timer.start(200)

        # 按钮事件
        self.restart_btn.clicked.connect(self.restart_game)
        self.exit_btn.clicked.connect(self.close)

    def update_all(self):
        """更新折线图、统计、排名表格"""
        # 1. 更新折线图数据
        while not self.fruit_queue.empty():
            data = self.fruit_queue.get()
            self.plot_canvas.times.append(data["time"])
        self.plot_canvas.update_plot()
        
        # 2. 更新平均时间和统计标签
        self.average_label.setText(f"平均耗时：{self.plot_canvas.average_time:.2f} 秒")
        self.stats_label.setText(f"低于平均：{self.plot_canvas.below_average} 个 | 高于平均：{self.plot_canvas.above_average} 个")
        
        # 3. 接收子进程的游戏记录并更新表格
        while not self.record_queue.empty():
            record = self.record_queue.get()
            self.game_records.append(record)
            # 按得分降序、平均耗时升序排序
            self.game_records.sort(key=lambda x: (-x["score"], x["avg_time"]))
        
        # 4. 更新排名表格
        self.update_rank_table()

    def update_rank_table(self):
        """更新排名表格"""
        # 清空原有行
        self.rank_table.setRowCount(0)
        # 填充新数据
        for row_idx, record in enumerate(self.game_records):
            self.rank_table.insertRow(row_idx)
            # 排名
            self.rank_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(row_idx+1)))
            # 蛇编号
            self.rank_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(record["snake_id"]))
            # 得分
            self.rank_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(str(record["score"])))
            # 总耗时
            self.rank_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(record["total_time"])))
            # 平均耗时
            self.rank_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(str(record["avg_time"])))

    def update_speed(self):
        """更新蛇的移动速度"""
        try:
            new_speed = int(self.speed_input.text())
            if new_speed <= 0:
                self.tip_label.setText("❌ 速度必须大于0")
                self.tip_label.setStyleSheet("color: red;")
                return
            self.speed.value = new_speed
            self.tip_label.setText(f"✅ 速度已设为{new_speed}")
            self.tip_label.setStyleSheet("color: green;")
        except ValueError:
            self.tip_label.setText("❌ 请输入有效数字")
            self.tip_label.setStyleSheet("color: red;")

    def restart_game(self):
        """重新开始游戏"""
        # 停止子进程
        self.stop_event.set()
        if self.p_game.is_alive():
            self.p_game.join(timeout=2)

        # 清空队列和数据
        while not self.snake_queue.empty():
            self.snake_queue.get()
        while not self.fruit_queue.empty():
            self.fruit_queue.get()
        while not self.record_queue.empty():  # 新增：清空记录队列
            self.record_queue.get()
        
        self.plot_canvas.times.clear()
        self.plot_canvas.average_time = 0.0
        self.plot_canvas.below_average = 0
        self.plot_canvas.above_average = 0
        self.average_label.setText("平均耗时：0.00 秒")
        self.stats_label.setText("低于平均：0 个 | 高于平均：0 个")

        # 重启子进程（传入record_queue）
        self.stop_event.clear()
        self.p_game = Process(target=game_process_main, args=(self.snake_queue, self.fruit_queue, self.stop_event, self.speed, self.record_queue))
        self.p_game.start()
        
        # 重置提示标签
        self.tip_label.setText("（数值越大速度越快）")
        self.tip_label.setStyleSheet("color: gray;")
        self.speed_input.setText(str(self.speed.value))

    def closeEvent(self, event):
        self.stop_event.set()
        if self.p_game.is_alive():
            self.p_game.join(timeout=2)
        event.accept()

# ===== 启动程序 =====
if __name__=="__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    snake_queue = Queue()
    fruit_queue = Queue()
    record_queue = Queue()  # 新增：记录队列
    stop_event = Event()
    speed = Value('i', 10)

    # 启动游戏进程（传入record_queue）
    p_game = Process(target=game_process_main, args=(snake_queue, fruit_queue, stop_event, speed, record_queue))
    p_game.start()

    app = QtWidgets.QApplication(sys.argv)
    # 主窗口传入record_queue
    window = SnakeMainWindow(snake_queue, fruit_queue, record_queue, stop_event, speed, p_game)
    window.show()
    sys.exit(app.exec_())