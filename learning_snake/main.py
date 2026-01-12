import sys
import time
import random
import numpy as np
from collections import defaultdict
import pickle
import os

# PyQt5 核心组件
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLineEdit, QPushButton, QLabel, QGroupBox, QFormLayout,
    QScrollBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIntValidator, QDoubleValidator

# Pygame 仅用于游戏渲染
import pygame

# 绘图相关
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ====================== 1. 全局常量 ======================
# 游戏尺寸
GAME_WIDTH, GAME_HEIGHT = 500, 500
BLOCK_SIZE = 25

# 强化学习参数默认值 & 范围限制
DEFAULT_FPS = 10
MIN_FPS, MAX_FPS = 1, 60

DEFAULT_ALPHA = 0.1
MIN_ALPHA, MAX_ALPHA = 0.01, 1.0

DEFAULT_GAMMA = 0.9
MIN_GAMMA, MAX_GAMMA = 0.01, 1.0

DEFAULT_EPSILON = 0.1
MIN_EPSILON, MAX_EPSILON = 0.01, 1.0

DEFAULT_EPISODES = 1000
MIN_EPISODES, MAX_EPISODES = 100, 5000

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

Q_TABLE_PATH = "snake_q_table.pkl"

# 图表显示参数
PLOT_VIEW_WIDTH = 50  # 每次显示50个数据点
AUTO_SCROLL_DELAY = 3000  # 手动操作后恢复自动滚动的延迟（毫秒）

# ====================== 2. 强化学习智能体 ======================
class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = DEFAULT_ALPHA      
        self.gamma = DEFAULT_GAMMA      
        self.epsilon = DEFAULT_EPSILON  
        self.load_q_table()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1, 2, 3])
            return action, "探索"
        else:
            action = np.argmax(self.q_table[state])
            return action, "利用"

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        return current_q, new_q

    def save_q_table(self):
        """保存Q表到文件"""
        q_table_dict = dict(self.q_table)
        with open(Q_TABLE_PATH, 'wb') as f:
            pickle.dump(q_table_dict, f)
        return len(self.q_table)

    def load_q_table(self):
        """从文件加载Q表"""
        if os.path.exists(Q_TABLE_PATH):
            with open(Q_TABLE_PATH, 'rb') as f:
                q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(4), q_table_dict)
            return len(self.q_table)
        return 0

    def reset(self):
        """重置Q表和参数"""
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = DEFAULT_ALPHA
        self.gamma = DEFAULT_GAMMA
        self.epsilon = DEFAULT_EPSILON
        if os.path.exists(Q_TABLE_PATH):
            os.remove(Q_TABLE_PATH)
        return 0

# ====================== 3. 贪吃蛇游戏核心 ======================
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.reset()

    def reset(self):
        self.snake = [(GAME_WIDTH//2, GAME_HEIGHT//2)]
        self.direction = (BLOCK_SIZE, 0)
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.max_steps = 200
        self.collision_reason = ""
        return self._get_state()

    def _generate_food(self):
        while True:
            x = random.randint(0, (GAME_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (GAME_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food_pos = (x, y)
            if food_pos not in self.snake:
                return food_pos

    def _get_state(self):
        head_x, head_y = self.snake[0]
        up_obstacle = (head_y - BLOCK_SIZE < 0) or ((head_x, head_y - BLOCK_SIZE) in self.snake)
        down_obstacle = (head_y + BLOCK_SIZE >= GAME_HEIGHT) or ((head_x, head_y + BLOCK_SIZE) in self.snake)
        left_obstacle = (head_x - BLOCK_SIZE < 0) or ((head_x - BLOCK_SIZE, head_y) in self.snake)
        right_obstacle = (head_x + BLOCK_SIZE >= GAME_WIDTH) or ((head_x + BLOCK_SIZE, head_y) in self.snake)
        food_up = (self.food[1] < head_y)
        food_down = (self.food[1] > head_y)
        food_left = (self.food[0] < head_x)
        food_right = (self.food[0] > head_x)
        return (up_obstacle, down_obstacle, left_obstacle, right_obstacle,
                food_up, food_down, food_left, food_right)

    def _check_collision(self):
        head_x, head_y = self.snake[0]
        if head_x < 0 or head_x >= GAME_WIDTH or head_y < 0 or head_y >= GAME_HEIGHT:
            self.collision_reason = "撞墙"
            return True
        if (head_x, head_y) in self.snake[1:]:
            self.collision_reason = "撞自身"
            return True
        return False

    def step(self, action):
        action_dirs = [(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)]
        action_dir = action_dirs[action]
        
        if (self.direction == (0, -BLOCK_SIZE) and action_dir == (0, BLOCK_SIZE)) or \
           (self.direction == (0, BLOCK_SIZE) and action_dir == (0, -BLOCK_SIZE)) or \
           (self.direction == (-BLOCK_SIZE, 0) and action_dir == (BLOCK_SIZE, 0)) or \
           (self.direction == (BLOCK_SIZE, 0) and action_dir == (-BLOCK_SIZE, 0)):
            action_dir = self.direction
        
        self.direction = action_dir
        new_head = (self.snake[0][0] + action_dir[0], self.snake[0][1] + action_dir[1])
        self.snake.insert(0, new_head)
        self.steps += 1
        reward = 0
        eat_food = False

        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._generate_food()
            self.steps = 0
            eat_food = True
        else:
            self.snake.pop()

        if self._check_collision():
            self.game_over = True
            reward = -10
        elif self.steps >= self.max_steps:
            self.game_over = True
            self.collision_reason = "步数超限"
            reward = -10

        return self._get_state(), reward, self.game_over, eat_food, self.collision_reason

    def render(self):
        self.screen.fill(BLACK)
        for i, segment in enumerate(self.snake):
            color = BLUE if i == 0 else GREEN
            pygame.draw.rect(self.screen, color, (segment[0], segment[1], BLOCK_SIZE-1, BLOCK_SIZE-1))
        pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], BLOCK_SIZE-1, BLOCK_SIZE-1))
        
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        h, w, ch = frame.shape
        frame_bytes = frame.tobytes()
        bytes_per_line = ch * w
        q_image = QImage(frame_bytes, w, h, bytes_per_line, QImage.Format_RGB888)
        return q_image

# ====================== 4. 智能自动滑动折线图组件 ======================
class AutoScrollableScorePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)
        
        # 初始化数据
        self.x_data = []  # 蛇的编号
        self.y_data = []  # 得分
        self.scroll_pos = 0  # 滚动位置
        self.auto_scroll = True  # 自动滚动开关
        self.manual_scroll_timer = QTimer()  # 手动操作后恢复自动滚动的定时器
        self.manual_scroll_timer.setSingleShot(True)
        self.manual_scroll_timer.timeout.connect(self.resume_auto_scroll)
        
        # 创建Figure和Canvas
        self.fig = Figure(figsize=(6, 3.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        # 创建水平滚动条
        self.scroll_bar = QScrollBar(Qt.Horizontal, self)
        self.scroll_bar.valueChanged.connect(self.on_scroll)
        # 监听滚动条的鼠标按下/释放事件，判断是否手动操作
        self.scroll_bar.sliderPressed.connect(self.pause_auto_scroll)
        self.scroll_bar.sliderReleased.connect(self.start_manual_timer)
        
        # 添加组件到布局
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.scroll_bar)
        
        # 初始化图表
        self.init_plot()
        
    def init_plot(self):
        """初始化图表样式"""
        self.ax.clear()
        self.ax.set_title('贪吃蛇训练得分趋势', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('蛇的出场编号', fontsize=10)
        self.ax.set_ylabel('得分', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, PLOT_VIEW_WIDTH)
        self.ax.set_ylim(0, 20)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
        self.fig.tight_layout()
        self.canvas.draw()
        
    def update_data(self, snake_id, score):
        """添加新数据并更新图表"""
        # 添加数据
        self.x_data.append(snake_id)
        self.y_data.append(score)
        
        # 更新滚动条范围
        max_scroll = max(0, len(self.x_data) - PLOT_VIEW_WIDTH)
        self.scroll_bar.setRange(0, max_scroll)
        self.scroll_bar.setPageStep(PLOT_VIEW_WIDTH // 5)  # 每次滚动10个点
        self.scroll_bar.setSingleStep(5)  # 单次步长5个点
        
        # 如果开启自动滚动，滚动到最右侧
        if self.auto_scroll:
            self.scroll_pos = max_scroll
            self.scroll_bar.setValue(self.scroll_pos)
        
        # 更新图表显示
        self.update_plot()
        
    def update_plot(self):
        """根据滚动位置更新图表显示"""
        self.ax.clear()
        
        # 计算显示范围
        end_pos = self.scroll_pos + PLOT_VIEW_WIDTH
        display_x = self.x_data[self.scroll_pos:end_pos]
        display_y = self.y_data[self.scroll_pos:end_pos]
        
        # 绘制折线图
        if display_x and display_y:
            self.ax.plot(display_x, display_y, 
                        color='#2196F3', linewidth=2, marker='o', markersize=4, 
                        markerfacecolor='#FF9800', markeredgecolor='white', markeredgewidth=1)
            
            # 设置x轴范围
            self.ax.set_xlim(min(display_x) - 1 if display_x else 0, 
                           max(display_x) + 1 if display_x else PLOT_VIEW_WIDTH)
            
            # 设置y轴范围（自适应）
            y_max = max(max(display_y) + 2, 20) if display_y else 20
            self.ax.set_ylim(0, y_max)
            
            # 添加最优得分标注
            if self.y_data:
                global_max_score = max(self.y_data)
                global_max_idx = self.y_data.index(global_max_score)
                global_max_id = self.x_data[global_max_idx]
                
                # 只在当前视图范围内显示标注
                if self.scroll_pos <= global_max_idx < self.scroll_pos + PLOT_VIEW_WIDTH:
                    self.ax.annotate(f'最优: {global_max_score}', 
                                   xy=(global_max_id, global_max_score), 
                                   xytext=(global_max_id+2, global_max_score+1),
                                   arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5),
                                   fontsize=9, color='#4CAF50', fontweight='bold')
        
        else:
            self.ax.set_xlim(0, PLOT_VIEW_WIDTH)
            self.ax.set_ylim(0, 20)
        
        # 重置样式
        self.ax.set_title('贪吃蛇训练得分趋势', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('蛇的出场编号', fontsize=10)
        self.ax.set_ylabel('得分', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def on_scroll(self, value):
        """滚动条事件处理"""
        self.scroll_pos = value
        self.update_plot()
        
    def pause_auto_scroll(self):
        """暂停自动滚动（手动操作时）"""
        self.auto_scroll = False
        self.manual_scroll_timer.stop()  # 停止之前的定时器
        
    def start_manual_timer(self):
        """启动定时器，延迟后恢复自动滚动"""
        self.manual_scroll_timer.start(AUTO_SCROLL_DELAY)
        
    def resume_auto_scroll(self):
        """恢复自动滚动"""
        self.auto_scroll = True
        # 滚动到最新数据
        max_scroll = max(0, len(self.x_data) - PLOT_VIEW_WIDTH)
        self.scroll_pos = max_scroll
        self.scroll_bar.setValue(self.scroll_pos)
        self.update_plot()
        
    def clear_plot(self):
        """清空图表和数据"""
        self.x_data = []
        self.y_data = []
        self.scroll_pos = 0
        self.auto_scroll = True
        self.scroll_bar.setValue(0)
        self.scroll_bar.setRange(0, 0)
        self.manual_scroll_timer.stop()
        self.init_plot()

# ====================== 5. 主窗口 ======================
class SnakeRLMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("强化学习贪吃蛇")
        self.setFixedSize(1200, 700)

        # 初始化核心组件
        self.game = SnakeGame()
        self.agent = QLearningAgent()
        self.best_score = 0
        self.current_episode = 0  # 蛇的出场编号
        self.total_episodes = DEFAULT_EPISODES
        self.paused = False

        # 保存参数原始值
        self.original_params = {
            "fps": DEFAULT_FPS,
            "alpha": DEFAULT_ALPHA,
            "gamma": DEFAULT_GAMMA,
            "epsilon": DEFAULT_EPSILON,
            "episodes": DEFAULT_EPISODES
        }

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setCentralWidget(main_widget)

        # ========== 左侧：游戏显示区 + 状态信息 ==========
        left_widget = QWidget()
        left_widget.setFixedSize(GAME_WIDTH, 600)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(left_widget)

        # 状态信息栏
        status_bar_widget = QWidget()
        status_bar_widget.setFixedHeight(50)
        status_bar_layout = QHBoxLayout(status_bar_widget)
        status_bar_layout.setSpacing(20)
        status_bar_layout.setContentsMargins(10, 0, 10, 0)
        status_bar_layout.setAlignment(Qt.AlignCenter)

        # 当前得分
        current_score_label = QLabel("当前得分：")
        current_score_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.current_score_value = QLabel(f"{self.game.score}")
        self.current_score_value.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.current_score_value.setStyleSheet("color: #4CAF50;")

        # 最优得分
        best_score_label = QLabel("最优得分：")
        best_score_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.best_score_value = QLabel(f"{self.best_score}")
        self.best_score_value.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.best_score_value.setStyleSheet("color: #FF9800;")

        # 训练进度
        progress_label = QLabel("训练进度：")
        progress_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.progress_value = QLabel(f"{self.current_episode}/{self.total_episodes}")
        self.progress_value.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.progress_value.setStyleSheet("color: #2196F3;")

        # 添加到状态栏布局
        status_bar_layout.addWidget(current_score_label)
        status_bar_layout.addWidget(self.current_score_value)
        status_bar_layout.addWidget(best_score_label)
        status_bar_layout.addWidget(self.best_score_value)
        status_bar_layout.addWidget(progress_label)
        status_bar_layout.addWidget(self.progress_value)

        # 游戏显示标签
        self.game_label = QLabel()
        self.game_label.setFixedSize(GAME_WIDTH, GAME_HEIGHT)
        self.game_label.setStyleSheet("border: 3px solid #333; background: black;")

        # 添加到左侧布局
        left_layout.addWidget(status_bar_widget)
        left_layout.addWidget(self.game_label)

        # ========== 右侧：控制面板 + 智能自动滑动折线图 ==========
        right_widget = QWidget()
        right_widget.setFixedSize(650, 650)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(20)
        main_layout.addWidget(right_widget)

        # ---------- 子布局1：参数调节组 ----------
        param_group = QGroupBox()
        param_group.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        
        # 自定义标题栏
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 5, 10, 5)
        title_layout.setSpacing(20)
        
        # 标题文字
        title_label = QLabel("强化学习参数调节")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        
        # 拉伸因子
        title_layout.addStretch()
        
        # 确认/取消按钮
        self.confirm_btn = QPushButton("确认修改")
        self.confirm_btn.setFixedSize(90, 35)
        self.confirm_btn.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        self.confirm_btn.setStyleSheet("""
            QPushButton {background-color: #4CAF50; color: white; border: none; border-radius: 6px;}
            QPushButton:hover {background-color: #388E3C;}
        """)
        self.confirm_btn.clicked.connect(self.confirm_params)
        
        self.cancel_btn = QPushButton("取消修改")
        self.cancel_btn.setFixedSize(90, 35)
        self.cancel_btn.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        self.cancel_btn.setStyleSheet("""
            QPushButton {background-color: #FF5722; color: white; border: none; border-radius: 6px;}
            QPushButton:hover {background-color: #E64A19;}
        """)
        self.cancel_btn.clicked.connect(self.cancel_params)
        
        title_layout.addWidget(self.confirm_btn)
        title_layout.addWidget(self.cancel_btn)

        # 参数表单布局
        param_form_layout = QFormLayout()
        param_form_layout.setSpacing(15)
        param_form_layout.setContentsMargins(10, 5, 10, 10)

        # 1. FPS参数行
        fps_label = QLabel("运行速度(FPS) [1-60]:")
        fps_label.setFont(QFont("Microsoft YaHei", 7))
        self.fps_edit = QLineEdit(str(DEFAULT_FPS))
        self.fps_edit.setFixedWidth(100)
        fps_validator = QIntValidator(MIN_FPS, MAX_FPS, self)
        self.fps_edit.setValidator(fps_validator)
        self.fps_edit.setFont(QFont("Microsoft YaHei", 7))
        param_form_layout.addRow(fps_label, self.fps_edit)

        # 2. 学习率α参数行
        alpha_label = QLabel("学习率α [0.01-1.0]:")
        alpha_label.setFont(QFont("Microsoft YaHei", 7))
        self.alpha_edit = QLineEdit(f"{DEFAULT_ALPHA:.2f}")
        self.alpha_edit.setFixedWidth(100)
        alpha_validator = QDoubleValidator(MIN_ALPHA, MAX_ALPHA, 2, self)
        alpha_validator.setNotation(QDoubleValidator.StandardNotation)
        self.alpha_edit.setValidator(alpha_validator)
        self.alpha_edit.setFont(QFont("Microsoft YaHei", 7))
        param_form_layout.addRow(alpha_label, self.alpha_edit)

        # 3. 折扣因子γ参数行
        gamma_label = QLabel("折扣因子γ [0.01-1.0]:")
        gamma_label.setFont(QFont("Microsoft YaHei", 7))
        self.gamma_edit = QLineEdit(f"{DEFAULT_GAMMA:.2f}")
        self.gamma_edit.setFixedWidth(100)
        gamma_validator = QDoubleValidator(MIN_GAMMA, MAX_GAMMA, 2, self)
        gamma_validator.setNotation(QDoubleValidator.StandardNotation)
        self.gamma_edit.setFont(QFont("Microsoft YaHei", 7))
        param_form_layout.addRow(gamma_label, self.gamma_edit)

        # 4. 探索率ε参数行
        epsilon_label = QLabel("探索率ε [0.01-1.0]:")
        epsilon_label.setFont(QFont("Microsoft YaHei", 7))
        self.epsilon_edit = QLineEdit(f"{DEFAULT_EPSILON:.2f}")
        self.epsilon_edit.setFixedWidth(100)
        epsilon_validator = QDoubleValidator(MIN_EPSILON, MAX_EPSILON, 2, self)
        epsilon_validator.setNotation(QDoubleValidator.StandardNotation)
        self.epsilon_edit.setFont(QFont("Microsoft YaHei", 7))
        param_form_layout.addRow(epsilon_label, self.epsilon_edit)

        # 5. 训练轮次参数行
        episode_label = QLabel("训练总轮次 [100-5000]:")
        episode_label.setFont(QFont("Microsoft YaHei", 7))
        self.episode_edit = QLineEdit(str(DEFAULT_EPISODES))
        self.episode_edit.setFixedWidth(100)
        episode_validator = QIntValidator(MIN_EPISODES, MAX_EPISODES, self)
        self.episode_edit.setValidator(episode_validator)
        self.episode_edit.setFont(QFont("Microsoft YaHei", 7))
        param_form_layout.addRow(episode_label, self.episode_edit)

        # 组合参数组的布局
        param_group_layout = QVBoxLayout(param_group)
        param_group_layout.setContentsMargins(0, 0, 0, 0)
        param_group_layout.setSpacing(0)
        param_group_layout.addWidget(title_bar)
        param_group_layout.addLayout(param_form_layout)

        right_layout.addWidget(param_group)

        # ---------- 子布局2：功能按钮组 ----------
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setFixedSize(110, 45)
        self.pause_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        self.pause_btn.setStyleSheet("""
            QPushButton {background-color: #2196F3; color: white; border: none; border-radius: 8px; font-size: 11px;}
            QPushButton:hover {background-color: #1976D2;}
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)

        self.restart_btn = QPushButton("重新开始")
        self.restart_btn.setFixedSize(110, 45)
        self.restart_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        self.restart_btn.setStyleSheet("""
            QPushButton {background-color: #FF9800; color: white; border: none; border-radius: 8px; font-size: 11px;}
            QPushButton:hover {background-color: #F57C00;}
        """)
        self.restart_btn.clicked.connect(self.restart_training)

        # 保存强化学习成果按钮
        self.save_btn = QPushButton("保存强化学习成果")
        self.save_btn.setFixedSize(150, 45)
        self.save_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        self.save_btn.setStyleSheet("""
            QPushButton {background-color: #8BC34A; color: white; border: none; border-radius: 8px; font-size: 11px;}
            QPushButton:hover {background-color: #7CB342;}
        """)
        self.save_btn.clicked.connect(self.save_rl_results)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.setFixedSize(110, 45)
        self.exit_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        self.exit_btn.setStyleSheet("""
            QPushButton {background-color: #F44336; color: white; border: none; border-radius: 8px; font-size: 11px;}
            QPushButton:hover {background-color: #D32F2F;}
        """)
        self.exit_btn.clicked.connect(self.safe_exit)

        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.restart_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.exit_btn)

        right_layout.addLayout(btn_layout)

        # ---------- 子布局3：智能自动滑动折线图 ----------
        plot_group = QGroupBox("得分趋势图")
        plot_group.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建智能自动滑动折线图组件
        self.auto_scroll_plot = AutoScrollableScorePlot(self)
        plot_layout.addWidget(self.auto_scroll_plot)
        
        right_layout.addWidget(plot_group)

        # ========== 定时器 ==========
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(int(1000/DEFAULT_FPS))

        # 初始游戏状态
        self.state = self.game.reset()

    # ---------- 保存强化学习成果 ----------
    def save_rl_results(self):
        """手动保存Q表和训练成果"""
        try:
            exp_count = self.agent.save_q_table()
            print(f"成功保存强化学习成果 | Q表经验数：{exp_count} | 当前ε={self.agent.epsilon:.2f} | 最优得分={self.best_score}")
        except Exception as e:
            print(f"保存失败：{str(e)}")

    # ---------- 参数确认/取消 ----------
    def confirm_params(self):
        """确认参数修改并生效"""
        # 1. 处理FPS
        try:
            fps = int(self.fps_edit.text())
            if not (MIN_FPS <= fps <= MAX_FPS):
                raise ValueError
            self.timer.setInterval(int(1000/fps))
            self.original_params["fps"] = fps
        except ValueError:
            fps = self.original_params["fps"]
            self.fps_edit.setText(str(fps))

        # 2. 处理学习率α
        try:
            alpha = float(self.alpha_edit.text())
            if not (MIN_ALPHA <= alpha <= MAX_ALPHA):
                raise ValueError
            self.agent.alpha = alpha
            self.original_params["alpha"] = alpha
        except ValueError:
            alpha = self.original_params["alpha"]
            self.alpha_edit.setText(f"{alpha:.2f}")

        # 3. 处理折扣因子γ
        try:
            gamma = float(self.gamma_edit.text())
            if not (MIN_GAMMA <= gamma <= MAX_GAMMA):
                raise ValueError
            self.agent.gamma = gamma
            self.original_params["gamma"] = gamma
        except ValueError:
            gamma = self.original_params["gamma"]
            self.gamma_edit.setText(f"{gamma:.2f}")

        # 4. 处理探索率ε
        try:
            epsilon = float(self.epsilon_edit.text())
            if not (MIN_EPSILON <= epsilon <= MAX_EPSILON):
                raise ValueError
            self.agent.epsilon = epsilon
            self.original_params["epsilon"] = epsilon
        except ValueError:
            epsilon = self.original_params["epsilon"]
            self.epsilon_edit.setText(f"{epsilon:.2f}")

        # 5. 处理训练轮次
        try:
            episodes = int(self.episode_edit.text())
            if not (MIN_EPISODES <= episodes <= MAX_EPISODES):
                raise ValueError
            self.total_episodes = episodes
            self.original_params["episodes"] = episodes
            self.progress_value.setText(f"{self.current_episode}/{episodes}")
        except ValueError:
            episodes = self.original_params["episodes"]
            self.episode_edit.setText(str(episodes))
            self.progress_value.setText(f"{self.current_episode}/{episodes}")

    def cancel_params(self):
        """取消参数修改，恢复原始值"""
        # 恢复输入框值
        self.fps_edit.setText(str(self.original_params["fps"]))
        self.alpha_edit.setText(f"{self.original_params['alpha']:.2f}")
        self.gamma_edit.setText(f"{self.original_params['gamma']:.2f}")
        self.epsilon_edit.setText(f"{self.original_params['epsilon']:.2f}")
        self.episode_edit.setText(str(self.original_params["episodes"]))
        
        # 恢复进度显示
        self.progress_value.setText(f"{self.current_episode}/{self.original_params['episodes']}")

    # ---------- 功能函数 ----------
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("开始" if self.paused else "暂停")

    def restart_training(self):
        self.game.reset()
        self.agent.reset()
        self.current_episode = 0
        self.best_score = 0
        self.paused = False
        self.pause_btn.setText("暂停")
        
        # 重置参数为默认值
        self.original_params = {
            "fps": DEFAULT_FPS,
            "alpha": DEFAULT_ALPHA,
            "gamma": DEFAULT_GAMMA,
            "epsilon": DEFAULT_EPSILON,
            "episodes": DEFAULT_EPISODES
        }
        
        # 恢复输入框
        self.fps_edit.setText(str(DEFAULT_FPS))
        self.alpha_edit.setText(f"{DEFAULT_ALPHA:.2f}")
        self.gamma_edit.setText(f"{DEFAULT_GAMMA:.2f}")
        self.epsilon_edit.setText(f"{DEFAULT_EPSILON:.2f}")
        self.episode_edit.setText(str(DEFAULT_EPISODES))
        
        # 恢复定时器
        self.timer.setInterval(int(1000/DEFAULT_FPS))
        
        # 清空折线图
        self.auto_scroll_plot.clear_plot()
        
        self.update_status_labels()

    def safe_exit(self):
        QApplication.quit()

    def update_status_labels(self):
        """更新状态显示"""
        self.current_score_value.setText(f"{self.game.score}")
        self.best_score_value.setText(f"{self.best_score}")
        self.progress_value.setText(f"{self.current_episode}/{self.total_episodes}")

    def update_game(self):
        """游戏主循环"""
        try:
            current_fps = int(self.fps_edit.text()) if self.fps_edit.text() else self.original_params["fps"]
        except ValueError:
            current_fps = self.original_params["fps"]

        if not self.paused and self.current_episode < self.total_episodes:
            action, action_type = self.agent.choose_action(self.state)
            next_state, reward, game_over, eat_food, collision_reason = self.game.step(action)

            if not game_over:
                self.agent.update_q_table(self.state, action, reward, next_state)

            # 更新当前得分显示
            self.current_score_value.setText(f"{self.game.score}")

            if game_over:
                # 更新最优得分
                if self.game.score > self.best_score:
                    self.best_score = self.game.score
                    self.best_score_value.setText(f"{self.best_score}")
                
                # 增加蛇的出场编号
                self.current_episode += 1
                
                # 更新折线图数据
                self.auto_scroll_plot.update_data(self.current_episode, self.game.score)
                
                # 更新训练进度
                self.progress_value.setText(f"{self.current_episode}/{self.total_episodes}")
                
                # 动态调整探索率
                if self.current_episode > self.total_episodes * 0.8:
                    new_epsilon = max(MIN_EPSILON, self.agent.epsilon - 0.0001)
                    self.agent.epsilon = new_epsilon
                    self.epsilon_edit.setText(f"{new_epsilon:.2f}")
                    self.original_params["epsilon"] = new_epsilon
                
                # 重置游戏状态
                self.state = self.game.reset()

            else:
                self.state = next_state

        # 渲染游戏画面
        q_image = self.game.render()
        self.game_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        event.accept()

# ====================== 6. 程序入口 ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)
    window = SnakeRLMainWindow()
    window.show()
    sys.exit(app.exec_())