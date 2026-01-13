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
    QScrollBar, QSizePolicy, QMessageBox, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIntValidator, QDoubleValidator, QColor, QPalette

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

# 颜色定义 - 现代化配色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 70, 90)  # 更鲜艳的红色
GREEN = (70, 220, 120)  # 更鲜艳的绿色
BLUE = (80, 150, 255)  # 更鲜艳的蓝色
PURPLE = (160, 90, 255)  # 紫色用于蛇头

# UI颜色方案
PRIMARY_COLOR = "#2C3E50"  # 主色调
SECONDARY_COLOR = "#34495E"  # 次色调
ACCENT_COLOR = "#3498DB"  # 强调色
SUCCESS_COLOR = "#2ECC71"  # 成功色
WARNING_COLOR = "#E74C3C"  # 警告色
LIGHT_BG = "#ECF0F1"  # 浅色背景

# Q表保存路径（区分普通版和最优版）
Q_TABLE_PATH = "snake_q_table.pkl"
BEST_Q_TABLE_PATH = "snake_best_q_table.pkl"
# 最优得分记录文件
BEST_SCORE_PATH = "best_score_record.txt"
# 新增：最优参数保存路径
BEST_PARAMS_PATH = "snake_best_params.pkl"

# 图表显示参数
PLOT_VIEW_WIDTH = 50  # 每次显示50个数据点
AUTO_SCROLL_DELAY = 3000  # 手动操作后恢复自动滚动的延迟（毫秒）

# ====================== 2. 强化学习智能体（集成最优成果保存） ======================
class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = DEFAULT_ALPHA      
        self.gamma = DEFAULT_GAMMA      
        self.epsilon = DEFAULT_EPSILON  
        # 初始化最优得分记录
        self.best_score = self.load_best_score()
        # 初始化最优参数
        self.best_params = {
            "alpha": DEFAULT_ALPHA,
            "gamma": DEFAULT_GAMMA,
            "epsilon": DEFAULT_EPSILON
        }
        # 加载最新Q表（训练中使用）
        self.load_q_table()
        # 加载最优Q表（初始时如果有则使用）
        self.load_best_q_table()
        # 新增：加载最优参数
        self.load_best_params()

    def choose_action(self, state):
        """选择动作（集成安全移动逻辑）"""
        # 先获取安全动作列表
        safe_actions = self.get_safe_actions(state)
        
        # 如果有安全动作，只在安全动作中选择
        if safe_actions:
            if random.uniform(0, 1) < self.epsilon:
                # 探索：从安全动作中随机选
                action = random.choice(safe_actions)
                return action, "探索(安全)"
            else:
                # 利用：从安全动作中选Q值最大的
                safe_q_values = [self.q_table[state][a] for a in safe_actions]
                max_q = max(safe_q_values)
                best_actions = [a for a, q in zip(safe_actions, safe_q_values) if q == max_q]
                action = random.choice(best_actions)
                return action, "利用(安全)"
        else:
            # 无安全动作时，按原逻辑选择（避免死锁）
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice([0, 1, 2, 3])
                return action, "探索(危险)"
            else:
                action = np.argmax(self.q_table[state])
                return action, "利用(危险)"

    def get_safe_actions(self, state):
        """安全移动核心函数：返回所有安全的动作"""
        safe_actions = []
        
        # 解析状态中的障碍物信息
        up_obstacle, down_obstacle, left_obstacle, right_obstacle = state[:4]
        
        # 检查每个方向是否安全（无墙壁/自身身体）
        if not up_obstacle:
            safe_actions.append(0)  # 上安全
        if not down_obstacle:
            safe_actions.append(1)  # 下安全
        if not left_obstacle:
            safe_actions.append(2)  # 左安全
        if not right_obstacle:
            safe_actions.append(3)  # 右安全
            
        return safe_actions

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        return current_q, new_q

    def save_q_table(self):
        """保存当前Q表（训练过程中常规保存）"""
        q_table_dict = dict(self.q_table)
        with open(Q_TABLE_PATH, 'wb') as f:
            pickle.dump(q_table_dict, f)
        return len(self.q_table)

    def load_q_table(self):
        """加载当前Q表（训练中使用）"""
        if os.path.exists(Q_TABLE_PATH):
            with open(Q_TABLE_PATH, 'rb') as f:
                q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(4), q_table_dict)
            return len(self.q_table)
        return 0

    def save_best_q_table(self, current_score):
        """
        保存最优强化学习成果（仅当当前得分超过历史最优时）
        :param current_score: 当前轮次的得分
        :return: 是否保存了新的最优成果
        """
        if current_score > self.best_score:
            # 更新最优得分记录
            self.best_score = current_score
            # 保存最优Q表
            best_q_table_dict = dict(self.q_table)
            with open(BEST_Q_TABLE_PATH, 'wb') as f:
                pickle.dump(best_q_table_dict, f)
            # 新增：保存当前参数作为最优参数
            self.save_best_params()
            # 保存最优得分记录（便于查看）
            self.save_best_score()
            # 打印日志
            exp_count = len(self.q_table)
            print(f"🎉 发现最优成果！得分：{self.best_score} | Q表经验数：{exp_count} | 最优参数：α={self.alpha:.2f}, γ={self.gamma:.2f}, ε={self.epsilon:.2f} | 已保存到 {BEST_Q_TABLE_PATH} & {BEST_PARAMS_PATH}")
            return True
        return False

    def load_best_q_table(self):
        """加载最优Q表（用于恢复最佳训练成果）"""
        if os.path.exists(BEST_Q_TABLE_PATH):
            with open(BEST_Q_TABLE_PATH, 'rb') as f:
                best_q_table_dict = pickle.load(f)
            # 最优Q表仅作为参考，训练仍使用当前Q表
            print(f"📌 加载最优Q表 | 历史最优得分：{self.best_score} | 经验数：{len(best_q_table_dict)}")
            return len(best_q_table_dict)
        return 0

    # 新增：保存最优参数
    def save_best_params(self):
        """保存当前参数作为最优参数"""
        self.best_params = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "score": self.best_score  # 关联得分，便于追溯
        }
        with open(BEST_PARAMS_PATH, 'wb') as f:
            pickle.dump(self.best_params, f)
        print(f"📌 保存最优参数 | α={self.alpha:.2f}, γ={self.gamma:.2f}, ε={self.epsilon:.2f} | 已保存到 {BEST_PARAMS_PATH}")

    # 新增：加载最优参数
    def load_best_params(self):
        """加载最优参数（程序启动时）"""
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH, 'rb') as f:
                self.best_params = pickle.load(f)
            # 打印加载日志
            print(f"📌 加载最优参数 | α={self.best_params['alpha']:.2f}, γ={self.best_params['gamma']:.2f}, ε={self.best_params['epsilon']:.2f} | 对应得分：{self.best_params.get('score', 0)}")
            return self.best_params
        return None

    def save_best_score(self):
        """保存最优得分到文件"""
        with open(BEST_SCORE_PATH, 'w') as f:
            f.write(f"{self.best_score}")

    def load_best_score(self):
        """从文件加载最优得分"""
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH, 'r') as f:
                try:
                    return int(f.read().strip())
                except:
                    return 0
        return 0

    def reset(self):
        """重置Q表和参数（保留最优成果）"""
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = DEFAULT_ALPHA
        self.gamma = DEFAULT_GAMMA
        self.epsilon = DEFAULT_EPSILON
        # 重置时仅删除当前训练的Q表，保留最优Q表、最优得分和最优参数
        if os.path.exists(Q_TABLE_PATH):
            os.remove(Q_TABLE_PATH)
        # 不删除最优成果文件
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
        self.max_steps = 500  # 增加最大步数，给小蛇更多移动空间
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
        """获取游戏状态（包含障碍物和食物位置信息）"""
        head_x, head_y = self.snake[0]
        
        # 检测各个方向的障碍物（墙壁/自身身体）
        up_obstacle = (head_y - BLOCK_SIZE < 0) or ((head_x, head_y - BLOCK_SIZE) in self.snake)
        down_obstacle = (head_y + BLOCK_SIZE >= GAME_HEIGHT) or ((head_x, head_y + BLOCK_SIZE) in self.snake)
        left_obstacle = (head_x - BLOCK_SIZE < 0) or ((head_x - BLOCK_SIZE, head_y) in self.snake)
        right_obstacle = (head_x + BLOCK_SIZE >= GAME_WIDTH) or ((head_x + BLOCK_SIZE, head_y) in self.snake)
        
        # 检测食物相对位置
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
        """执行动作并返回新状态"""
        # 动作映射：0-上, 1-下, 2-左, 3-右
        action_dirs = [(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)]
        action_dir = action_dirs[action]
        
        # 禁止直接反向移动（额外安全保障）
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

        # 吃到食物
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._generate_food()
            self.steps = 0  # 重置步数计数器
            eat_food = True
        else:
            self.snake.pop()

        # 碰撞检测
        if self._check_collision():
            self.game_over = True
            reward = -10
        elif self.steps >= self.max_steps:
            self.game_over = True
            self.collision_reason = "步数超限"
            reward = -5  # 降低步数超限的惩罚

        return self._get_state(), reward, self.game_over, eat_food, self.collision_reason

    def render(self):
        """渲染游戏画面"""
        self.screen.fill(BLACK)
        for i, segment in enumerate(self.snake):
            color = PURPLE if i == 0 else GREEN  # 蛇头用紫色，身体用绿色
            pygame.draw.rect(self.screen, color, (segment[0], segment[1], BLOCK_SIZE-1, BLOCK_SIZE-1))
            # 给蛇头添加眼睛效果
            if i == 0:
                eye_size = BLOCK_SIZE // 5
                # 根据方向绘制眼睛
                if self.direction == (0, -BLOCK_SIZE):  # 向上
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + 5, segment[1] + 5, eye_size, eye_size))
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + BLOCK_SIZE - 10, segment[1] + 5, eye_size, eye_size))
                elif self.direction == (0, BLOCK_SIZE):  # 向下
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + 5, segment[1] + BLOCK_SIZE - 10, eye_size, eye_size))
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + BLOCK_SIZE - 10, segment[1] + BLOCK_SIZE - 10, eye_size, eye_size))
                elif self.direction == (-BLOCK_SIZE, 0):  # 向左
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + 5, segment[1] + 5, eye_size, eye_size))
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + 5, segment[1] + BLOCK_SIZE - 10, eye_size, eye_size))
                elif self.direction == (BLOCK_SIZE, 0):  # 向右
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + BLOCK_SIZE - 10, segment[1] + 5, eye_size, eye_size))
                    pygame.draw.rect(self.screen, WHITE, (segment[0] + BLOCK_SIZE - 10, segment[1] + BLOCK_SIZE - 10, eye_size, eye_size))
        
        # 绘制食物（带高光效果）
        pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], BLOCK_SIZE-1, BLOCK_SIZE-1))
        # 添加食物高光
        pygame.draw.rect(self.screen, (255, 200, 200), (self.food[0] + 3, self.food[1] + 3, BLOCK_SIZE//3, BLOCK_SIZE//3))
        
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
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor='#F5F7FA')  # 增加高度
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        # 创建水平滚动条
        self.scroll_bar = QScrollBar(Qt.Horizontal, self)
        self.scroll_bar.setStyleSheet("""
            QScrollBar:horizontal {
                border: none;
                background: #E0E0E0;
                height: 12px;
                border-radius: 6px;
                margin: 5px 0px 5px 0px;  # 添加上下边距
            }
            QScrollBar::handle:horizontal {
                background: #90A4AE;
                border-radius: 6px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #607D8B;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
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
        self.ax.set_facecolor('#F5F7FA')
        self.ax.set_title('贪吃蛇训练得分趋势', fontsize=10, fontweight='bold', color=PRIMARY_COLOR, pad=15)
        self.ax.set_xlabel('蛇的出场编号', fontsize=11, color=SECONDARY_COLOR, labelpad=10)
        self.ax.set_ylabel('得分', fontsize=11, color=SECONDARY_COLOR, labelpad=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlim(0, PLOT_VIEW_WIDTH)
        self.ax.set_ylim(0, 20)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
        self.ax.tick_params(colors=SECONDARY_COLOR, labelsize=10)
        # 调整图表布局，为坐标轴标签留出足够空间
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
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
        self.ax.set_facecolor('#F5F7FA')
        
        # 计算显示范围
        end_pos = self.scroll_pos + PLOT_VIEW_WIDTH
        display_x = self.x_data[self.scroll_pos:end_pos]
        display_y = self.y_data[self.scroll_pos:end_pos]
        
        # 绘制折线图
        if display_x and display_y:
            self.ax.plot(display_x, display_y, 
                        color=ACCENT_COLOR, linewidth=2, marker='o', markersize=4, 
                        markerfacecolor=SUCCESS_COLOR, markeredgecolor='white', markeredgewidth=1)
            
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
                                   arrowprops=dict(arrowstyle='->', color=SUCCESS_COLOR, lw=1.5),
                                   fontsize=10, color=SUCCESS_COLOR, fontweight='bold')
        
        else:
            self.ax.set_xlim(0, PLOT_VIEW_WIDTH)
            self.ax.set_ylim(0, 20)
        
        # 重置样式
        self.ax.set_title('贪吃蛇训练得分趋势', fontsize=13, fontweight='bold', color=PRIMARY_COLOR, pad=15)
        self.ax.set_xlabel('蛇的出场编号', fontsize=11, color=SECONDARY_COLOR, labelpad=10)
        self.ax.set_ylabel('得分', fontsize=11, color=SECONDARY_COLOR, labelpad=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.tick_params(colors=SECONDARY_COLOR, labelsize=10)
        
        # 调整图表布局
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
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

# ====================== 5. 主窗口（集成最优成果保存逻辑 + 测试学习成果） ======================
class SnakeRLMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐍 强化学习贪吃蛇 - AI训练平台")
        # 增大窗口尺寸，给所有组件更多空间
        self.setFixedSize(1400, 1000)
        
        # 设置应用程序样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {LIGHT_BG};
            }}
            QLabel {{
                color: {PRIMARY_COLOR};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {PRIMARY_COLOR};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {PRIMARY_COLOR};
            }}
            QLineEdit {{
                border: 1px solid #B0BEC5;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }}
            QLineEdit:focus {{
                border: 2px solid {ACCENT_COLOR};
            }}
            QPushButton {{
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }}
        """)

        # 新增：测试模式标记
        self.test_mode = False  # 是否处于测试模式
        self.best_q_table = defaultdict(lambda: np.zeros(4))  # 存储最优Q表用于测试

        # 初始化核心组件
        self.game = SnakeGame()
        self.agent = QLearningAgent()
        self.best_score = self.agent.best_score  # 同步最优得分
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

        # ========== 左侧：游戏显示区 ==========
        left_widget = QWidget()
        left_widget.setFixedWidth(GAME_WIDTH + 60)  # 稍微增加宽度
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(left_widget)

        # 游戏显示标签
        self.game_label = QLabel()
        self.game_label.setFixedSize(GAME_WIDTH + 20, GAME_HEIGHT + 20)
        self.game_label.setStyleSheet(f"""
            border: 2px solid {PRIMARY_COLOR};
            border-radius: 10px;
            background: black;
            padding: 10px;
        """)

        # 游戏状态信息卡片 - 重新设计布局
        status_card = QFrame()
        status_card.setFixedHeight(240)  # 稍微增加高度
        status_card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 2px solid {PRIMARY_COLOR};
                border-radius: 10px;
                padding: 18px;
            }}
        """)

        status_layout = QVBoxLayout(status_card)
        status_layout.setSpacing(22)
        status_layout.setContentsMargins(10, 10, 10, 10)

        # 第一行：得分信息
        score_row = QWidget()
        score_layout = QHBoxLayout(score_row)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setSpacing(20) # 减少间距让框有更多空间
        score_layout.setAlignment(Qt.AlignCenter)

        # 当前得分（直接创建并保存引用）
        current_score_widget = QWidget()
        current_score_widget.setMinimumWidth(150)  # 增加10%（原130 → 143 → 取整145）
        current_score_layout = QVBoxLayout(current_score_widget)
        current_score_layout.setContentsMargins(14, 14, 14, 14)  # 增加内边距
        current_score_layout.setSpacing(8)  # 稍微增加间距

        current_score_title = QLabel("当前得分")
        current_score_title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        current_score_title.setStyleSheet(f"color: {SECONDARY_COLOR};")
        current_score_title.setAlignment(Qt.AlignCenter)

        self.current_score_value = QLabel(f"{self.game.score}")
        self.current_score_value.setFont(QFont("Arial", 14,QFont.Weight.Bold))  # 减小5%（原20 → 19 → 取整18）
        self.current_score_value.setStyleSheet(f"color: {ACCENT_COLOR};")
        self.current_score_value.setAlignment(Qt.AlignCenter)

        current_score_layout.addWidget(current_score_title)
        current_score_layout.addWidget(self.current_score_value)

        # 最优得分（直接创建并保存引用）
        best_score_widget = QWidget()
        best_score_widget.setMinimumWidth(150)  # 加10%（原130 → 143 → 取整145）
        best_score_layout = QVBoxLayout(best_score_widget)
        best_score_layout.setContentsMargins(14, 14, 14, 14)  # 增加内边距
        best_score_layout.setSpacing(8)  # 稍微增加间距

        best_score_title = QLabel("历史最优")
        best_score_title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        best_score_title.setStyleSheet(f"color: {SECONDARY_COLOR};")
        best_score_title.setAlignment(Qt.AlignCenter)

        self.best_score_value = QLabel(f"{self.best_score}")
        self.best_score_value.setFont(QFont("Arial", 14,QFont.Weight.Bold))  # 减小5%（原20 → 19 → 取整18）
        self.best_score_value.setStyleSheet(f"color: {SUCCESS_COLOR};")
        self.best_score_value.setAlignment(Qt.AlignCenter)

        best_score_layout.addWidget(best_score_title)
        best_score_layout.addWidget(self.best_score_value)

        # 训练进度（直接创建并保存引用）
        progress_widget = QWidget()
        progress_widget.setMinimumWidth(150)  # 增加10%（原155 → 170.5 → 取整175）
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(14, 14, 14, 14)  # 增加内边距
        progress_layout.setSpacing(8)  # 稍微增加间距

        progress_title = QLabel("训练进度")
        progress_title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        progress_title.setStyleSheet(f"color: {SECONDARY_COLOR};")
        progress_title.setAlignment(Qt.AlignCenter)

        self.progress_value = QLabel(f"{self.current_episode}/{self.total_episodes}")
        self.progress_value.setFont(QFont("Arial", 14, QFont.Weight.Bold))  # 减小5%（原14 → 13.3 → 取整12）
        self.progress_value.setStyleSheet(f"color: {WARNING_COLOR};")
        self.progress_value.setAlignment(Qt.AlignCenter)

        progress_layout.addWidget(progress_title)
        progress_layout.addWidget(self.progress_value)

        # 为每个信息框添加样式
        for widget in [current_score_widget, best_score_widget, progress_widget]:
            widget.setStyleSheet(f"""
                QWidget {{
                    background-color: {LIGHT_BG};
                    border-radius: 10px;
                    border: 1px solid #D0D0D0;
                }}
            """)

        score_layout.addWidget(current_score_widget)
        score_layout.addWidget(best_score_widget)
        score_layout.addWidget(progress_widget)
        score_layout.setStretch(0, 1)
        score_layout.setStretch(1, 1)
        score_layout.setStretch(2, 1)
        #score_layout.addStretch()
        
        # 第二行：最优参数
        params_row = QWidget()
        params_layout = QVBoxLayout(params_row)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(20)  # 减少间距

        best_params_title = QLabel("📊 最优参数记录")
        best_params_title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))  # 字体从11调整为10
        best_params_title.setStyleSheet(f"color: {PRIMARY_COLOR}; background: transparent; padding: 4px 8px;")
        best_params_title.setContentsMargins(0, 0, 0, 6)  # 底部留 6px 空隙，避免覆盖到徽章
        #best_params_title.setStyleSheet(f"color: {PRIMARY_COLOR};")

        best_params_container = QWidget()
        best_params_container.setStyleSheet("background: transparent;")
        best_params_container_layout = QHBoxLayout(best_params_container)
        best_params_container_layout.setContentsMargins(0, 0, 0, 0)
        best_params_container_layout.setSpacing(30)  # 减少间距
        best_params_container_layout.setAlignment(Qt.AlignCenter)  # 居中展示徽章

        self.best_alpha_value = self.create_param_badge(f"α={self.agent.best_params['alpha']:.2f}", "#E3F2FD", PRIMARY_COLOR)
        self.best_gamma_value = self.create_param_badge(f"γ={self.agent.best_params['gamma']:.2f}", "#E8F5E9", PRIMARY_COLOR)
        self.best_epsilon_value = self.create_param_badge(f"ε={self.agent.best_params['epsilon']:.2f}", "#FFF3E0", PRIMARY_COLOR)

        best_params_container_layout.addWidget(self.best_alpha_value)
        best_params_container_layout.addWidget(self.best_gamma_value)
        best_params_container_layout.addWidget(self.best_epsilon_value)
        #best_params_container_layout.addStretch()

        params_layout.addWidget(best_params_title)
        params_layout.addWidget(best_params_container)
        
        # 添加到状态卡片
        status_layout.addWidget(score_row)
        status_layout.addWidget(params_row)
        status_layout.addStretch()

        # 添加到左侧布局
        left_layout.addWidget(self.game_label)
        left_layout.addWidget(status_card)

        # ========== 右侧：控制面板 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(20)
        main_layout.addWidget(right_widget)

        # ---------- 子布局1：参数调节组 ----------
        param_group = QGroupBox("⚙️ 强化学习参数调节")
        param_group.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(15)
        
        # 参数表单布局
        param_form_layout = QFormLayout()
        param_form_layout.setSpacing(12)
        param_form_layout.setLabelAlignment(Qt.AlignRight)
        
        # 创建参数行
        params = [
            ("运行速度(FPS) [1-60]:", "fps", str(DEFAULT_FPS), QIntValidator(MIN_FPS, MAX_FPS)),
            ("学习率α [0.01-1.0]:", "alpha", f"{DEFAULT_ALPHA:.2f}", QDoubleValidator(MIN_ALPHA, MAX_ALPHA, 2)),
            ("折扣因子γ [0.01-1.0]:", "gamma", f"{DEFAULT_GAMMA:.2f}", QDoubleValidator(MIN_GAMMA, MAX_GAMMA, 2)),
            ("探索率ε [0.01-1.0]:", "epsilon", f"{DEFAULT_EPSILON:.2f}", QDoubleValidator(MIN_EPSILON, MAX_EPSILON, 2)),
            ("训练总轮次 [100-5000]:", "episodes", str(DEFAULT_EPISODES), QIntValidator(MIN_EPISODES, MAX_EPISODES))
        ]
        
        self.param_edits = {}
        for label_text, param_name, default_value, validator in params:
            label = QLabel(label_text)
            label.setFont(QFont("Microsoft YaHei", 9))
            edit = QLineEdit(default_value)
            edit.setFixedWidth(120)
            edit.setValidator(validator)
            edit.setFont(QFont("Microsoft YaHei", 9))
            edit.setStyleSheet("padding: 6px;")
            self.param_edits[param_name] = edit
            param_form_layout.addRow(label, edit)
        
        param_layout.addLayout(param_form_layout)
        
        # 参数操作按钮
        param_buttons_layout = QHBoxLayout()
        param_buttons_layout.setSpacing(15)
        
        self.confirm_btn = self.create_button("✅ 确认修改", ACCENT_COLOR, "#2980B9")
        self.confirm_btn.clicked.connect(self.confirm_params)
        
        self.cancel_btn = self.create_button("❌ 取消修改", "#7F8C8D", "#95A5A6")
        self.cancel_btn.clicked.connect(self.cancel_params)
        
        self.use_best_btn = self.create_button("🚀 使用最优成果训练", "#9B59B6", "#8E44AD")
        self.use_best_btn.clicked.connect(self.use_best_achievements)
        
        param_buttons_layout.addWidget(self.confirm_btn)
        param_buttons_layout.addWidget(self.cancel_btn)
        param_buttons_layout.addWidget(self.use_best_btn)
        param_buttons_layout.addStretch()
        
        param_layout.addLayout(param_buttons_layout)
        
        right_layout.addWidget(param_group)

        # ---------- 子布局2：功能按钮组 ----------
        btn_group = QGroupBox("🎮 控制面板")
        btn_group.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        
        btn_layout = QGridLayout(btn_group)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(15, 15, 15, 15)
        
        # 创建按钮
        self.pause_btn = self.create_button("⏸️ 暂停", ACCENT_COLOR, "#2980B9", height=50)
        self.pause_btn.clicked.connect(self.toggle_pause)
        
        self.restart_btn = self.create_button("🔄 重新开始", "#F39C12", "#E67E22", height=50)
        self.restart_btn.clicked.connect(self.restart_training)
        
        self.test_btn = self.create_button("🧪 测试学习成果", "#00BCD4", "#0097A7", height=50)
        self.test_btn.clicked.connect(self.start_test_mode)
        
        self.save_best_btn = self.create_button("💾 保存最优成果", "#9B59B6", "#8E44AD", height=50)
        self.save_best_btn.clicked.connect(self.manual_save_best)
        
        self.exit_btn = self.create_button("🚪 安全退出", WARNING_COLOR, "#C0392B", height=50)
        self.exit_btn.clicked.connect(self.safe_exit)
        
        # 添加到网格布局（2行3列）
        btn_layout.addWidget(self.pause_btn, 0, 0)
        btn_layout.addWidget(self.restart_btn, 0, 1)
        btn_layout.addWidget(self.test_btn, 0, 2)
        btn_layout.addWidget(self.save_best_btn, 1, 0, 1, 2)
        btn_layout.addWidget(self.exit_btn, 1, 2)
        
        # 设置列宽比例
        for i in range(3):
            btn_layout.setColumnStretch(i, 1)
        
        right_layout.addWidget(btn_group)

        # ---------- 子布局3：智能自动滑动折线图 ----------
        plot_group = QGroupBox("📈 训练得分趋势图")
        plot_group.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(10, 15, 10, 15)  # 增加底部边距
        
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

    def create_info_box(self, title, value, color, font_size):
        """创建信息显示框"""
        box = QFrame()
        box.setStyleSheet(f"""
            QFrame {{
                background-color: {color}20;
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(box)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {SECONDARY_COLOR};")
        title_label.setAlignment(Qt.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Microsoft YaHei", font_size, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color};")
        value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        return box

    def create_param_badge(self, text, bg_color, text_color):
        """创建参数徽章 — 更宽更高且可水平伸缩，避免被裁切"""
        badge = QLabel(text)
        # 字体使用跨平台稳定的字体，字号略大
        badge.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        badge.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                padding: 8px 16px;            /* 更舒适的内边距 */
                border-radius: 8px;
                border: 1px solid {PRIMARY_COLOR}30;
            }}
        """)
        badge.setAlignment(Qt.AlignCenter)
        badge.setMinimumHeight(28)                 # 增高，避免被垂直裁切
        badge.setMinimumWidth(145)                 # 略宽一些
        badge.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # 水平方向可伸缩以居中对齐
        badge.setContentsMargins(0, 0, 0, 0)
        return badge




    def create_button(self, text, color, hover_color, width=None, height=40):
        """创建统一风格的按钮"""
        btn = QPushButton(text)
        if width:
            btn.setFixedWidth(width)
        btn.setFixedHeight(height)
        btn.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 15px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {color};
                padding: 9px 14px 7px 16px;
            }}
        """)
        return btn

    # ---------- 新增：使用最优成果训练按钮逻辑 ----------
    def use_best_achievements(self):
        """加载最优成果（Q表+参数）并用于训练"""
        try:
            # 前置校验：检查最优文件是否存在
            missing_files = []
            if not os.path.exists(BEST_Q_TABLE_PATH):
                missing_files.append("最优Q表文件")
            if not os.path.exists(BEST_PARAMS_PATH):
                missing_files.append("最优参数文件")
            
            if missing_files:
                QMessageBox.warning(self, "提示", f"以下最优成果文件缺失：{', '.join(missing_files)}\n请先完成至少一次最优成果保存！")
                return
            
            # 1. 加载最优Q表
            with open(BEST_Q_TABLE_PATH, 'rb') as f:
                best_q_table_dict = pickle.load(f)
            self.agent.q_table = defaultdict(lambda: np.zeros(4), best_q_table_dict)
            
            # 2. 加载最优参数
            with open(BEST_PARAMS_PATH, 'rb') as f:
                best_params = pickle.load(f)
            
            # 校验参数完整性
            required_params = ["alpha", "gamma", "epsilon"]
            if not all(p in best_params for p in required_params):
                QMessageBox.warning(self, "参数异常", "最优参数文件格式错误，缺少必要参数！")
                return
            
            self.agent.alpha = best_params["alpha"]
            self.agent.gamma = best_params["gamma"]
            self.agent.epsilon = best_params["epsilon"]
            
            # 3. 同步更新界面输入框
            self.param_edits["alpha"].setText(f"{self.agent.alpha:.2f}")
            self.param_edits["gamma"].setText(f"{self.agent.gamma:.2f}")
            self.param_edits["epsilon"].setText(f"{self.agent.epsilon:.2f}")
            
            # 4. 更新最优参数显示
            self.best_alpha_value.setText(f"α={self.agent.alpha:.2f}")
            self.best_gamma_value.setText(f"γ={self.agent.gamma:.2f}")
            self.best_epsilon_value.setText(f"ε={self.agent.epsilon:.2f}")
            
            # 5. 提示用户
            QMessageBox.information(self, "成功", 
                                  f"✅ 已加载最优成果！\n\n📊 Q表经验数：{len(self.agent.q_table)}\n⚙️ 最优参数：α={self.agent.alpha:.2f}, γ={self.agent.gamma:.2f}, ε={self.agent.epsilon:.2f}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载最优成果失败：{str(e)}")

    # ---------- 新增：测试学习成果按钮逻辑 ----------
    def start_test_mode(self):
        """启动测试模式：加载最优Q表，停止更新，重新开始游戏"""
        # 校验最优Q表文件是否存在
        if not os.path.exists(BEST_Q_TABLE_PATH):
            QMessageBox.warning(self, "提示", "未找到最优Q表文件！请先完成至少一次最优成果保存。")
            return
        
        # 加载最优Q表
        with open(BEST_Q_TABLE_PATH, 'rb') as f:
            best_q_table_dict = pickle.load(f)
        self.best_q_table = defaultdict(lambda: np.zeros(4), best_q_table_dict)
        
        # 设置测试模式
        self.test_mode = True
        self.paused = False  # 确保测试时游戏不暂停
        self.current_episode = 0  # 重置轮次显示
        self.progress_value.setText(f"测试模式/{self.total_episodes}")  # 更新进度显示
        self.test_btn.setText("🛑 停止测试")  # 按钮文字切换
        self.test_btn.clicked.disconnect()
        self.test_btn.clicked.connect(self.stop_test_mode)
        
        # 重新开始游戏
        self.state = self.game.reset()
        self.current_score_value.setText(f"{self.game.score}")
        # 清空测试模式下的得分图表
        self.auto_scroll_plot.clear_plot()
        
        QMessageBox.information(self, "测试模式", "🧪 已进入测试模式！\n\n小蛇将使用最优Q表循迹，Q表停止更新。")

    def stop_test_mode(self):
        """停止测试模式，恢复正常训练"""
        self.test_mode = False
        self.test_btn.setText("🧪 测试学习成果")
        self.test_btn.clicked.disconnect()
        self.test_btn.clicked.connect(self.start_test_mode)
        self.progress_value.setText(f"{self.current_episode}/{self.total_episodes}")
        QMessageBox.information(self, "测试模式", "已退出测试模式，可恢复正常训练。")

    def get_test_action(self, state):
        """测试模式下：仅使用最优Q表选择安全动作"""
        # 获取安全动作列表
        safe_actions = self.agent.get_safe_actions(state)
        if safe_actions:
            # 仅利用最优Q表，不探索
            safe_q_values = [self.best_q_table[state][a] for a in safe_actions]
            max_q = max(safe_q_values)
            best_actions = [a for a, q in zip(safe_actions, safe_q_values) if q == max_q]
            action = random.choice(best_actions)
            return action, "测试(最优Q表)"
        else:
            # 无安全动作时随机选
            action = random.choice([0, 1, 2, 3])
            return action, "测试(危险)"

    # ---------- 手动保存最优成果按钮逻辑 ----------
    def manual_save_best(self):
        """手动保存当前Q表和参数为最优成果"""
        try:
            # 1. 保存当前Q表为最优Q表
            best_q_table_dict = dict(self.agent.q_table)
            with open(BEST_Q_TABLE_PATH, 'wb') as f:
                pickle.dump(best_q_table_dict, f)
            
            # 2. 保存当前参数为最优参数
            self.agent.best_params = {
                "alpha": self.agent.alpha,
                "gamma": self.agent.gamma,
                "epsilon": self.agent.epsilon,
                "score": self.game.score  # 记录当前得分
            }
            with open(BEST_PARAMS_PATH, 'wb') as f:
                pickle.dump(self.agent.best_params, f)
            
            # 3. 更新最优得分（如果当前得分更高）
            if self.game.score > self.agent.best_score:
                self.agent.best_score = self.game.score
                self.agent.save_best_score()
                self.best_score_value.setText(f"{self.agent.best_score}")
            
            # 4. 更新最优参数显示
            self.best_alpha_value.setText(f"α={self.agent.alpha:.2f}")
            self.best_gamma_value.setText(f"γ={self.agent.gamma:.2f}")
            self.best_epsilon_value.setText(f"ε={self.agent.epsilon:.2f}")
            
            # 5. 提示用户
            QMessageBox.information(self, "成功", 
                                  f"💾 已手动保存当前成果为最优！\n\n📊 Q表经验数：{len(self.agent.q_table)}\n🏆 当前得分：{self.game.score}\n⚙️ 参数：α={self.agent.alpha:.2f}, γ={self.agent.gamma:.2f}, ε={self.agent.epsilon:.2f}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"手动保存最优成果失败：{str(e)}")

    # ---------- 参数修改确认/取消逻辑 ----------
    def confirm_params(self):
        """确认修改参数"""
        try:
            # 1. 读取输入值
            new_fps = int(self.param_edits["fps"].text())
            new_alpha = float(self.param_edits["alpha"].text())
            new_gamma = float(self.param_edits["gamma"].text())
            new_epsilon = float(self.param_edits["epsilon"].text())
            new_episodes = int(self.param_edits["episodes"].text())
            
            # 2. 校验范围（防止validator失效）
            if not (MIN_FPS <= new_fps <= MAX_FPS):
                raise ValueError(f"FPS必须在{MIN_FPS}-{MAX_FPS}之间")
            if not (MIN_ALPHA <= new_alpha <= MAX_ALPHA):
                raise ValueError(f"学习率α必须在{MIN_ALPHA}-{MAX_ALPHA}之间")
            if not (MIN_GAMMA <= new_gamma <= MAX_GAMMA):
                raise ValueError(f"折扣因子γ必须在{MIN_GAMMA}-{MAX_GAMMA}之间")
            if not (MIN_EPSILON <= new_epsilon <= MAX_EPSILON):
                raise ValueError(f"探索率ε必须在{MIN_EPSILON}-{MAX_EPSILON}之间")
            if not (MIN_EPISODES <= new_episodes <= MAX_EPISODES):
                raise ValueError(f"训练轮次必须在{MIN_EPISODES}-{MAX_EPISODES}之间")
            
            # 3. 更新参数
            self.agent.alpha = new_alpha
            self.agent.gamma = new_gamma
            self.agent.epsilon = new_epsilon
            self.total_episodes = new_episodes
            
            # 4. 更新定时器（FPS）
            self.timer.setInterval(int(1000/new_fps))
            
            # 5. 保存新参数为原始值
            self.original_params = {
                "fps": new_fps,
                "alpha": new_alpha,
                "gamma": new_gamma,
                "epsilon": new_epsilon,
                "episodes": new_episodes
            }
            
            # 6. 提示用户
            QMessageBox.information(self, "参数更新成功", 
                                  f"✅ 参数已更新！\n\n🎮 FPS: {new_fps}\n🧠 α: {new_alpha:.2f}\n🎯 γ: {new_gamma:.2f}\n🔍 ε: {new_epsilon:.2f}\n📈 训练轮次: {new_episodes}")
            
        except ValueError as e:
            QMessageBox.warning(self, "参数错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新参数失败：{str(e)}")

    def cancel_params(self):
        """取消修改参数，恢复原始值"""
        # 恢复输入框值
        self.param_edits["fps"].setText(str(self.original_params["fps"]))
        self.param_edits["alpha"].setText(f"{self.original_params['alpha']:.2f}")
        self.param_edits["gamma"].setText(f"{self.original_params['gamma']:.2f}")
        self.param_edits["epsilon"].setText(f"{self.original_params['epsilon']:.2f}")
        self.param_edits["episodes"].setText(str(self.original_params["episodes"]))
        
        # 提示用户
        QMessageBox.information(self, "已取消", "参数已恢复为修改前的值")

    # ---------- 暂停/继续逻辑 ----------
    def toggle_pause(self):
        """暂停/继续训练"""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.setText("▶️ 继续")
            self.pause_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {SUCCESS_COLOR};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 15px;
                }}
                QPushButton:hover {{
                    background-color: #27AE60;
                }}
            """)
        else:
            self.pause_btn.setText("⏸️ 暂停")
            self.pause_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ACCENT_COLOR};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 15px;
                }}
                QPushButton:hover {{
                    background-color: #2980B9;
                }}
            """)

    # ---------- 重新开始训练逻辑 ----------
    def restart_training(self):
        """重新开始训练（重置游戏和轮次）"""
        self.game.reset()
        self.current_episode = 0
        self.progress_value.setText(f"{self.current_episode}/{self.total_episodes}")
        self.current_score_value.setText(f"{self.game.score}")
        self.auto_scroll_plot.clear_plot()
        self.paused = False
        self.pause_btn.setText("⏸️ 暂停")
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 15px;
            }}
            QPushButton:hover {{
                background-color: #2980B9;
            }}
        """)
        QMessageBox.information(self, "重新开始", "🔄 已重置游戏，训练重新开始！")

    # ---------- 安全退出逻辑 ----------
    def safe_exit(self):
        """安全退出（保存当前Q表）"""
        reply = QMessageBox.question(self, "确认退出", 
                                    "确定要退出程序吗？\n当前Q表将自动保存。",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # 保存当前Q表
                self.agent.save_q_table()
                # 提示用户
                QMessageBox.information(self, "退出", f"📊 已保存当前Q表（经验数：{len(self.agent.q_table)}），即将退出程序。")
                # 退出程序
                QApplication.quit()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存Q表失败：{str(e)}\n程序将强制退出。")
                QApplication.quit()

    # ---------- 游戏更新逻辑（核心） ----------
    def update_game(self):
        if self.paused or self.game.game_over:
            if self.game.game_over:
                # 测试模式下游戏结束处理
                if self.test_mode:
                    # 测试模式不保存Q表，不更新最优成果
                    self.auto_scroll_plot.update_data(self.current_episode, self.game.score)
                    self.current_episode += 1
                    self.progress_value.setText(f"测试模式/{self.total_episodes}")
                    self.state = self.game.reset()
                    return
                # 正常训练模式下的游戏结束处理
                self.auto_scroll_plot.update_data(self.current_episode, self.game.score)
                # 保存最优成果
                self.agent.save_best_q_table(self.game.score)
                # 更新最优得分显示
                self.best_score = self.agent.best_score
                self.best_score_value.setText(f"{self.best_score}")
                # 保存当前Q表
                self.agent.save_q_table()
                # 重置游戏
                self.current_episode += 1
                self.progress_value.setText(f"{self.current_episode}/{self.total_episodes}")
                if self.current_episode >= self.total_episodes:
                    QMessageBox.information(self, "训练完成", f"🎉 已完成{self.total_episodes}轮训练！")
                    self.paused = True
                    self.pause_btn.setText("▶️ 继续")
                    self.pause_btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {SUCCESS_COLOR};
                            color: white;
                            border: none;
                            border-radius: 8px;
                            padding: 8px 15px;
                        }}
                        QPushButton:hover {{
                            background-color: #27AE60;
                        }}
                    """)
                self.state = self.game.reset()
            return

        # 选择动作：区分测试模式/训练模式
        if self.test_mode:
            action, _ = self.get_test_action(self.state)
        else:
            action, _ = self.agent.choose_action(self.state)
        
        # 执行动作
        next_state, reward, game_over, eat_food, collision_reason = self.game.step(action)
        
        # 仅训练模式更新Q表，测试模式跳过
        if not self.test_mode:
            self.agent.update_q_table(self.state, action, reward, next_state)
        
        # 更新状态和显示
        self.state = next_state
        self.current_score_value.setText(f"{self.game.score}")
        # 渲染游戏画面
        q_image = self.game.render()
        self.game_label.setPixmap(QPixmap.fromImage(q_image))

# ====================== 6. 程序入口 ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式，更现代化
    
    # 设置应用程序图标和字体
    app.setFont(QFont("Microsoft YaHei", 9))
    
    window = SnakeRLMainWindow()
    window.show()
    sys.exit(app.exec_())