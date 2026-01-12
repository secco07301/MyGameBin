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
    QLineEdit, QPushButton, QLabel, QTextEdit, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QTextCursor, QIntValidator, QDoubleValidator

# Pygame 仅用于游戏渲染
import pygame

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

# ====================== 2. 强化学习智能体（无修改） ======================
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
        q_table_dict = dict(self.q_table)
        with open(Q_TABLE_PATH, 'wb') as f:
            pickle.dump(q_table_dict, f)
        return len(self.q_table)

    def load_q_table(self):
        if os.path.exists(Q_TABLE_PATH):
            with open(Q_TABLE_PATH, 'rb') as f:
                q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(4), q_table_dict)
            return len(self.q_table)
        return 0

    def reset(self):
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = DEFAULT_ALPHA
        self.gamma = DEFAULT_GAMMA
        self.epsilon = DEFAULT_EPSILON
        if os.path.exists(Q_TABLE_PATH):
            os.remove(Q_TABLE_PATH)
        return 0

# ====================== 3. 贪吃蛇游戏核心（无修改） ======================
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

# ====================== 4. PyQt5主窗口（区分中文/数字字体大小） ======================
class SnakeRLMainWindow(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("强化学习贪吃蛇")
        self.setFixedSize(1200, 700)

        # 初始化核心组件
        self.game = SnakeGame()
        self.agent = QLearningAgent()
        self.best_score = 0
        self.current_episode = 0
        self.total_episodes = DEFAULT_EPISODES
        self.paused = False

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setCentralWidget(main_widget)

        # ========== 左侧：游戏显示区 ==========
        self.game_label = QLabel()
        self.game_label.setFixedSize(GAME_WIDTH, GAME_HEIGHT)
        self.game_label.setStyleSheet("border: 3px solid #333; background: black;")
        main_layout.addWidget(self.game_label)

        # ========== 右侧：控制面板 ==========
        control_widget = QWidget()
        control_widget.setFixedSize(650, 650)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(20)
        main_layout.addWidget(control_widget)

        # ---------- 子布局1：参数调节组（中文30%，数字50%） ----------
        param_group = QGroupBox("强化学习参数调节")
        param_group.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))  # 标题字体不变
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(15)

        # 1. FPS参数行：中文标签4号(30%)，输入框数字7号(50%)
        fps_label = QLabel("运行速度(FPS) [1-60]:")
        fps_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签30%
        self.fps_edit = QLineEdit(str(DEFAULT_FPS))
        self.fps_edit.setFixedWidth(100)
        fps_validator = QIntValidator(MIN_FPS, MAX_FPS, self)
        self.fps_edit.setValidator(fps_validator)
        self.fps_edit.setFont(QFont("Microsoft YaHei", 7))  # 数字50%
        self.fps_edit.textChanged.connect(self.on_fps_changed)
        param_layout.addRow(fps_label, self.fps_edit)

        # 2. 学习率α参数行
        alpha_label = QLabel("学习率α [0.01-1.0]:")
        alpha_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签30%
        self.alpha_edit = QLineEdit(f"{DEFAULT_ALPHA:.2f}")
        self.alpha_edit.setFixedWidth(100)
        alpha_validator = QDoubleValidator(MIN_ALPHA, MAX_ALPHA, 2, self)
        alpha_validator.setNotation(QDoubleValidator.StandardNotation)
        self.alpha_edit.setValidator(alpha_validator)
        self.alpha_edit.setFont(QFont("Microsoft YaHei", 7))  # 数字50%
        self.alpha_edit.textChanged.connect(self.on_alpha_changed)
        param_layout.addRow(alpha_label, self.alpha_edit)

        # 3. 折扣因子γ参数行
        gamma_label = QLabel("折扣因子γ [0.01-1.0]:")
        gamma_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签30%
        self.gamma_edit = QLineEdit(f"{DEFAULT_GAMMA:.2f}")
        self.gamma_edit.setFixedWidth(100)
        gamma_validator = QDoubleValidator(MIN_GAMMA, MAX_GAMMA, 2, self)
        gamma_validator.setNotation(QDoubleValidator.StandardNotation)
        self.gamma_edit.setValidator(gamma_validator)
        self.gamma_edit.setFont(QFont("Microsoft YaHei", 7))  # 数字50%
        self.gamma_edit.textChanged.connect(self.on_gamma_changed)
        param_layout.addRow(gamma_label, self.gamma_edit)

        # 4. 探索率ε参数行
        epsilon_label = QLabel("探索率ε [0.01-1.0]:")
        epsilon_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签30%
        self.epsilon_edit = QLineEdit(f"{DEFAULT_EPSILON:.2f}")
        self.epsilon_edit.setFixedWidth(100)
        epsilon_validator = QDoubleValidator(MIN_EPSILON, MAX_EPSILON, 2, self)
        epsilon_validator.setNotation(QDoubleValidator.StandardNotation)
        self.epsilon_edit.setValidator(epsilon_validator)
        self.epsilon_edit.setFont(QFont("Microsoft YaHei", 7))  # 数字50%
        self.epsilon_edit.textChanged.connect(self.on_epsilon_changed)
        param_layout.addRow(epsilon_label, self.epsilon_edit)

        # 5. 训练轮次参数行
        episode_label = QLabel("训练总轮次 [100-5000]:")
        episode_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签30%
        self.episode_edit = QLineEdit(str(DEFAULT_EPISODES))
        self.episode_edit.setFixedWidth(100)
        episode_validator = QIntValidator(MIN_EPISODES, MAX_EPISODES, self)
        self.episode_edit.setValidator(episode_validator)
        self.episode_edit.setFont(QFont("Microsoft YaHei", 7))  # 数字50%
        self.episode_edit.textChanged.connect(self.on_episode_changed)
        param_layout.addRow(episode_label, self.episode_edit)

        control_layout.addWidget(param_group)

        # ---------- 子布局2：功能按钮组（无修改） ----------
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setFixedSize(120, 45)
        self.pause_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.pause_btn.setStyleSheet("""
            QPushButton {background-color: #2196F3; color: white; border: none; border-radius: 8px; font-size: 12px;}
            QPushButton:hover {background-color: #1976D2;}
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)

        self.restart_btn = QPushButton("重新开始")
        self.restart_btn.setFixedSize(120, 45)
        self.restart_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.restart_btn.setStyleSheet("""
            QPushButton {background-color: #FF9800; color: white; border: none; border-radius: 8px; font-size: 12px;}
            QPushButton:hover {background-color: #F57C00;}
        """)
        self.restart_btn.clicked.connect(self.restart_training)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.setFixedSize(120, 45)
        self.exit_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.exit_btn.setStyleSheet("""
            QPushButton {background-color: #F44336; color: white; border: none; border-radius: 8px; font-size: 12px;}
            QPushButton:hover {background-color: #D32F2F;}
        """)
        self.exit_btn.clicked.connect(self.safe_exit)

        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.restart_btn)
        btn_layout.addWidget(self.exit_btn)
        control_layout.addLayout(btn_layout)

        # ---------- 子布局3：状态信息组（无修改） ----------
        # ---------- 子布局3：状态信息组（标题14号不变，中文标签改为7号） ----------
        status_group = QGroupBox("训练状态信息")
        status_group.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))  # 标题字体保持14号加粗不变
        status_layout = QFormLayout(status_group)
        status_layout.setSpacing(8)

        # 数字显示的Label字体保持原有设置（你没提修改，这里维持默认）
        self.current_score_label = QLabel(f"{self.game.score}")
        self.current_score_label.setFont(QFont("Microsoft YaHei", 6))  # 原数值字体，保持不变
        self.best_score_label = QLabel(f"{self.best_score}")
        self.best_score_label.setFont(QFont("Microsoft YaHei", 6))  # 原数值字体，保持不变
        self.exp_count_label = QLabel(f"{len(self.agent.q_table)}")
        self.exp_count_label.setFont(QFont("Microsoft YaHei", 6))  # 原数值字体，保持不变
        self.current_epsilon_label = QLabel(f"{self.agent.epsilon:.2f}")
        self.current_epsilon_label.setFont(QFont("Microsoft YaHei", 6))  # 原数值字体，保持不变
        self.episode_progress_label = QLabel(f"{self.current_episode}/{self.total_episodes}")
        self.episode_progress_label.setFont(QFont("Microsoft YaHei", 6))  # 原数值字体，保持不变

        # 中文标签拆成独立QLabel，设置为7号字体
        score_label = QLabel("当前得分:")
        score_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签改为7号
        best_score_label = QLabel("最优得分:")
        best_score_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签改为7号
        exp_label = QLabel("经验总数:")
        exp_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签改为7号
        epsilon_label = QLabel("当前探索率ε:")
        epsilon_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签改为7号
        progress_label = QLabel("训练进度:")
        progress_label.setFont(QFont("Microsoft YaHei", 7))  # 中文标签改为7号

        # 布局添加行（使用设置好字体的中文Label）
        status_layout.addRow(score_label, self.current_score_label)
        status_layout.addRow(best_score_label, self.best_score_label)
        status_layout.addRow(exp_label, self.exp_count_label)
        status_layout.addRow(epsilon_label, self.current_epsilon_label)
        status_layout.addRow(progress_label, self.episode_progress_label)

        control_layout.addWidget(status_group)

        # ---------- 子布局4：日志显示组（无修改） ----------
        log_group = QGroupBox("训练日志")
        log_group.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setFixedHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.log_text.setStyleSheet("""
            QTextEdit {background-color: #222; color: #EEE; border: 2px solid #555; border-radius: 5px; font-size: 12px;}
        """)
        log_layout.addWidget(self.log_text)

        control_layout.addWidget(log_group)

        # ========== 定时器 ==========
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(int(1000/DEFAULT_FPS))

        # ========== 日志信号连接 ==========
        self.log_signal.connect(self.add_log)

        # 初始日志
        self.add_log(f"程序启动 | 加载经验数：{len(self.agent.q_table)}")
        self.add_log(f"默认参数：FPS={DEFAULT_FPS}, α={DEFAULT_ALPHA}, γ={DEFAULT_GAMMA}, ε={DEFAULT_EPSILON}, 训练轮次={DEFAULT_EPISODES}")

        # 初始游戏状态
        self.state = self.game.reset()

    # ---------- 参数输入框回调函数（无修改） ----------
    def on_fps_changed(self, text):
        try:
            fps = int(text) if text else DEFAULT_FPS
            if MIN_FPS <= fps <= MAX_FPS:
                self.timer.setInterval(int(1000/fps))
                self.add_log(f"FPS已更新为：{fps}")
            else:
                self.add_log(f"FPS输入非法（{text}），使用默认值：{DEFAULT_FPS}")
                self.fps_edit.setText(str(DEFAULT_FPS))
        except ValueError:
            self.add_log(f"FPS输入不是整数（{text}），使用默认值：{DEFAULT_FPS}")
            self.fps_edit.setText(str(DEFAULT_FPS))

    def on_alpha_changed(self, text):
        try:
            alpha = float(text) if text else DEFAULT_ALPHA
            if MIN_ALPHA <= alpha <= MAX_ALPHA:
                self.agent.alpha = alpha
                self.add_log(f"学习率α已更新为：{alpha:.2f}")
            else:
                self.add_log(f"学习率α输入非法（{text}），使用默认值：{DEFAULT_ALPHA:.2f}")
                self.alpha_edit.setText(f"{DEFAULT_ALPHA:.2f}")
        except ValueError:
            self.add_log(f"学习率α输入不是数字（{text}），使用默认值：{DEFAULT_ALPHA:.2f}")
            self.alpha_edit.setText(f"{DEFAULT_ALPHA:.2f}")

    def on_gamma_changed(self, text):
        try:
            gamma = float(text) if text else DEFAULT_GAMMA
            if MIN_GAMMA <= gamma <= MAX_GAMMA:
                self.agent.gamma = gamma
                self.add_log(f"折扣因子γ已更新为：{gamma:.2f}")
            else:
                self.add_log(f"折扣因子γ输入非法（{text}），使用默认值：{DEFAULT_GAMMA:.2f}")
                self.gamma_edit.setText(f"{DEFAULT_GAMMA:.2f}")
        except ValueError:
            self.add_log(f"折扣因子γ输入不是数字（{text}），使用默认值：{DEFAULT_GAMMA:.2f}")
            self.gamma_edit.setText(f"{DEFAULT_GAMMA:.2f}")

    def on_epsilon_changed(self, text):
        try:
            epsilon = float(text) if text else DEFAULT_EPSILON
            if MIN_EPSILON <= epsilon <= MAX_EPSILON:
                self.agent.epsilon = epsilon
                self.current_epsilon_label.setText(f"{epsilon:.2f}")
                self.add_log(f"探索率ε已更新为：{epsilon:.2f}")
            else:
                self.add_log(f"探索率ε输入非法（{text}），使用默认值：{DEFAULT_EPSILON:.2f}")
                self.epsilon_edit.setText(f"{DEFAULT_EPSILON:.2f}")
        except ValueError:
            self.add_log(f"探索率ε输入不是数字（{text}），使用默认值：{DEFAULT_EPSILON:.2f}")
            self.epsilon_edit.setText(f"{DEFAULT_EPSILON:.2f}")

    def on_episode_changed(self, text):
        try:
            episodes = int(text) if text else DEFAULT_EPISODES
            if MIN_EPISODES <= episodes <= MAX_EPISODES:
                self.total_episodes = episodes
                self.add_log(f"训练总轮次已更新为：{episodes}")
            else:
                self.add_log(f"训练轮次输入非法（{text}），使用默认值：{DEFAULT_EPISODES}")
                self.episode_edit.setText(str(DEFAULT_EPISODES))
        except ValueError:
            self.add_log(f"训练轮次输入不是整数（{text}），使用默认值：{DEFAULT_EPISODES}")
            self.episode_edit.setText(str(DEFAULT_EPISODES))

    # ---------- 原有功能函数（无修改） ----------
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("开始" if self.paused else "暂停")
        self.add_log(f"训练{'暂停' if self.paused else '继续'} | 当前ε={self.agent.epsilon:.2f}")

    def restart_training(self):
        self.game.reset()
        exp_count = self.agent.reset()
        self.current_episode = 0
        self.best_score = 0
        self.paused = False
        self.pause_btn.setText("暂停")
        
        self.fps_edit.setText(str(DEFAULT_FPS))
        self.alpha_edit.setText(f"{DEFAULT_ALPHA:.2f}")
        self.gamma_edit.setText(f"{DEFAULT_GAMMA:.2f}")
        self.epsilon_edit.setText(f"{DEFAULT_EPSILON:.2f}")
        self.episode_edit.setText(str(DEFAULT_EPISODES))
        
        self.update_status_labels()
        self.add_log("重新开始训练 | 清空所有经验 | 所有参数重置为默认值")

    def safe_exit(self):
        exp_count = self.agent.save_q_table()
        self.add_log(f"保存经验（{exp_count}条）并退出 | 最终ε={self.agent.epsilon:.2f}")
        QApplication.quit()

    def add_log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {text}"
        self.log_text.append(log_text)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def update_status_labels(self):
        self.current_score_label.setText(f"{self.game.score}")
        self.best_score_label.setText(f"{self.best_score}")
        self.exp_count_label.setText(f"{len(self.agent.q_table)}")
        self.current_epsilon_label.setText(f"{self.agent.epsilon:.2f}")
        self.episode_progress_label.setText(f"{self.current_episode}/{self.total_episodes}")

    def update_game(self):
        try:
            current_fps = int(self.fps_edit.text()) if self.fps_edit.text() else DEFAULT_FPS
        except ValueError:
            current_fps = DEFAULT_FPS

        if not self.paused and self.current_episode < self.total_episodes:
            action, action_type = self.agent.choose_action(self.state)
            next_state, reward, game_over, eat_food, collision_reason = self.game.step(action)

            if not game_over:
                self.agent.update_q_table(self.state, action, reward, next_state)

            if eat_food:
                self.add_log(f"吃到食物 | 当前得分：{self.game.score} | 动作类型：{action_type}（ε={self.agent.epsilon:.2f}）")
                if self.game.score > self.best_score:
                    self.best_score = self.game.score
                    self.add_log(f"刷新最优得分：{self.best_score}")

            if game_over:
                self.current_episode += 1
                self.add_log(f"第{self.current_episode}轮结束 | 得分：{self.game.score} | 原因：{collision_reason} | 当前ε={self.agent.epsilon:.2f}")
                
                if self.current_episode % 100 == 0:
                    exp_count = self.agent.save_q_table()
                    self.add_log(f"第{self.current_episode}轮 | 保存经验（{exp_count}条）")
                
                if self.current_episode > self.total_episodes * 0.8:
                    new_epsilon = max(MIN_EPSILON, self.agent.epsilon - 0.0001)
                    self.agent.epsilon = new_epsilon
                    self.epsilon_edit.setText(f"{new_epsilon:.2f}")
                    self.current_epsilon_label.setText(f"{new_epsilon:.2f}")
                
                self.state = self.game.reset()

            else:
                self.state = next_state

        self.update_status_labels()
        q_image = self.game.render()
        self.game_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        exp_count = self.agent.save_q_table()
        self.add_log(f"窗口关闭 | 保存经验（{exp_count}条）")
        event.accept()

# ====================== 5. 程序入口 ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)
    window = SnakeRLMainWindow()
    window.show()
    sys.exit(app.exec_())