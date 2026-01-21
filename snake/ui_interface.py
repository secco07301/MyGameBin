"""
自动循迹贪吃蛇 - 美观的UI界面（优化版）
UI界面的单独模块，包含所有界面相关的类和组件
"""

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 颜色主题配置 =====
THEME_COLORS = {
    'primary': '#2E86AB',      # 深蓝
    'secondary': '#A23B72',    # 紫红
    'success': '#06A77D',      # 绿色
    'warning': '#F77F00',      # 橙色
    'danger': '#D62828',       # 红色
    'light': '#F3F3F3',        # 浅灰
    'dark': '#2C3E50',         # 深灰
    'border': '#E0E0E0',       # 边界灰
}

# ===== 数据分析 matplotlib 绘图 =====
class GameStatisticsCanvas(FigureCanvas):
    """游戏统计数据可视化画布"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5.2, 3.2), tight_layout=False)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 数据存储
        self.times = []
        self.average_time = 0.0
        self.below_average = 0
        self.above_average = 0
        self.equal_average = 0
        
        # 图表样式配置
        self.fig.patch.set_facecolor('#FFFFFF')
        self.fig.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.90, wspace=0, hspace=0)
        self.setStyleSheet("background-color: white; border-radius: 6px; border: 1px solid #E0E0E0;")
        
        # 初始化坐标轴
        self._init_axes()
        self.setMinimumHeight(250)

    def _init_axes(self):
        """初始化坐标轴样式"""
        self.ax.set_xlabel("水果编号", fontsize=9, fontweight='bold', color='#2C3E50')
        self.ax.set_ylabel("耗时 (秒)", fontsize=9, fontweight='bold', color='#2C3E50')
        self.ax.set_title("每个水果的耗时趋势", fontsize=10, fontweight='bold', 
                         color='#2E86AB', pad=10)
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax.set_facecolor('#F9F9F9')

    def update_plot(self):
        """更新折线图"""
        self.ax.clear()
        
        if len(self.times) > 0:
            x = np.arange(1, len(self.times) + 1)
            y = np.array(self.times)
            
            # 动态平滑处理
            window = min(3, len(y)) if len(y) >= 1 else 1
            y_smooth = np.convolve(y, np.ones(window) / window, mode='same')
            
            # 绘制折线和数据点
            self.ax.plot(x, y_smooth, marker='o', color=THEME_COLORS['primary'], 
                        linewidth=2, markersize=5, label='耗时曲线', alpha=0.8)
            self.ax.fill_between(x, y_smooth, alpha=0.1, color=THEME_COLORS['primary'])
            
            # 计算统计数据
            self.average_time = np.mean(y)
            self.below_average = np.sum(y < self.average_time)
            self.above_average = np.sum(y > self.average_time)
            self.equal_average = np.sum(y == self.average_time)
            
            # 绘制平均线
            self.ax.axhline(y=self.average_time, color=THEME_COLORS['danger'], 
                           linestyle='--', linewidth=2, label=f'平均: {self.average_time:.3f}s', 
                           alpha=0.8)
            
            # 添加数值标注
            for i, t in enumerate(y):
                self.ax.text(x[i], t, f"{t:.2f}", fontsize=7, ha='center', 
                            va='bottom', color='#2C3E50')
            
            # 统计信息框
            stats_text = f"低: {self.below_average} | 高: {self.above_average} | 等: {self.equal_average}"
            self.ax.text(0.02, 0.97, stats_text, transform=self.ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4F8', 
                                 edgecolor=THEME_COLORS['primary'], linewidth=1, alpha=0.9))
            
            # 设置坐标轴范围
            self.ax.set_ylim(0, max(y) * 1.3 if max(y) > 0 else 1)
            self.ax.set_xlim(0.5, len(x) + 0.5)
            self.ax.legend(fontsize=8, loc='upper right', framealpha=0.95, edgecolor='gray')
        else:
            # 无数据状态
            self.ax.text(0.5, 0.5, '等待游戏开始...', transform=self.ax.transAxes,
                        fontsize=12, ha='center', va='center', color='#AAAAAA', 
                        fontweight='bold', style='italic')
            self.average_time = 0.0
            self.below_average = 0
            self.above_average = 0
            self.equal_average = 0
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 1)
        
        self._init_axes()
        self.draw()


# ===== 游戏画布 =====
class GameCanvas(QtWidgets.QWidget):
    """贪吃蛇游戏绘制区域"""
    
    def __init__(self, game_queue, game_width=600, game_height=400, block_size=20, parent=None):
        super().__init__(parent)
        self.setFixedSize(game_width, game_height)
        self.game_queue = game_queue
        self.block_size = block_size
        
        # 游戏状态
        self.snake = [(5, 5)]
        self.food = (10, 10)
        self.score = 0
        
        # 设置游戏画布样式
        self.setStyleSheet(f"""
            QWidget {{
                background-color: white;
                border: 3px solid {THEME_COLORS['primary']};
                border-radius: 8px;
            }}
        """)
        
        # 定时更新
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_state)
        self.timer.start(30)

    def update_state(self):
        """更新游戏状态"""
        while not self.game_queue.empty():
            data = self.game_queue.get()
            if data.get("update_snake"):
                self.snake = data["snake"]
                self.food = data["food"]
                self.score = data["score"]
        self.update()

    def paintEvent(self, event):
        """绘制游戏界面"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # 绘制网格背景
        painter.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
        
        # 绘制网格线
        pen = QtGui.QPen(QtGui.QColor(230, 230, 230), 1)
        pen.setStyle(QtCore.Qt.DotLine)
        painter.setPen(pen)
        
        grid_w = self.width() // self.block_size
        grid_h = self.height() // self.block_size
        for i in range(1, grid_w):
            painter.drawLine(i * self.block_size, 0, i * self.block_size, self.height())
        for i in range(1, grid_h):
            painter.drawLine(0, i * self.block_size, self.width(), i * self.block_size)
        
        # 绘制蛇身体
        for x, y in self.snake[:-1]:
            rect = QtCore.QRect(x * self.block_size + 1, y * self.block_size + 1,
                               self.block_size - 2, self.block_size - 2)
            painter.fillRect(rect, QtGui.QColor(THEME_COLORS['success']))
            painter.drawRect(rect)
        
        # 绘制蛇头
        if self.snake:
            hx, hy = self.snake[-1]
            head_rect = QtCore.QRect(hx * self.block_size + 1, hy * self.block_size + 1,
                                     self.block_size - 2, self.block_size - 2)
            painter.fillRect(head_rect, QtGui.QColor(THEME_COLORS['primary']))
            # 蛇头描边
            pen = QtGui.QPen(QtGui.QColor(10, 30, 80), 2)
            painter.setPen(pen)
            painter.drawRect(head_rect)
        
        # 绘制食物
        fx, fy = self.food
        food_rect = QtCore.QRect(fx * self.block_size + 2, fy * self.block_size + 2,
                                self.block_size - 4, self.block_size - 4)
        painter.fillRect(food_rect, QtGui.QColor(THEME_COLORS['danger']))
        # 食物高光
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 150, 150), 1))
        painter.drawEllipse(food_rect.adjusted(2, 2, -2, -2))
        
        # 绘制得分
        painter.setPen(QtGui.QPen(QtGui.QColor(THEME_COLORS['dark']), 1))
        font = QtGui.QFont('Arial', 14, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.drawText(10, 25, f"得分: {self.score}")


# ===== 主窗口 =====
class SnakeGameWindow(QtWidgets.QMainWindow):
    """贪吃蛇游戏主窗口"""
    
    def __init__(self, snake_queue, fruit_queue, record_queue, stop_event, start_event, speed, game_process):
        super().__init__()
        
        # 窗口配置
        self.setWindowTitle("🐍 自动循迹贪吃蛇 - AI Edition")
        self.setWindowIcon(self.create_window_icon())
        self.setGeometry(40, 40, 1380, 750)
        self.setMinimumSize(1280, 700)
        
        # 游戏相关
        self.snake_queue = snake_queue
        self.fruit_queue = fruit_queue
        self.record_queue = record_queue
        self.stop_event = stop_event
        self.start_event = start_event
        self.speed = speed
        self.game_process = game_process
        
        # 算法列表
        self.algorithms = ["BFS", "DFS", "A*", "Dijkstra", "Greedy", "Double_BFS"]
        self.current_algorithm = "BFS"
        
        # 游戏记录
        self.game_records = []
        
        # 创建UI
        self._create_ui()
        
        # 应用样式
        self._apply_stylesheet()
        
        # 定时器
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(200)

    def create_window_icon(self):
        """创建窗口图标"""
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtGui.QColor(255, 255, 255, 0))
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(8, 8, 8, 8, QtGui.QColor(THEME_COLORS['primary']))
        painter.fillRect(16, 8, 8, 8, QtGui.QColor(THEME_COLORS['success']))
        painter.fillRect(8, 16, 8, 8, QtGui.QColor(THEME_COLORS['warning']))
        painter.fillRect(16, 16, 8, 8, QtGui.QColor(THEME_COLORS['danger']))
        painter.end()
        return QtGui.QIcon(pixmap)

    def _create_ui(self):
        """创建用户界面"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        
        # ========== 左侧：游戏区域 ==========
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 游戏标题
        game_title = QtWidgets.QLabel("🎮 游戏区域")
        game_title.setFont(QtGui.QFont('Arial', 13, QtGui.QFont.Bold))
        game_title.setStyleSheet(f"color: {THEME_COLORS['primary']}; padding: 5px 0px;")
        left_layout.addWidget(game_title)
        
        # 游戏画布
        self.game_canvas = GameCanvas(self.snake_queue)
        left_layout.addWidget(self.game_canvas)
        
        # 左侧下方：实时统计信息
        stats_group = self._create_stats_group()
        left_layout.addWidget(stats_group)
        
        main_layout.addLayout(left_layout, 1)
        
        # ========== 右侧：控制和统计区域 ==========
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建滚动区域
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # ---- 1. 算法选择区 ----
        algo_group = self._create_algo_group()
        scroll_layout.addWidget(algo_group)
        
        # ---- 2. 游戏统计图 ----
        chart_label = QtWidgets.QLabel("📊 游戏统计分析")
        chart_label.setFont(QtGui.QFont('Arial', 11, QtGui.QFont.Bold))
        chart_label.setStyleSheet(f"color: {THEME_COLORS['primary']}; padding: 8px 0px;")
        scroll_layout.addWidget(chart_label)
        
        self.statistics_canvas = GameStatisticsCanvas()
        scroll_layout.addWidget(self.statistics_canvas)
        
        # 添加分隔线
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.HLine)
        separator1.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator1.setStyleSheet(f"background-color: {THEME_COLORS['border']}; height: 1px;")
        scroll_layout.addWidget(separator1)
        
        # ---- 3. 排名榜单 ----
        ranking_label = QtWidgets.QLabel("🏆 性能排名榜")
        ranking_label.setFont(QtGui.QFont('Arial', 11, QtGui.QFont.Bold))
        ranking_label.setStyleSheet(f"color: {THEME_COLORS['primary']}; padding: 8px 0px;")
        scroll_layout.addWidget(ranking_label)
        
        self.ranking_table = self._create_ranking_table()
        scroll_layout.addWidget(self.ranking_table)
        
        # 添加分隔线
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator2.setStyleSheet(f"background-color: {THEME_COLORS['border']}; height: 1px;")
        scroll_layout.addWidget(separator2)
        
        # ---- 4. 速度控制 ----
        speed_group = self._create_speed_group()
        scroll_layout.addWidget(speed_group)
        
        # 添加分隔线
        separator3 = QtWidgets.QFrame()
        separator3.setFrameShape(QtWidgets.QFrame.HLine)
        separator3.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator3.setStyleSheet(f"background-color: {THEME_COLORS['border']}; height: 1px;")
        scroll_layout.addWidget(separator3)
        
        # ---- 5. 操作按钮 ----
        button_layout = self._create_button_layout()
        scroll_layout.addLayout(button_layout)
        
        scroll_layout.addStretch()
        
        # 配置滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                width: 8px;
                background-color: #F0F0F0;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {THEME_COLORS['primary']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {THEME_COLORS['secondary']};
            }}
        """)
        right_layout.addWidget(scroll_area)
        
        main_layout.addLayout(right_layout, 1)

    def _create_algo_group(self):
        """创建算法选择组"""
        group = QtWidgets.QGroupBox("🔀 寻路算法选择")
        group.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        group.setMinimumHeight(90)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 15, 12, 12)
        
        algo_label = QtWidgets.QLabel("选择你要测试的路径规划算法：")
        algo_label.setFont(QtGui.QFont('Arial', 9))
        algo_label.setStyleSheet("color: #555555;")
        layout.addWidget(algo_label)
        
        combo_layout = QtWidgets.QHBoxLayout()
        combo_layout.setSpacing(10)
        combo_layout.setContentsMargins(0, 0, 0, 0)
        
        combo_label = QtWidgets.QLabel("算法：")
        combo_label.setFont(QtGui.QFont('Arial', 10))
        combo_layout.addWidget(combo_label, 0)
        
        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(self.algorithms)
        self.algo_combo.setMinimumHeight(32)
        self.algo_combo.setFont(QtGui.QFont('Arial', 10))
        self.algo_combo.currentTextChanged.connect(self.on_algorithm_changed)
        combo_layout.addWidget(self.algo_combo, 1)
        
        layout.addLayout(combo_layout)
        group.setLayout(layout)
        return group

    def _create_ranking_table(self):
        """创建排名表格"""
        table = QtWidgets.QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["排名", "算法", "得分", "总耗时(s)", "平均耗时(s)"])
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(250)  # 增加最小高度以显示更多行
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        
        # 设置表格样式
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setRowHeight(0, 28)  # 设置行高以增加可读性
        table.setStyleSheet(f"""
            QTableWidget {{
                gridline-color: {THEME_COLORS['border']};
                border: 1px solid {THEME_COLORS['border']};
                border-radius: 4px;
            }}
            QHeaderView::section {{
                background-color: {THEME_COLORS['primary']};
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-size: 9pt;
                height: 28px;
            }}
            QTableWidget::item {{
                padding: 6px;
                font-size: 9pt;
                height: 28px;
            }}
            QTableWidget::item:selected {{
                background-color: {THEME_COLORS['secondary']};
                color: white;
            }}
            alternate-background-color: #F5F5F5;
        """)
        
        return table

    def _create_stats_group(self):
        """创建统计信息组"""
        group = QtWidgets.QGroupBox("📈 实时统计")
        group.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        group.setMinimumHeight(115)
        group.setMaximumHeight(140)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 15, 12, 12)
        
        # 平均耗时
        self.avg_time_label = QtWidgets.QLabel("平均耗时：0.00 秒")
        self.avg_time_label.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.avg_time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.avg_time_label.setStyleSheet(f"color: {THEME_COLORS['primary']}; padding: 6px;")
        self.avg_time_label.setMinimumHeight(24)
        layout.addWidget(self.avg_time_label)
        
        # 统计信息
        self.stats_info_label = QtWidgets.QLabel("低于平均: 0个 | 高于平均: 0个 | 等于平均: 0个")
        self.stats_info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stats_info_label.setFont(QtGui.QFont('Arial', 9))
        self.stats_info_label.setStyleSheet("color: #666666; padding: 6px;")
        self.stats_info_label.setMinimumHeight(48)
        self.stats_info_label.setWordWrap(True)
        layout.addWidget(self.stats_info_label)
        
        group.setLayout(layout)
        return group

    def _create_speed_group(self):
        """创建速度控制组"""
        group = QtWidgets.QGroupBox("⚡ 蛇移动速度")
        group.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        group.setMinimumHeight(120)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 15, 12, 12)
        
        info_label = QtWidgets.QLabel("调整蛇的移动速度（值越大越快）：")
        info_label.setFont(QtGui.QFont('Arial', 9))
        info_label.setStyleSheet("color: #555555;")
        layout.addWidget(info_label)
        
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QtWidgets.QLabel("速度值：")
        label.setFont(QtGui.QFont('Arial', 10))
        control_layout.addWidget(label, 0)
        
        self.speed_input = QtWidgets.QSpinBox()
        self.speed_input.setRange(1, 200)
        self.speed_input.setValue(self.speed.value)
        self.speed_input.setMinimumHeight(32)
        self.speed_input.setFont(QtGui.QFont('Arial', 10))
        # 注意：不连接valueChanged，只有确认按钮点击时才更新速度
        control_layout.addWidget(self.speed_input, 1)
        
        # 确认按钮
        self.speed_confirm_btn = QtWidgets.QPushButton("确认")
        self.speed_confirm_btn.setFixedWidth(60)
        self.speed_confirm_btn.setMinimumHeight(32)
        self.speed_confirm_btn.setFont(QtGui.QFont('Arial', 9, QtGui.QFont.Bold))
        self.speed_confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME_COLORS['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1E5A7A;
            }}
            QPushButton:pressed {{
                background-color: #154360;
            }}
        """)
        self.speed_confirm_btn.clicked.connect(self.confirm_speed)
        control_layout.addWidget(self.speed_confirm_btn, 0)
        
        layout.addLayout(control_layout)
        group.setLayout(layout)
        return group

    def _create_button_layout(self):
        """创建按钮布局"""
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 开始按钮
        self.start_btn = QtWidgets.QPushButton("▶  开始游戏")
        self.start_btn.setFixedHeight(40)
        self.start_btn.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME_COLORS['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #058568;
            }}
            QPushButton:pressed {{
                background-color: #046551;
            }}
            QPushButton:disabled {{
                background-color: #ccc;
            }}
        """)
        self.start_btn.clicked.connect(self.start_game)
        layout.addWidget(self.start_btn)
        
        # 重新开始按钮
        self.restart_btn = QtWidgets.QPushButton("🔄 重新开始")
        self.restart_btn.setFixedHeight(40)
        self.restart_btn.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.restart_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME_COLORS['warning']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #D97000;
            }}
            QPushButton:pressed {{
                background-color: #B85C00;
            }}
        """)
        self.restart_btn.clicked.connect(self.restart_game)
        layout.addWidget(self.restart_btn)
        
        # 退出按钮
        self.exit_btn = QtWidgets.QPushButton("✕  退出")
        self.exit_btn.setFixedHeight(40)
        self.exit_btn.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.exit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME_COLORS['danger']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #B81C1C;
            }}
            QPushButton:pressed {{
                background-color: #900000;
            }}
        """)
        self.exit_btn.clicked.connect(self.close)
        layout.addWidget(self.exit_btn)
        
        return layout

    def _apply_stylesheet(self):
        """应用全局样式表"""
        stylesheet = f"""
            QMainWindow {{
                background-color: #F5F5F5;
            }}
            QGroupBox {{
                color: {THEME_COLORS['dark']};
                border: 2px solid {THEME_COLORS['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }}
            QLabel {{
                color: {THEME_COLORS['dark']};
            }}
            QComboBox, QLineEdit, QSpinBox {{
                border: 1px solid {THEME_COLORS['border']};
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }}
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{
                border: 2px solid {THEME_COLORS['primary']};
                outline: none;
            }}
            QComboBox::drop-down {{
                border: none;
                background-color: transparent;
            }}
        """
        self.setStyleSheet(stylesheet)

    def on_algorithm_changed(self, algorithm_name):
        """算法选择变更"""
        self.current_algorithm = algorithm_name
        self.restart_game()

    def start_game(self):
        """开始游戏"""
        self.start_event.set()
        self.start_btn.setEnabled(False)
        self.algo_combo.setEnabled(False)

    def confirm_speed(self):
        """确认速度设置"""
        new_speed = self.speed_input.value()
        if 1 <= new_speed <= 200:
            self.speed.value = new_speed
            # 显示确认提示
            self.speed_confirm_btn.setText("✓ 已确认")
            self.speed_confirm_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME_COLORS['success']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 5px;
                    font-weight: bold;
                }}
            """)
            # 1.5秒后恢复按钮
            QtCore.QTimer.singleShot(1500, self._reset_speed_confirm_btn)

    def _reset_speed_confirm_btn(self):
        """重置速度确认按钮"""
        self.speed_confirm_btn.setText("确认")
        self.speed_confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME_COLORS['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1E5A7A;
            }}
            QPushButton:pressed {{
                background-color: #154360;
            }}
        """)

    def update_data(self):
        """更新所有数据"""
        # 更新统计图
        while not self.fruit_queue.empty():
            data = self.fruit_queue.get()
            self.statistics_canvas.times.append(data["time"])
        
        self.statistics_canvas.update_plot()
        
        # 更新统计标签
        self.avg_time_label.setText(f"平均耗时：{self.statistics_canvas.average_time:.3f} 秒")
        stats_text = (f"低于平均: {self.statistics_canvas.below_average}个 | "
                     f"高于平均: {self.statistics_canvas.above_average}个 | "
                     f"等于平均: {self.statistics_canvas.equal_average}个")
        self.stats_info_label.setText(stats_text)
        
        # 更新排名表
        while not self.record_queue.empty():
            record = self.record_queue.get()
            self.game_records.append(record)
            self.game_records.sort(key=lambda x: (-x["score"], x["avg_time"]))
        
        self.update_ranking_table()

    def update_ranking_table(self):
        """更新排名表格"""
        self.ranking_table.setRowCount(0)
        for idx, record in enumerate(self.game_records):
            self.ranking_table.insertRow(idx)
            self.ranking_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(idx + 1)))
            self.ranking_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(record["snake_id"]))
            self.ranking_table.setItem(idx, 2, QtWidgets.QTableWidgetItem(str(record["score"])))
            self.ranking_table.setItem(idx, 3, QtWidgets.QTableWidgetItem(f"{record['total_time']:.2f}"))
            self.ranking_table.setItem(idx, 4, QtWidgets.QTableWidgetItem(f"{record['avg_time']:.3f}"))

    def restart_game(self):
        """重新开始游戏"""
        from multiprocessing import Process
        from main import game_process_main
        
        # 停止当前游戏
        self.stop_event.set()
        if self.game_process.is_alive():
            self.game_process.join(timeout=2)
        
        # 清空队列
        while not self.snake_queue.empty():
            self.snake_queue.get()
        while not self.fruit_queue.empty():
            self.fruit_queue.get()
        while not self.record_queue.empty():
            self.record_queue.get()
        
        # 重置统计
        self.statistics_canvas.times.clear()
        self.statistics_canvas.average_time = 0.0
        self.statistics_canvas.below_average = 0
        self.statistics_canvas.above_average = 0
        self.statistics_canvas.equal_average = 0
        self.avg_time_label.setText("平均耗时：0.00 秒")
        self.stats_info_label.setText("低于平均: 0个 | 高于平均: 0个 | 等于平均: 0个")
        
        # 重置事件
        self.start_event.clear()
        self.start_btn.setEnabled(True)
        self.algo_combo.setEnabled(True)
        
        # 重启游戏进程
        self.stop_event.clear()
        self.game_process = Process(
            target=game_process_main,
            args=(self.snake_queue, self.fruit_queue, self.stop_event, 
                 self.start_event, self.speed, self.record_queue, self.current_algorithm)
        )
        self.game_process.start()

    def closeEvent(self, event):
        """关闭窗口事件"""
        self.stop_event.set()
        if self.game_process.is_alive():
            self.game_process.join(timeout=2)
        event.accept()
