# 先安装依赖（如果未安装）：pip install numpy
import random
import numpy as np

def q_learning_snake(start, goal, snake_body, grid_width, grid_height):
    """
    精简版Q-Learning贪吃蛇循迹函数
    输入：起点、终点、蛇身、网格宽高
    输出：路径列表（无路径返回空）
    """
    # ===================== 1. 基础配置 =====================
    # 动作定义：0=上, 1=下, 2=左, 3=右（对应坐标偏移）
    ACTIONS = [0, 1, 2, 3]
    ACTION_TO_DIR = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    
    # Q-Learning超参数（新手无需大幅调整）
    LEARNING_RATE = 0.1    # 学习率：控制每次更新的幅度
    DISCOUNT_FACTOR = 0.9  # 折扣因子：未来奖励的权重（越近的未来越重要）
    EPSILON = 0.1          # 探索率：10%概率随机选动作（避免死学）
    TRAIN_EPISODES = 800   # 训练轮数：越多越精准（800轮新手测试足够）
    
    # 蛇身转集合，提升查询速度
    snake_body_set = set(snake_body)
    
    # ===================== 2. 辅助函数（核心工具） =====================
    def is_valid(pos):
        """校验坐标是否有效：在网格内 + 不在蛇身内"""
        x, y = pos
        return 0 <= x < grid_width and 0 <= y < grid_height and pos not in snake_body_set
    
    def get_state(head):
        """生成简化状态（6维，新手易理解）：蛇头的核心处境"""
        hx, hy = head
        fx, fy = goal
        
        # 状态组成（转元组，作为Q表的key）：
        # 1-2：蛇头相对食物的x/y方向（1=在左边/上边，0=对齐，-1=在右边/下边）
        # 3-6：蛇头上下左右是否有障碍（1=有，0=无）
        state = [
            1 if hx < fx else (-1 if hx > fx else 0),  # x方向相对食物
            1 if hy < fy else (-1 if hy > fy else 0),  # y方向相对食物
            0 if is_valid((hx, hy-1)) else 1,          # 上方向是否有障碍
            0 if is_valid((hx, hy+1)) else 1,          # 下方向是否有障碍
            0 if is_valid((hx-1, hy)) else 1,          # 左方向是否有障碍
            0 if is_valid((hx+1, hy)) else 1           # 右方向是否有障碍
        ]
        return tuple(state)
    
    def get_reward(head, done, collided):
        """奖励函数：引导蛇学习正确行为"""
        if collided:  # 撞墙/撞自己：惩罚
            return -100
        if done:      # 吃到食物：大奖
            return 50
        return 1      # 每存活一步：小奖励（鼓励存活）
    
    def choose_action(state, q_table):
        """ε-贪心选动作：兼顾探索（随机）和利用（选最优）"""
        # 新状态初始化Q值为0
        if state not in q_table:
            q_table[state] = [0.0, 0.0, 0.0, 0.0]
        
        # 10%概率随机探索，90%选Q值最大的动作
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        else:
            return np.argmax(q_table[state])
    
    # ===================== 3. 训练Q表（核心步骤） =====================
    q_table = {}  # Q表：key=状态，value=[动作0的价值, 动作1的价值, ...]
    
    for _ in range(TRAIN_EPISODES):
        current_head = start  # 每轮从起点重新开始
        collided = False      # 是否撞墙/撞自己
        done = False          # 是否吃到食物
        
        while not collided and not done:
            # 步骤1：获取当前状态，选择动作
            current_state = get_state(current_head)
            action = choose_action(current_state, q_table)
            
            # 步骤2：执行动作，得到新位置
            dx, dy = ACTION_TO_DIR[action]
            new_head = (current_head[0] + dx, current_head[1] + dy)
            
            # 步骤3：判断执行结果
            if not is_valid(new_head):
                collided = True  # 撞墙/撞自己
            elif new_head == goal:
                done = True      # 吃到食物
            
            # 步骤4：计算奖励，更新Q表
            reward = get_reward(new_head, done, collided)
            new_state = get_state(new_head)
            
            # Q表更新公式（核心！）：
            # 新价值 = 旧价值 + 学习率*(当前奖励 + 折扣因子*未来最优价值 - 旧价值)
            old_value = q_table[current_state][action]
            future_best_value = np.max(q_table.get(new_state, [0.0]*4))
            new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_best_value - old_value)
            q_table[current_state][action] = new_value
            
            # 步骤5：更新当前位置
            current_head = new_head
    
    # ===================== 4. 用训练好的Q表生成路径 =====================
    path = []
    current_head = start
    visited = set([start])  # 防止绕圈
    max_steps = grid_width * grid_height  # 防止无限循环
    
    for _ in range(max_steps):
        # 到达终点，结束
        if current_head == goal:
            break
        
        # 获取当前状态，选择最优动作（不再探索，只利用）
        current_state = get_state(current_head)
        if current_state not in q_table:
            break  # 无训练数据，无法选动作
        
        action = np.argmax(q_table[current_state])
        dx, dy = ACTION_TO_DIR[action]
        new_head = (current_head[0] + dx, current_head[1] + dy)
        
        # 校验新位置是否安全（不撞、不绕圈）
        if not is_valid(new_head) or new_head in visited:
            break
        
        # 加入路径，更新状态
        path.append(new_head)
        visited.add(new_head)
        current_head = new_head
    
    # 只有到达终点才返回路径，否则返回空
    return path if current_head == goal else []