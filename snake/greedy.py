def greedy_snake_path(start, goal, snake_body, grid_width, grid_height):
    """
    贪心算法实现贪吃蛇自动循迹，返回从起点到终点的路径列表
    
    参数:
        start (tuple): 起点坐标 (x, y)
        goal (tuple): 终点坐标 (x, y)
        snake_body (iterable): 蛇身坐标集合/列表，如 [(x1,y1), (x2,y2), ...]
        grid_width (int): 网格宽度（x轴最大值为 grid_width-1）
        grid_height (int): 网格高度（y轴最大值为 grid_height-1）
    
    返回:
        list: 从start到goal的路径列表，每个元素是(x,y)元组，无路径时返回空列表
    """
    # 辅助函数1：校验坐标是否在网格内
    def is_in_grid(pos):
        x, y = pos
        return 0 <= x < grid_width and 0 <= y < grid_height
    
    # 辅助函数2：校验坐标是否安全（不在网格外、不在蛇身内、不在已走路径内）
    def is_safe(pos, visited):
        return is_in_grid(pos) and pos not in snake_body_set and pos not in visited
    
    # 辅助函数3：计算两个点的曼哈顿距离（评估到食物的远近）
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # 辅助函数4：计算某个位置的活动空间大小（用BFS统计可移动区域，避免死局）
    def calculate_space(pos, visited):
        # 临时蛇身：原蛇身 + 已走路径（模拟蛇移动后的身体）
        temp_body = snake_body_set.union(visited)
        queue = [pos]
        space_visited = set(queue)
        count = 0
        
        while queue:
            current = queue.pop(0)
            count += 1
            # 遍历四个方向
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_x = current[0] + dx
                next_y = current[1] + dy
                next_pos = (next_x, next_y)
                if is_in_grid(next_pos) and next_pos not in temp_body and next_pos not in space_visited:
                    space_visited.add(next_pos)
                    queue.append(next_pos)
        return count
    
    # 基础校验
    snake_body_set = set(snake_body)
    if not (is_in_grid(start) and is_in_grid(goal)):
        return []
    if goal in snake_body_set:
        return []
    if start == goal:
        return []
    
    # 初始化路径和已访问集合（防止绕圈）
    path = []
    current_pos = start
    visited = set([current_pos])
    # 移动方向：上下左右
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # 核心贪心循环：直到到达终点或无安全方向
    max_steps = grid_width * grid_height  # 最大步数（防止无限循环）
    step = 0
    while current_pos != goal and step < max_steps:
        step += 1
        # 1. 收集所有安全的候选方向（下一步坐标）
        candidates = []
        for dx, dy in directions:
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_pos = (next_x, next_y)
            if is_safe(next_pos, visited):
                candidates.append(next_pos)
        
        # 无安全方向，返回空路径
        if not candidates:
            return []
        
        # 2. 对候选方向打分（核心贪心策略）
        scored_candidates = []
        for pos in candidates:
            # 分数1：到食物的距离（越小越好，取负数转为加分项）
            dist_score = -manhattan_distance(pos, goal)
            # 分数2：活动空间（越大越好）
            space_score = calculate_space(pos, visited)
            # 分数3：远离边界（边界坐标扣分，非边界加分）
            border_score = 0
            if pos[0] in (0, grid_width-1) or pos[1] in (0, grid_height-1):
                border_score = -2  # 边界减分
            else:
                border_score = 1   # 非边界加分
            # 总得分（权重可调整）
            total_score = dist_score * 3 + space_score * 2 + border_score * 1
            scored_candidates.append((-total_score, pos))  # 用负数方便堆排序
        
        # 3. 选择得分最高的方向（最小堆弹出最小负数=最大得分）
        heapq.heapify(scored_candidates)
        best_pos = heapq.heappop(scored_candidates)[1]
        
        # 4. 更新路径和当前位置
        path.append(best_pos)
        visited.add(best_pos)
        current_pos = best_pos
    
    # 循环结束：到达终点则返回路径，否则返回空
    return path if current_pos == goal else []

# 补充：需要导入heapq（上面代码用到堆排序）
import heapq