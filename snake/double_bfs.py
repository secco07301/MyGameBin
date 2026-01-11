def bidirectional_bfs_snake_path(start, goal, snake_body, grid_width, grid_height, max_steps=1000):
    """
    双向BFS算法实现贪吃蛇自动循迹，返回从start到goal的最短路径（BFS特性）
    
    参数:
        start (tuple): 起点坐标 (x, y)
        goal (tuple): 终点坐标 (x, y)
        snake_body (iterable): 蛇身坐标集合/列表，如 [(x1,y1), (x2,y2), ...]
        grid_width (int): 网格宽度（x轴最大值为 grid_width-1）
        grid_height (int): 网格高度（y轴最大值为 grid_height-1）
        max_steps (int): 最大迭代步数，防止无限循环
    
    返回:
        list: 从start到goal的路径列表，每个元素是(x,y)元组，无路径时返回空列表
    """
    # 辅助函数：校验坐标是否在网格内
    def is_in_grid(pos):
        x, y = pos
        return 0 <= x < grid_width and 0 <= y < grid_height
    
    # 基础校验
    snake_body_set = set(snake_body)
    # 起点/终点越界、终点在蛇身、起点=终点，直接返回空
    if not (is_in_grid(start) and is_in_grid(goal)):
        return []
    if goal in snake_body_set or start == goal:
        return []
    
    # 1. 初始化双向BFS的核心数据结构
    # 队列：存储待扩展的节点（每层扩展一批，保证广度优先）
    start_queue = [start]
    goal_queue = [goal]
    # 父节点字典：key=节点，value=父节点（用于回溯路径）
    start_parent = {start: None}
    goal_parent = {goal: None}
    # 移动方向：上下左右
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    steps = 0
    meet_node = None  # 记录两个搜索树相遇的节点
    
    # 2. 核心双向BFS循环
    while start_queue and goal_queue and steps < max_steps:
        steps += 1
        
        # 扩展起点侧的队列（一层）
        start_level_size = len(start_queue)
        for _ in range(start_level_size):
            cur = start_queue.pop(0)
            # 检查当前节点是否在终点侧的父字典中（相遇）
            if cur in goal_parent:
                meet_node = cur
                break
            # 遍历四个方向
            for dx, dy in directions:
                nx, ny = cur[0] + dx, cur[1] + dy
                nxt = (nx, ny)
                # 校验：在网格内、不在蛇身、未被起点侧访问过
                if is_in_grid(nxt) and nxt not in snake_body_set and nxt not in start_parent:
                    start_parent[nxt] = cur
                    start_queue.append(nxt)
        if meet_node:
            break
        
        # 扩展终点侧的队列（一层）
        goal_level_size = len(goal_queue)
        for _ in range(goal_level_size):
            cur = goal_queue.pop(0)
            # 检查当前节点是否在起点侧的父字典中（相遇）
            if cur in start_parent:
                meet_node = cur
                break
            # 遍历四个方向
            for dx, dy in directions:
                nx, ny = cur[0] + dx, cur[1] + dy
                nxt = (nx, ny)
                # 校验：在网格内、不在蛇身、未被终点侧访问过
                if is_in_grid(nxt) and nxt not in snake_body_set and nxt not in goal_parent:
                    goal_parent[nxt] = cur
                    goal_queue.append(nxt)
        if meet_node:
            break
    
    # 3. 未找到相遇节点，返回空路径
    if not meet_node:
        return []
    
    # 4. 拼接路径：起点→相遇节点 → 相遇节点→终点
    # 第一步：回溯起点到相遇节点的路径
    path_start = []
    current = meet_node
    while current is not None:
        path_start.append(current)
        current = start_parent[current]
    path_start.reverse()  # 反转得到 起点→相遇节点 的正向路径
    
    # 第二步：回溯相遇节点到终点的路径
    path_goal = []
    current = meet_node
    while current is not None:
        path_goal.append(current)
        current = goal_parent[current]
    # 相遇节点会被重复，所以去掉path_goal的第一个元素，再反转
    path_goal = path_goal[1:][::-1]
    
    # 合并路径：起点→相遇节点 + 相遇节点→终点（去重）
    full_path = path_start + path_goal
    
    # 5. 移除起点（符合贪吃蛇路径格式：下一步要走的坐标列表）
    if full_path and full_path[0] == start:
        full_path = full_path[1:]
    
    return full_path