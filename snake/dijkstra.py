import heapq

def dijkstra_snake_path(start, goal, snake_body, grid_width, grid_height):
    """
    迪杰斯特拉算法实现贪吃蛇自动循迹，返回从起点到终点的路径
    
    参数:
        start (tuple): 起点坐标 (x, y)
        goal (tuple): 终点坐标 (x, y)
        snake_body (iterable): 蛇身坐标集合/列表，如 [(x1,y1), (x2,y2), ...]
        grid_width (int): 网格宽度（x轴最大值为 grid_width-1）
        grid_height (int): 网格高度（y轴最大值为 grid_height-1）
    
    返回:
        list: 从start到goal的路径列表，每个元素是(x,y)元组，无路径时返回空列表
    """
    # 1. 基础校验：起点/终点越界、终点在蛇身内、起点等于终点，直接返回空
    def is_in_grid(pos):
        """校验坐标是否在网格内"""
        x, y = pos
        return 0 <= x < grid_width and 0 <= y < grid_height
    
    # 转换蛇身为集合，提升查询效率
    snake_body_set = set(snake_body)
    
    # 基础校验逻辑
    if not (is_in_grid(start) and is_in_grid(goal)):
        return []
    if goal in snake_body_set:
        return []
    if start == goal:
        return []
    
    # 2. 初始化迪杰斯特拉算法核心数据结构
    # 距离字典：key为坐标(x,y)，value为从start到该点的最小权重距离
    dist = {}
    # 初始化所有网格内点的距离为无穷大
    for x in range(grid_width):
        for y in range(grid_height):
            dist[(x, y)] = float('inf')
    dist[start] = 0  # 起点距离为0
    
    # 优先队列：元素为 (当前距离, 坐标)，堆排序会优先弹出距离最小的元素
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    
    # 父节点字典：记录每个节点的前驱节点，用于回溯路径
    parent = {}
    
    # 3. 定义移动方向：上下左右四个方向
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # 4. 核心算法循环
    while priority_queue:
        # 弹出当前距离最小的节点
        current_dist, current_pos = heapq.heappop(priority_queue)
        
        # 找到终点，提前终止循环
        if current_pos == goal:
            break
        
        # 如果当前节点的已知距离已经小于堆中弹出的距离，跳过（已处理过更优路径）
        if current_dist > dist[current_pos]:
            continue
        
        # 遍历四个移动方向
        for dx, dy in directions:
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_pos = (next_x, next_y)
            
            # 跳过越界、蛇身的节点
            if not is_in_grid(next_pos) or next_pos in snake_body_set:
                continue
            
            # 计算权重：普通格子权重1，边界格子（x/y为0或最大值）权重2（优先选非边界路径）
            if next_x in (0, grid_width-1) or next_y in (0, grid_height-1):
                weight = 2
            else:
                weight = 1
            
            # 计算新路径的总距离
            new_dist = current_dist + weight
            
            # 如果新路径更优，更新距离和父节点
            if new_dist < dist[next_pos]:
                dist[next_pos] = new_dist
                parent[next_pos] = current_pos
                heapq.heappush(priority_queue, (new_dist, next_pos))
    
    # 5. 回溯路径：从终点倒推回起点
    path = []
    current = goal
    # 检查终点是否有父节点（是否存在有效路径）
    while current in parent:
        path.append(current)
        current = parent[current]
    # 如果路径不为空，说明找到路径，反转得到从start到goal的顺序
    if path:
        path.append(start)
        path.reverse()
        # 移除起点（路径是从起点下一步到终点，符合贪吃蛇移动逻辑）
        path = path[1:]
    return path