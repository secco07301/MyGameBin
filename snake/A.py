from heapq import heappush, heappop

def a_star(start, goal, snake_body, grid_width, grid_height):
    """
    A* 寻路算法（适配贪吃蛇场景，对齐DFS/BFS输入输出）
    :param start: 蛇头坐标，格式为 (x, y)
    :param goal: 食物坐标，格式为 (x, y)
    :param snake_body: 蛇身列表，格式为 [(x1,y1), (x2,y2), ...]
    :param grid_width: 游戏网格宽度（列数）
    :param grid_height: 游戏网格高度（行数）
    :return: 路径列表（从蛇头下一步到食物的坐标），无路径返回空列表 []
    """
    # 转换蛇身为集合，O(1) 碰撞检测（比列表遍历快）
    snake_set = set(snake_body)
    
    # 启发函数：曼哈顿距离（网格场景最优，计算量最小）
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # 优先队列：(总代价f, 已走代价g, 当前位置, 父节点)
    open_heap = []
    heappush(open_heap, (heuristic(start), 0, start, None))
    
    # 已访问节点：key=坐标，value=(父节点, 已走代价g)
    closed_dict = {start: (None, 0)}
    
    while open_heap:
        # 取出f值最小的节点（最优优先）
        f_val, g_val, current_pos, parent_pos = heappop(open_heap)
        
        # 到达食物，回溯生成路径
        if current_pos == goal:
            path = []
            # 从食物往回找父节点，直到蛇头
            while current_pos is not None:
                path.append(current_pos)
                current_pos = closed_dict[current_pos][0]
            # 反转路径并去掉蛇头（只保留下一步到食物的路径）
            path.reverse()
            return path[1:] if len(path) > 1 else []
        
        # 遍历四个移动方向（右、左、下、上，对齐你原有寻路的方向顺序）
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            next_pos = (new_x, new_y)
            
            # 1. 检查边界：是否在网格内
            if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
                # 2. 检查碰撞：是否撞到蛇身
                if next_pos not in snake_set:
                    new_g_val = g_val + 1  # 每走一步代价+1
                    # 3. 检查是否已访问，或是否有更优路径（g值更小）
                    if next_pos not in closed_dict or new_g_val < closed_dict[next_pos][1]:
                        closed_dict[next_pos] = (current_pos, new_g_val)
                        new_f_val = new_g_val + heuristic(next_pos)  # f = g + h
                        heappush(open_heap, (new_f_val, new_g_val, next_pos, current_pos))
    
    # 无有效路径，返回空列表（对齐你原有DFS/BFS的返回格式）
    return []