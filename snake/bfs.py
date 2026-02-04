# 小蛇寻路算法模块 BFS 实现
from collections import deque

def bfs(start, goal, snake, grid_w, grid_h):
    """
    BFS寻路核心函数
    :param start: 起点坐标 (x,y)
    :param goal: 终点坐标 (x,y)
    :param snake: 蛇身坐标列表
    :param grid_w: 网格宽度（列数）
    :param grid_h: 网格高度（行数）
    :return: 路径列表或None
    """
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
            # 使用传入的网格参数判断边界
            if 0 <= nx < grid_w and 0 <= ny < grid_h and nxt not in visited and nxt not in snake_set:
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

def safe_move(head, snake, grid_w, grid_h):
    snake_set = set(snake)
    for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
        nx, ny = head[0] + dx, head[1] + dy
        if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in snake_set:
            return (nx, ny)
    return None