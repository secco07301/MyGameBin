def dfs(start, goal, snake, grid_w, grid_h, max_steps=800):
    """
    DFS 路径搜索
    返回：从 start 到 goal 的一条路径（不保证最短）
    """
    snake_set = set(snake)

    # stack: (current_pos, path, visited)
    stack = [(start, [], set([start]))]
    steps = 0

    while stack:
        cur, path, visited = stack.pop()
        steps += 1
        if steps > max_steps:
            break

        if cur == goal:
            return path

        x, y = cur

        # 按“靠近果子”排序方向（贪心）
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        dirs.sort(
            key=lambda d: abs(x + d[0] - goal[0]) + abs(y + d[1] - goal[1]),
            reverse=True  # ⚠️ 栈是 LIFO，要反着来
        )

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)

            if (
                0 <= nx < grid_w and
                0 <= ny < grid_h and
                nxt not in snake_set and
                nxt not in visited
            ):
                stack.append(
                    (nxt, path + [nxt], visited | {nxt})
                )

    return None

def safe_move(head, snake, grid_w, grid_h):
    snake_set = set(snake)
    for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
        nx, ny = head[0] + dx, head[1] + dy
        if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in snake_set:
            return (nx, ny)
    return None