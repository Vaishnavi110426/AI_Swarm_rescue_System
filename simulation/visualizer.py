import cv2
import numpy as np

def draw_env(grid_size, drone_positions, human_positions, obstacles=None):
    """
    Draw a top-down grid map:
    - grid_size: number of cells per side
    - drone_positions: array-like of shape (N,2)
    - human_positions: array-like of shape (M,2)
    - obstacles: iterable of (x,y) tuples
    """
    cell = 24
    img_size = grid_size * cell
    map_img = np.full((img_size, img_size, 3), 30, dtype=np.uint8)

    # grid lines
    for i in range(0, img_size, cell):
        cv2.line(map_img, (i, 0), (i, img_size), (60, 60, 60), 1)
        cv2.line(map_img, (0, i), (img_size, i), (60, 60, 60), 1)

    # obstacles
    if obstacles:
        for ox, oy in obstacles:
            if 0 <= ox < grid_size and 0 <= oy < grid_size:
                x1, y1 = ox * cell, oy * cell
                x2, y2 = x1 + cell, y1 + cell
                cv2.rectangle(map_img, (x1, y1), (x2, y2), (100, 100, 100), -1)

    # humans (red)
    for h in human_positions:
        hx, hy = int(h[0]), int(h[1])
        if hx >= 0 and hy >= 0:
            cx, cy = hx * cell + cell//2, hy * cell + cell//2
            cv2.circle(map_img, (cx, cy), cell//4, (0, 0, 255), -1)

    # drones (green)
    for d in drone_positions:
        dx, dy = int(d[0]), int(d[1])
        if dx >= 0 and dy >= 0:
            cx, cy = dx * cell + cell//2, dy * cell + cell//2
            cv2.circle(map_img, (cx, cy), cell//3, (0, 255, 0), -1)

    # border
    cv2.rectangle(map_img, (0, 0), (img_size-1, img_size-1), (150,150,150), 2)
    return map_img
