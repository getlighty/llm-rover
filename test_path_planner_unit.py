import numpy as np

from path_planner import CELL_SIZE, MAP_CENTER, MAP_SIZE, PathPlanner


class DummyWorld:
    def __init__(self):
        self.grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)

    def _world_to_grid(self, x, y):
        gx = int(x / CELL_SIZE) + MAP_CENTER
        gy = int(y / CELL_SIZE) + MAP_CENTER
        return gx, gy

    def _grid_to_world(self, gx, gy):
        x = (gx - MAP_CENTER) * CELL_SIZE
        y = (gy - MAP_CENTER) * CELL_SIZE
        return x, y


def test_astar_prevents_diagonal_corner_cut():
    world = DummyWorld()
    planner = PathPlanner(world)

    grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    sx, sy = MAP_CENTER, MAP_CENTER
    gx, gy = sx + 1, sy + 1

    # Block the two orthogonal neighbors of the diagonal.
    grid[sy, gx] = -1
    grid[gy, sx] = -1

    path = planner._astar(grid, (sx, sy), (gx, gy))
    assert path, "Expected alternate path around the blocked diagonal corner"
    assert len(path) >= 3
    assert path[1] != (gx, gy), "Path should not cut diagonally through obstacle corners"


def test_smoothing_keeps_segments_obstacle_safe():
    world = DummyWorld()
    planner = PathPlanner(world)

    grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
    # Obstacle in the center blocks direct line from start to end.
    grid[MAP_CENTER + 1, MAP_CENTER + 1] = -1

    path_cells = [
        (MAP_CENTER, MAP_CENTER),
        (MAP_CENTER, MAP_CENTER + 1),
        (MAP_CENTER, MAP_CENTER + 2),
        (MAP_CENTER + 1, MAP_CENTER + 2),
        (MAP_CENTER + 2, MAP_CENTER + 2),
    ]
    smoothed = planner._smooth_path_cells(path_cells, grid)

    assert smoothed[0] == path_cells[0]
    assert smoothed[-1] == path_cells[-1]
    assert len(smoothed) >= 3, "Obstacle should force at least one intermediate waypoint"

    for a, b in zip(smoothed, smoothed[1:]):
        assert planner._line_is_clear(grid, a, b), f"Unsafe smoothed segment {a}->{b}"
