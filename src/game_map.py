import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from queue import Queue, LifoQueue, PriorityQueue
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import List, Optional, Set, Tuple


# Color definitions for visualization
class CellColor(IntEnum):
    """Enum defining colors for different cell states in the grid."""

    DEFAULT = 0  # white
    PATH = 1  # blue
    VISITED = 2  # green
    SOURCE = 3  # lime
    TO_VISIT = 4  # cyan
    TARGET = 5  # red
    OBSTACLE = 6  # black


COLORS = [
    "white",  # DEFAULT
    "blue",  # PATH
    "green",  # VISITED
    "lime",  # SOURCE
    "cyan",  # TO_VISIT
    "red",  # TARGET
    "black",  # OBSTACLE
]

# Animation speed (seconds)
PAUSE_PERIOD = 0.001


@dataclass
class Point:
    """Represents a 2D point with x,y coordinates."""

    x: int
    y: int

    def __eq__(self, other: object) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def norm(self) -> float:
        """Calculate Euclidean norm of the point vector."""
        return math.hypot(self.x, self.y)


class Grid:
    """Represents a single cell in the game grid."""

    def __init__(self, x: int, y: int) -> None:
        self.coordinate = Point(x, y)
        self.previous_coordinate: Point = None
        self.cost_to_come: float = None
        self.heuristic: float = None

        # Cell state flags
        self.is_source: bool = False
        self.is_target: bool = False
        self.is_visited: bool = False
        self.is_to_visit: bool = False
        self.is_obstacle: bool = False
        self.is_on_path: bool = False


class GameMap:
    """Main class for managing the game grid and pathfinding visualization."""

    def __init__(self, X: int, Y: int):
        """Initialize game map with given dimensions."""
        self.X = X
        self.Y = Y
        self.init_point: Point = None
        self.term_point: Point = None
        self.point_path: List[Point] = []

        self.custom_cmap = ListedColormap(COLORS)
        self.grid_map = [[Grid(x, y) for y in range(Y)] for x in range(X)]
        self.visualization_map = np.zeros((X, Y), dtype=int)
        self.img: Optional[plt.AxesImage] = None

    def set_init_point(self, x: int, y: int) -> None:
        """Set the starting point on the grid."""
        self.init_point = Point(x, y)
        self.grid_map[x][y].cost_to_come = 0
        self.grid_map[x][y].is_source = True

    def set_term_point(self, x: int, y: int) -> None:
        """Set the target point on the grid."""
        self.term_point = Point(x, y)
        self.grid_map[x][y].is_target = True

    def add_obstacle_point(self, point: Point) -> None:
        """Mark a point as an obstacle."""
        self.grid_map[point.x][point.y].is_obstacle = True

    def add_obstacle_line(self, point1: Point, point2: Point) -> None:
        """Add a line of obstacles between two points."""
        if point1.x == point2.x:
            for y in range(min(point1.y, point2.y), max(point1.y, point2.y) + 1):
                self.add_obstacle_point(Point(point1.x, y))
            return

        left_point = point1 if point1.x < point2.x else point2
        right_point = point2 if point1.x < point2.x else point1
        delta_x = right_point.x - left_point.x
        delta_y = right_point.y - left_point.y
        slope = delta_y / delta_x

        for dx in range(delta_x + 1):
            x = left_point.x + dx
            y = round(left_point.y + slope * dx)
            self.add_obstacle_point(Point(x, y))

    def is_obstacle(self, point: Point) -> bool:
        """Check if a point is an obstacle."""
        return self.grid_map[point.x][point.y].is_obstacle

    def get_adjacent_points(self, point: Point) -> List[Point]:
        """Get list of adjacent points (up, down, left, right)."""
        return [Point(point.x + i, point.y) for i in [-1, 1]] + [
            Point(point.x, point.y + j) for j in [-1, 1]
        ]

    def update_visualization_map(self) -> None:
        """Update the visualization map based on current grid state."""
        for x in range(self.X):
            for y in range(self.Y):
                grid = self.grid_map[x][y]
                if grid.is_target:
                    self.visualization_map[x][y] = CellColor.TARGET
                    continue
                if grid.is_source:
                    self.visualization_map[x][y] = CellColor.SOURCE
                    continue
                if grid.is_obstacle:
                    self.visualization_map[x][y] = CellColor.OBSTACLE
                    continue
                if grid.is_visited:
                    self.visualization_map[x][y] = CellColor.VISITED
                    continue
                if grid.is_to_visit:
                    self.visualization_map[x][y] = CellColor.TO_VISIT
                    continue
                self.visualization_map[x][y] = CellColor.DEFAULT

    def show_map(self) -> None:
        """Display the current state of the map."""
        plt.figure(figsize=(6, 6))
        extent = [0, self.X, self.Y, 0]
        self.img = plt.imshow(
            self.visualization_map,
            cmap=self.custom_cmap,
            interpolation="nearest",
            extent=extent,
            vmin=0,
            vmax=len(COLORS) - 1,
            origin="upper",
        )

        # Set the aspect ratio to be equal
        plt.gca().set_aspect("equal", adjustable="box")

        # Add grid lines
        plt.xticks(np.arange(self.X))
        plt.yticks(np.arange(self.Y))
        plt.grid(which="both", color="black", linestyle="-", linewidth=1)

        # Set the title
        plt.title("Game Grid Map")

        # Show the plot
        self.img.set_array(self.visualization_map)
        plt.draw()

    def calculate_heuristic(self, point: Point) -> float:
        """Calculate Manhattan distance heuristic to target."""
        diff = self.term_point - point
        return abs(diff.x) + abs(diff.y)

    def backtrace_path(self, point: Point) -> List[Point]:
        """Reconstruct path from target back to source."""
        curr = point
        path = [curr]
        while not self.grid_map[curr.x][curr.y].is_source:
            curr = self.grid_map[curr.x][curr.y].previous_coordinate
            path.append(curr)
        path.reverse()
        return path

    def update_path(self, point: Point) -> None:
        """Update the current path."""
        grid_path = self.backtrace_path(point)
        self.point_path = []
        for grid in grid_path:
            self.point_path.append(Point(grid.x, grid.y))

    def visualize_path(self) -> None:
        """Draw the current path on the plot."""
        x = [point.y + 0.5 for point in self.point_path]
        y = [point.x + 0.5 for point in self.point_path]
        plt.plot(x, y, color=COLORS[CellColor.PATH])

    def breadth_first_search(self, source: Point, target: Point) -> bool:
        """Perform BFS pathfinding from source to target."""
        queue: Queue[Point] = Queue()
        queue.put(source)

        while not queue.empty():
            curr = queue.get()
            if self.grid_map[curr.x][curr.y].is_target:
                return True

            if self.grid_map[curr.x][curr.y].is_visited:
                continue

            self.grid_map[curr.x][curr.y].is_to_visit = False
            self.grid_map[curr.x][curr.y].is_visited = True

            for point in self.get_adjacent_points(curr):
                if (
                    not self.is_point_inside_grid_map(point)
                    or self.grid_map[point.x][point.y].is_visited
                    or self.is_obstacle(point)
                ):
                    continue

                queue.put(point)
                self.grid_map[point.x][point.y].is_to_visit = True
                self.grid_map[point.x][point.y].previous_coordinate = curr

            self.update_visualization_map()
            self.img.set_array(self.visualization_map)
            plt.draw()
            plt.pause(PAUSE_PERIOD)

        return False

    # def a_star(self, source: Point, target: Point):
    #     visited = set()
    #     to_visit = set()
    #     pq = PriorityQueue()
    #     pq.put(GridWithCost(source, 0, self.calculate_heuristic(source)))
    #     to_visit.add(source)
    #     while not pq.empty():
    #         curr = pq.get()
    #         if curr.point in to_visit:
    #             to_visit.remove(curr.point)
    #         if curr.point in visited:
    #             continue
    #         visited.add(curr.point)
    #         value = self.grid_map[curr.point.x][curr.point.y]
    #         if value == TARGET_COLOR:
    #             return True
    #         adj = self.get_adjacent_points(curr.point)
    #         for point in adj:
    #             if (
    #                 (not self.is_point_inside_grid_map(point))
    #                 or (point in visited)
    #                 or (self.is_obstacle(point))
    #             ):
    #                 continue
    #             pq.put(
    #                 GridWithCost(
    #                     point, curr.cost_to_come + 1, self.calculate_heuristic(point)
    #                 )
    #             )
    #             to_visit.add(point)

    #         self.update_grid_map(visited, to_visit)
    #         self.img.set_array(self.grid_map)
    #         plt.draw()
    #         plt.pause(PAUSE_PERIOD)
    #     return False

    # def depth_first_search(self, source: Point, target: Point) -> bool:
    #     visited = set()
    #     to_visit = set()
    #     stack = LifoQueue()
    #     stack.put(source)
    #     to_visit.add(source)
    #     while not stack.empty():
    #         curr = stack.get()
    #         if curr in to_visit:
    #             to_visit.remove(curr)
    #         if curr in visited:
    #             continue
    #         visited.add(curr)
    #         value = self.grid_map[curr.x][curr.y]
    #         if value == TARGET_COLOR:
    #             return True
    #         adj = self.get_adjacent_points(curr)
    #         for point in adj:
    #             if (
    #                 (not self.is_point_inside_grid_map(point))
    #                 or (point in visited)
    #                 or (self.is_obstacle(point))
    #             ):
    #                 continue
    #             stack.put(point)
    #             to_visit.add(point)

    #         self.update_grid_map(visited, to_visit)
    #         self.img.set_array(self.grid_map)
    #         plt.draw()
    #         plt.pause(PAUSE_PERIOD)
    #         return False
    #     return False

    def update_grid_map(self, visited: Set[Point], to_visit: Set[Point]) -> None:
        """Update grid map with visited and to-visit cells."""
        for point in visited:
            if point in (self.init_point, self.term_point):
                continue
            self.grid_map[point.x][point.y] = CellColor.VISITED

        for point in to_visit:
            if point in (self.init_point, self.term_point) or point in visited:
                continue
            self.grid_map[point.x][point.y] = CellColor.TO_VISIT

    def is_point_inside_grid_map(self, point: Point) -> bool:
        """Check if point is within grid boundaries."""
        return 0 <= point.x < self.X and 0 <= point.y < self.Y


if __name__ == "__main__":
    X = 16
    Y = 16
    game_map = GameMap(X, Y)

    source_point = Point(5, 2)
    target_point = Point(15, 10)

    game_map.set_init_point(source_point.x, source_point.y)
    game_map.set_term_point(target_point.x, target_point.y)

    # Add obstacles
    game_map.add_obstacle_line(Point(14, 9), Point(14, 6))
    game_map.add_obstacle_line(Point(13, 3), Point(5, 0))
    game_map.add_obstacle_line(Point(13, 10), Point(9, 14))

    game_map.show_map()
    game_map.breadth_first_search(source_point, target_point)

    game_map.update_visualization_map()
    game_map.update_path(game_map.term_point)
    game_map.visualize_path()

    if game_map.img is not None:
        game_map.img.set_array(game_map.visualization_map)
        plt.draw()
        plt.pause(PAUSE_PERIOD)
        plt.show()
