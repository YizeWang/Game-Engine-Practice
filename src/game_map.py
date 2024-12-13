import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from queue import Queue, LifoQueue,PriorityQueue
import math

colors = ["white", "blue", "green", "lime", "cyan", "red", "black"]

TARGET_COLOR = 5
SOURCE_COLOR = 3
VISITED_COLOR = 2
TO_VISIT_COLOR = 4
DEFAULT_COLOR = 0
OBSTACLE_COLOR = 6

PAUSE_PERIOD = 0.01


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, rhs):
        return self.x == rhs.x and self.y == rhs.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __sub__(self, rhs):
        return Point(self.x - rhs.x, self.y - rhs.y)
    
    def norm(self):
        return math.hypot(self.x,self.y)


class GridWithCost:
    def __init__(self, point:Point,cost_to_come:int,heuristic:int):
        self.point = point
        self.cost_to_come = cost_to_come
        self.heuristic = heuristic

    def __gt__(self, rhs):
        return self.cost_to_come + self.heuristic > rhs.cost_to_come + rhs.heuristic

    def __ls__(self, rhs):
        return self.cost_to_come +self.heuristic < rhs.cost_to_come+ rhs.heuristic

    def __eq__(self, rhs):
        return self.cost_to_come+self.heuristic == rhs.cost_to_come+ rhs.heuristic


class GameMap:
    def __init__(self, X: int, Y: int):
        # Map dimension.
        self.X = X
        self.Y = Y

        self.custom_cmap = ListedColormap(colors)
        self.grid_map = DEFAULT_COLOR * np.ones(np.array([self.X, self.Y]))

    def set_init_point(self, x: int, y: int):
        self.init_point = Point(x, y)
        self.grid_map[x, y] = SOURCE_COLOR

    def set_term_point(self, x: int, y: int):
        self.term_point = Point(x, y)
        self.grid_map[x, y] = TARGET_COLOR

    def add_obstacle_point(self, point: Point):
        self.grid_map[point.x][point.y] = OBSTACLE_COLOR

    def add_obstacle_line(self, point1: Point, point2: Point):
        if point1.x == point2.x:
            for y in range(min(point1.y, point2.y), max(point1.y, point2.y) + 1, 1):
                self.add_obstacle_point(Point(point1.x, y))
            return

        left_point = point1 if point1.x < point2.x else point2
        right_point = point1 if point1.x > point2.x else point2
        delta_x = right_point.x - left_point.x
        delta_y = right_point.y - left_point.y
        slope = delta_y / delta_x
        for dx in range(0, delta_x + 1, 1):
            x = left_point.x + dx
            y = round(left_point.y + slope * dx)
            self.add_obstacle_point(Point(x, y))

    def is_obstacle(self, point: Point) ->bool:
        return self.grid_map[point.x][point.y] == OBSTACLE_COLOR

    def get_adjacent_points(self, point:Point) -> list:
        return [Point(point.x + i, point.y) for i in [-1, 1]] + [Point(point.x, point.y + j) for j in [-1, 1]]

    def show_map(self):
        plt.figure(figsize=(6, 6))
        extent = [0,self.X, 0, self.Y]
        self.img = plt.imshow(
            self.grid_map,
            cmap=self.custom_cmap,
            interpolation="nearest",
            extent=extent,
            vmin=0,
            vmax=len(colors) - 1,
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
        self.img.set_array(self.grid_map)
        plt.draw()

    def calculate_heuristic(self, point: Point) -> float:
        diff = self.term_point - point
        return abs(diff.x) + abs(diff.y)

    def a_star(self, source: Point, target: Point):
        visited = set()
        to_visit = set()
        pq = PriorityQueue()
        pq.put(GridWithCost(source, 0,self.calculate_heuristic(source)))
        to_visit.add(source)
        while not pq.empty():
            curr = pq.get()
            if curr.point in to_visit:
                to_visit.remove(curr.point)
            if curr.point in visited:
                continue
            visited.add(curr.point)
            value = self.grid_map[curr.point.x][curr.point.y]
            if value == TARGET_COLOR:
                return True
            adj = self.get_adjacent_points(curr.point)
            for point in adj:
                if (not self.is_point_inside_grid_map(point)) or (point in visited) or (self.is_obstacle(point)):
                    continue
                pq.put(GridWithCost(point, curr.cost_to_come + 1,self.calculate_heuristic(point)))
                to_visit.add(point)
                
            self.update_grid_map(visited, to_visit)
            self.img.set_array(self.grid_map)
            plt.draw()
            plt.pause(PAUSE_PERIOD)
        return False


    def breadth_first_search(self, source: Point, target: Point):
        visited = set()
        to_visit = set()
        queue = Queue()
        queue.put(source)
        to_visit.add(source)
        while not queue.empty():
            curr = queue.get()
            if curr in to_visit:
                to_visit.remove(curr)
            if curr in visited:
                continue
            visited.add(curr)
            value = self.grid_map[curr.x][curr.y]
            if value == TARGET_COLOR:
                return True
            adj = self.get_adjacent_points(curr)
            for point in adj:
                if (not self.is_point_inside_grid_map(point)) or (point in visited) or (self.is_obstacle(point)):
                    continue
                queue.put(point)
                to_visit.add(point)
                
            self.update_grid_map(visited, to_visit)
            self.img.set_array(self.grid_map)
            plt.draw()
            plt.pause(PAUSE_PERIOD)
        return False

    def depth_first_search(self, source: Point, target: Point) -> bool:
        visited = set()
        to_visit = set()
        stack = LifoQueue()
        stack.put(source)
        to_visit.add(source)
        while not stack.empty():
            curr = stack.get()
            if curr in to_visit:
                to_visit.remove(curr)
            if curr in visited:
                continue
            visited.add(curr)
            value = self.grid_map[curr.x][curr.y]
            if value == TARGET_COLOR:
                return True
            adj = self.get_adjacent_points(curr)
            for point in adj:
                if (not self.is_point_inside_grid_map(point)) or (point in visited) or (self.is_obstacle(point)):
                    continue
                stack.put(point)
                to_visit.add(point)

            self.update_grid_map(visited, to_visit)
            self.img.set_array(self.grid_map)
            plt.draw()
            plt.pause(PAUSE_PERIOD)
        return False

    def update_grid_map(self, visited: set, to_visit: set):
        for point in visited:
            if point == self.init_point or point == self.term_point:
                continue
            self.grid_map[point.x][point.y] = VISITED_COLOR
        for point in to_visit:
            if point == self.init_point or point == self.term_point or point in visited:
                continue
            self.grid_map[point.x][point.y] = TO_VISIT_COLOR

    def is_point_inside_grid_map(self, point:Point) -> bool:
        return point.x >= 0 and point.x < X and point.y >= 0 and point.y < Y


if __name__ == "__main__":
    X = 16
    Y = 16
    A = GameMap(X, Y)

    source_point = Point(5, 2)
    target_point = Point(15, 10)

    A.set_init_point(source_point.x, source_point.y)
    A.set_term_point(target_point.x, target_point.y)
    A.add_obstacle_line(Point(14,9), Point(14,6))
    A.add_obstacle_line(Point(13,3), Point(5,0))
    A.add_obstacle_line(Point(13,10), Point(9,14))
    A.show_map()
    # A.breadth_first_search(source_point, target_point)
    # A.depth_first_search(source_point,target_point)
    A.a_star(source_point,target_point)
    plt.show()
