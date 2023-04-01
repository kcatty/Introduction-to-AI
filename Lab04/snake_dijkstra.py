import numpy as np
from typing import List, Set
from dataclasses import dataclass
import pygame
from enum import Enum, unique
import sys
import random

FPS = 10

INIT_LENGTH = 4

WIDTH = 480
HEIGHT = 480
GRID_SIDE = 24
GRID_WIDTH = WIDTH // GRID_SIDE
GRID_HEIGHT = HEIGHT // GRID_SIDE

BRIGHT_BG = (103, 223, 235)
DARK_BG = (78, 165, 173)

SNAKE_COL = (6, 38, 7)
FOOD_COL = (224, 160, 38)
OBSTACLE_COL = (209, 59, 59)
VISITED_COL = (24, 42, 142)


@unique
class Direction(tuple, Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def reverse(self):
        x, y = self.value
        return Direction((x * -1, y * -1))


@dataclass
class Position:
    x: int
    y: int

    def check_bounds(self, width: int, height: int):
        return (self.x >= width) or (self.x < 0) or (self.y >= height) or (self.y < 0)

    def draw_node(self, surface: pygame.Surface, color: tuple, background: tuple):
        r = pygame.Rect(
            (int(self.x * GRID_SIDE), int(self.y * GRID_SIDE)), (GRID_SIDE, GRID_SIDE)
        )
        pygame.draw.rect(surface, color, r)
        pygame.draw.rect(surface, background, r, 1)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Position):
            return (self.x == o.x) and (self.y == o.y)
        else:
            return False

    def __str__(self):
        return f"X{self.x};Y{self.y};"

    def __hash__(self):
        return hash(str(self))


class GameNode:
    nodes: Set[Position] = set()

    def __init__(self):
        self.position = Position(0, 0)
        self.color = (0, 0, 0)

    def randomize_position(self):
        try:
            GameNode.nodes.remove(self.position)
        except KeyError:
            pass

        condidate_position = Position(
            random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1),
        )

        if condidate_position not in GameNode.nodes:
            self.position = condidate_position
            GameNode.nodes.add(self.position)
        else:
            self.randomize_position()

    def draw(self, surface: pygame.Surface):
        self.position.draw_node(surface, self.color, BRIGHT_BG)


class Food(GameNode):
    def __init__(self):
        super(Food, self).__init__()
        self.color = FOOD_COL
        self.randomize_position()


class Obstacle(GameNode):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.color = OBSTACLE_COL
        self.randomize_position()


class Snake:
    def __init__(self, screen_width, screen_height, init_length):
        self.color = SNAKE_COL
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.init_length = init_length
        self.reset()

    def reset(self):
        self.length = self.init_length
        self.positions = [Position((GRID_SIDE // 2), (GRID_SIDE // 2))]
        self.direction = random.choice([e for e in Direction])
        self.score = 0
        self.hasReset = True

    def get_head_position(self) -> Position:
        return self.positions[0]

    def turn(self, direction: Direction):
        if self.length > 1 and direction.reverse() == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        self.hasReset = False
        cur = self.get_head_position()
        x, y = self.direction.value
        new = Position(cur.x + x, cur.y + y, )
        if self.collide(new):
            self.reset()
        else:
            self.positions.insert(0, new)
            while len(self.positions) > self.length:
                self.positions.pop()

    def collide(self, new: Position):
        return (new in self.positions) or (new.check_bounds(GRID_WIDTH, GRID_HEIGHT))

    def eat(self, food: Food):
        if self.get_head_position() == food.position:
            self.length += 1
            self.score += 1
            while food.position in self.positions:
                food.randomize_position()

    def hit_obstacle(self, obstacle: Obstacle):
        if self.get_head_position() == obstacle.position:
            self.length -= 1
            self.score -= 1
            if self.length == 0:
                self.reset()

    def draw(self, surface: pygame.Surface):
        for p in self.positions:
            p.draw_node(surface, self.color, BRIGHT_BG)


class Player:
    def __init__(self) -> None:
        self.visited_color = VISITED_COL
        self.visited: Set[Position] = set()
        self.chosen_path: List[Direction] = []

    def move(self, snake: Snake) -> bool:
        try:
            next_step = self.chosen_path.pop(0)
            snake.turn(next_step)
            return False
        except IndexError:
            return True

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def turn(self, direction: Direction):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def draw_visited(self, surface: pygame.Surface):
        for p in self.visited:
            p.draw_node(surface, self.visited_color, BRIGHT_BG)


class SnakeGame:
    def __init__(self, snake: Snake, player: Player) -> None:
        pygame.init()
        pygame.display.set_caption("AIFundamentals - SnakeGame")

        self.snake = snake
        self.food = Food()
        self.obstacles: Set[Obstacle] = set()
        for _ in range(40):
            ob = Obstacle()
            while any([ob.position == o.position for o in self.obstacles]):
                ob.randomize_position()
            self.obstacles.add(ob)

        self.player = player

        self.fps_clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(
            (snake.screen_height, snake.screen_width), 0, 32
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.myfont = pygame.font.SysFont("monospace", 16)

    def drawGrid(self):
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                p = Position(x, y)
                if (x + y) % 2 == 0:
                    p.draw_node(self.surface, BRIGHT_BG, BRIGHT_BG)
                else:
                    p.draw_node(self.surface, DARK_BG, DARK_BG)

    def run(self):
        while not self.handle_events():
            self.fps_clock.tick(FPS)
            self.drawGrid()
            if self.player.move(self.snake) or self.snake.hasReset:
                self.player.search_path(self.snake, self.food, self.obstacles)
                self.player.move(self.snake)
            self.snake.move()
            self.snake.eat(self.food)
            for ob in self.obstacles:
                self.snake.hit_obstacle(ob)
            for ob in self.obstacles:
                ob.draw(self.surface)
            self.player.draw_visited(self.surface)
            self.snake.draw(self.surface)
            self.food.draw(self.surface)
            self.screen.blit(self.surface, (0, 0))
            text = self.myfont.render(
                "Score {0}".format(self.snake.score), 1, (0, 0, 0)
            )
            self.screen.blit(text, (5, 10))
            pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_UP:
                    self.player.turn(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.player.turn(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.player.turn(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.player.turn(Direction.RIGHT)
        return False


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()

    def turn(self, direction: Direction):
        self.chosen_path.append(direction)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------
obstacle_position = np.full((GRID_WIDTH, GRID_HEIGHT), False)


class Node:

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.is_obstacle = obstacle_position[x][y]
        if obstacle_position[x][y]:
            self.cost = 100
        else:
            self.cost = 1
        self.cumulatedCost = 0
        self.start = False
        self.goal = False
        self.open = False
        self.visited = False

        self.parent = None


class Dijkstra:

    def __init__(self, array, start, goal, snake):
        self.nodes = [[Node(col, row) for col in range(array.shape[1])] for row in range(array.shape[0])]
        self.nodes = np.array(self.nodes)

        self.start_node = self.nodes[start[0], start[1]]
        self.start_node.start = True

        self.goal_node = self.nodes[goal[0], goal[1]]
        self.goal_node.goal = True

        self.possible_nodes = [self.start_node]
        self.visited = [*(snake.positions)]
        self.goalReached = False

        self.heuristic_cost = abs(self.start_node.x - self.goal_node.x) + abs(self.start_node.y - self.goal_node.y) + 1
        self.cost = 0

        self.current_node = self.start_node
        self.open_node(self.current_node)

    def search(self):
        while self.goalReached is False:
            x = self.current_node.x
            y = self.current_node.y

            self.current_node.visited = True
            self.visited.append(self.current_node)
            self.possible_nodes.remove(self.current_node)

            if x - 1 >= 0:
                self.open_node(self.nodes[y, x - 1])
            if x + 1 < GRID_WIDTH:
                self.open_node(self.nodes[y, x + 1])
            if y - 1 >= 0:
                self.open_node(self.nodes[y - 1, x])
            if y + 1 < GRID_HEIGHT:
                self.open_node(self.nodes[y + 1, x])

            bestNodeIndex = 0
            bestNodeCost = self.heuristic_cost + self.cost

            for index, node in enumerate(self.possible_nodes):
                if node.cumulatedCost < bestNodeCost:
                    bestNodeIndex = index
                    bestNodeCost = node.cumulatedCost

            if len(self.possible_nodes) < 1:
                print("Not found")

            self.current_node = self.possible_nodes[bestNodeIndex]
            if self.current_node == self.goal_node:
                self.goalReached = True
                print(self.cost)
                print("Found")

    def open_node(self, node):
        if node.open is False and node.visited is False:
            for visited_node in self.visited:
                if visited_node.x == node.x and visited_node.y == node.y:
                    return
            node.open = True
            node.parent = self.current_node
            node.cumulatedCost = node.cost + self.current_node.cumulatedCost
            self.possible_nodes.append(node)

    def get_path(self):
        current = self.goal_node
        path = []
        while current != self.start_node:
            if current is None:
                continue
            path.append((current.y, current.x))
            current = current.parent
        return path[::-1]


class SearchBasedPlayer(Player):
    def __init__(self):
        super(SearchBasedPlayer, self).__init__()

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        board = np.ones((GRID_WIDTH, GRID_HEIGHT))
        obstacle_positions = [(obstacle.position.x, obstacle.position.y) for obstacle in obstacles[0]]
        for y in range(0, GRID_HEIGHT):
            for x in range(0, GRID_WIDTH):
                tile = Position(x, y)
                if tile in GameNode.nodes:
                    board[y, x] = 1
                else:
                    board[y, x] = 0
                if (x, y) in obstacle_positions:
                    obstacle_position[y, x] = True
                else:
                    obstacle_position[y, x] = False

        start_position = (snake.get_head_position().x, snake.get_head_position().y)
        end_position = (food.position.x, food.position.y)
        dijkstra = Dijkstra(np.array(board), start_position, end_position, snake)
        dijkstra.search()
        parent = dijkstra.get_path()
        path_dirs = []
        previous = [snake.get_head_position().x, snake.get_head_position().y]
        while len(parent) > 0:
            path = parent.pop(0)
            if path[0] > previous[0]:
                path_dirs.append(Direction.RIGHT)
            elif path[0] < previous[0]:
                path_dirs.append(Direction.LEFT)
            elif path[1] > previous[1]:
                path_dirs.append(Direction.DOWN)
            elif path[1] < previous[1]:
                path_dirs.append(Direction.UP)
            previous = path
        self.chosen_path = path_dirs


if __name__ == "__main__":
    snake = Snake(WIDTH, WIDTH, INIT_LENGTH)
    player = SearchBasedPlayer()
    game = SnakeGame(snake, player)
    game.run()
