import pygame
import random
from enum import Enum
from typing import NamedTuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

# reset
# reward
# play(action) -> direction
# game_iteration 
# is_collision

class Directions(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    

class Point(NamedTuple):
    x: int
    y: int

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BLOCK_SIZE = 20
SPEED = 20

class SnakeGameAI:
    def __init__(self, w: int = 640, h: int = 480):
        self.high_score = self._load_high_score()
    
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def _load_high_score(self):
        try:
            with open('snake_high_score.txt', 'r') as f:
                return int(f.read())
        except Exception:
            return 0

    def _save_high_score(self):
        with open('snake_high_score.txt', 'w') as f:
            f.write(str(self.high_score))
        
    def reset(self):
        self.direction = Directions.RIGHT
        self.head = Point(int(self.w/2), int(self.h/2))
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE       
        self.food = Point(x,y)
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action): 
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None): 
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0: 
            return True
        
        if pt in self.snake[1:]:
            return True

        return False
    
    def _update_ui(self):
        self.display.fill(BLACK)
        for i, pt in enumerate(self.snake):
            color = (0, 50 + min(205, i*10), 255 - min(205, i*10))
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), border_radius=8)
            # Add eyes to the head
            if i == 0:
                eye_radius = 3
                eye_offset_x = BLOCK_SIZE // 4
                eye_offset_y = BLOCK_SIZE // 4
                pygame.draw.circle(self.display, WHITE, (pt.x + eye_offset_x, pt.y + eye_offset_y), eye_radius)
                pygame.draw.circle(self.display, WHITE, (pt.x + BLOCK_SIZE - eye_offset_x, pt.y + eye_offset_y), eye_radius)
        # glowing effect
        if self.food is not None:
            for glow in range(6, 0, -2):
                pygame.draw.circle(self.display, (255, 50, 50, 100), (self.food.x + BLOCK_SIZE//2, self.food.y + BLOCK_SIZE//2), BLOCK_SIZE//2 + glow)
            pygame.draw.circle(self.display, RED, (self.food.x + BLOCK_SIZE//2, self.food.y + BLOCK_SIZE//2), BLOCK_SIZE//2)
        # Score and high score
        text = font.render(f"Score: {self.score}  High Score: {self.high_score}", True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()
            
    def _move(self, action): 
        
        clock_wise = [Directions.RIGHT, Directions.DOWN, Directions.LEFT, Directions.UP]
        
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1,0,0]): 
            new_dir = clock_wise[idx]
        if np.array_equal(action, [0,1,0]): 
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]
            
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        
        if self.direction == Directions.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Directions.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Directions.UP:
            y -= BLOCK_SIZE
        elif self.direction == Directions.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x,y)