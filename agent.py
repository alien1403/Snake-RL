import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Directions, Point
from model import Linear_QNet, QTrainer
from helper import plot
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate used on Bellman Equation
        self.memory = deque(maxlen=MAX_MEMORY) # type: ignore
        
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)
    
    def get_state(self, game): # type: ignore
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Directions.LEFT
        dir_r = game.direction == Directions.RIGHT
        dir_u = game.direction == Directions.UP
        dir_d = game.direction == Directions.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done): #type:ignore
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):  #type: ignore
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state): # type: ignore
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 #type: ignore
        return final_move

def train():
    plot_scores = []
    plot_mean = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    plt.ion()
    plt.figure()
    while True:
        
        # get old state
        old_state = agent.get_state(game) #type:ignore
        
        #get move
        final_move = agent.get_action(old_state)
        
        # perform move and get the new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)
        
        #train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done) 
        
        #remember
        agent.remember(old_state, final_move, reward, new_state, done)
        
        if done:
            # train long memory & plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game', agent.n_games, 'Score: ', score, 'Record: ', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean.append(mean_score)
            plot(plot_scores, plot_mean)
            
            
if __name__ == "__main__":
    train()
