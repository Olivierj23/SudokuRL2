import torch
import numpy as np
import random
from collections import deque
from game import SudokuGameAI, get_sudoku_game
from model import Linear_QNet, QTrainer, SudoCNN
from helper import plot
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display



MAX_Memory = 100000
BATCH_SIZE = 1000
LR = 0.05

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_Memory)
        self.model = Linear_QNet(324, 1024, 729)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # need to return an 81*4 vector (81 positions and 4 bits to represent the value from 0 (nothing there) to 1-9

        board = game.board

        state_vector = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == ".":
                    state_vector += [0, 0, 0, 0]
                else:
                    number_embedding = bin(int(board[i][j]))[2:].zfill(4)
                    state_vector += [int(x) for x in number_embedding]

        return state_vector



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # epsilon-greedy policy

        self.epsilon = 0.20
        if random.random() < self.epsilon:
            for i in range(81):
                if sum(state[4*i : 4*i + 4]) == 0:
                    location = i
                    break
            number = random.randint(1, 9)

            final_action = bin(int(location))[2:].zfill(7) + bin(int(number))[2:].zfill(4)
            final_action = [int(x) for x in final_action]

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_index = torch.argmax(prediction).item()
            location = move_index // 81
            number = move_index % 9 + 1

            final_action = bin(int(location))[2:].zfill(7) + bin(int(number))[2:].zfill(4)
            final_action = [int(x) for x in final_action]

        return final_action

class SudokuDataset(Dataset):
    def __init__(self, train=True):
        self.sudoku_dataframe = pd.read_csv("sudoku-3m.csv", nrows=100000)


    def __len__(self):
        return len(self.sudoku_dataframe)
    def __getitem__(self, idx):
        puzzle = self.sudoku_dataframe.at[idx, "puzzle"]
        solution = self.sudoku_dataframe.at[idx, "solution"]
        puzzle_list = []
        solution_list = []
        for i in range(len(puzzle)):
            if puzzle[i] == ".":
                puzzle_list.append(float(0))
            else:
                puzzle_list.append(float(puzzle[i]))

            if solution[i] == ".":
                solution_list.append(0)
            else:
                solution_list.append(int(solution[i]))


        puzzle_array = np.array(puzzle_list, dtype=float).reshape((9, 9))
        solution_array = np.array(solution_list).reshape((9, 9))
        solution_vector = []

        for i in range(len(solution_array)):
            for j in range(len(solution_array[0])):
                if solution_array[i][j] == ".":
                    solution_vector += [0, 0, 0, 0]
                else:
                    number_embedding = bin(int(solution_array[i][j]))[2:].zfill(4)
                    solution_vector += [int(x) for x in number_embedding]

        puzzle_tensor = torch.from_numpy(puzzle_array)
        solution_tensor = torch.tensor(solution_vector, dtype=torch.float)


        return puzzle_tensor, solution_tensor
def RL_train():
    games_queue = []
    plot_number_solved = []
    plot_average_number_solved = []
    total_solved = 0
    record = 0
    agent = Agent()
    game = SudokuGameAI()
    while True:
        # get old state
        print(np.array(game.board))
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            if game.is_loss():
                games_queue.append(1)
            else:
                games_queue.append(0)

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if len(games_queue) == 20:
                plot_number_solved.append(sum(games_queue))
                total_solved += sum(games_queue)

                if sum(games_queue) > record:
                    record = max(sum(games_queue), record)
                    agent.model.save()

                print("Game", agent.n_games, "Puzzles Solved of Batch", sum(games_queue), "Record:", record)
                games_queue = []

                # plot

                plot_average_number_solved.append(total_solved/agent.n_games)
                plot(plot_number_solved, plot_average_number_solved)

def CNN_train():
    plt.ion()
    model = SudoCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    training_data = SudokuDataset()
    train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
    plot_running_loss = []
    for epoch in range(1000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            puzzles, solutions = data
            puzzles = torch.reshape(puzzles, (10, 1, 9, 9))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(puzzles)
            loss = criterion(outputs, solutions)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                plot_running_loss.append(running_loss/100)
                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.clf()
                plt.plot(plot_running_loss) 
                running_loss = 0.0

    print('Finished Training')



if __name__ == "__main__":
    # training_data = SudokuDataset()
    # train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
    # puzzle, solution = next(iter(train_dataloader))
    # print(puzzle)
    # print(solution.shape)
    CNN_train()

