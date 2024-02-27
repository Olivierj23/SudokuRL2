import random
import numpy as np
import pandas as pd

sudoku_puzzles_df = pd.read_csv("sudoku-3m.csv", nrows=100000)

def get_sudoku_game():
    sudoku_string = sudoku_puzzles_df.at[random.randint(0, 999), "puzzle"]

    sudoku_array = np.array(list(sudoku_string)).reshape((9,9))

    return sudoku_array.tolist()



class SudokuGameAI:
    def __init__(self):
        self.board = None
        self.reset()

    def reset(self):
        self.board = get_sudoku_game()
        self.frame_iteration = 0

    def reward_calculator(self, is_bad_move):
        if self.is_over():
            if self.is_loss():
                return -100
            else:
                return +100
        elif is_bad_move:
            return -10
        else:
            for row in self.board:
                unique_nums = set()
                list_nums = []
                for x in row:
                    if x != ".":
                        unique_nums.add(x)
                        list_nums.append(x)

                if len(unique_nums) != len(list_nums):
                    return -10

            return 0

    def game_iteration(self):
        pass

    def is_loss(self):
        def distinct(input_list):
            list_to_compare = []
            for elem in input_list:
                if elem in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    list_to_compare.append(elem)
                else:
                    pass
            return len(set(list_to_compare)) == len(list_to_compare)

        for i in self.board:
            if distinct(i) == False:
                return True
        for i in range(9):
            input_list = []
            for j in range(9):
                input_list.append(self.board[j][i])
            if distinct(input_list) == False:
                return True

        for x in range(3):
            for y in range(3):
                input_list = []
                for i in range(3):
                    for j in range(3):
                        input_list.append(self.board[3 * x + i][3 * y + j])
                if distinct(input_list) == False:
                    return True
        else:
            return False

    def is_over(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == ".":
                    return False
        return True


    def play_step(self, action):
        self.frame_iteration += 1
        location_encoding = action[:7]
        number_encoding = action[7:]

        position = min(int("".join(str(x) for x in location_encoding), 2), 80)
        row = position // 9
        col = position % 9

        number = int("".join(str(x) for x in number_encoding), 2)

        if self.board[row][col] == ".":
            self.board[row][col] = number
            reward = self.reward_calculator(False)
            return reward, self.is_over()

        else:
            reward = self.reward_calculator(True)
            return reward, self.is_over()


if __name__ == "__main__":
    game = SudokuGameAI()
    print(game.is_loss(), game.is_over(False))