# See tutorial https://www.gymlibrary.dev/content/environment_creation/

import os
import time
import random
import gymnasium as gym
import wandb
from gymnasium import spaces
import numpy as np
from GameSimulation import replay_game_video


class BoardSpot(object):
    """
    The BoardSpot class represents a single spot in the mine sweeper game.
    """

    def __init__(self):
        self.value = 0
        self.selected = False
        self.mine = False

    def __str__(self):
        return str(BoardSpot.value)


class MinesweeperEnvMask(gym.Env):
    """
    Custom Environment for Minesweeper with lies.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, board_size=9, num_mines=10, num_lies=0):

        super(MinesweeperEnvMask, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(board_size ** 2)
        # TODO: check MultiDiscrete vs Discrete later
        # self.action_space = spaces.MultiDiscrete([board_size, board_size])
        self.observation_space = spaces.Box(low=-1, high=8, shape=(board_size, board_size), dtype=np.int32)

        # Set board size, number of mines, lies, selectable spots, and board
        self.board_size = board_size
        self.num_mines = num_mines if num_mines < 1/2 * board_size ** 2 else 10
        self.num_lies = num_lies
        self.lie_list = []
        self.selectable_spots = board_size ** 2 - num_mines
        self.board = [[BoardSpot() for i in range(self.board_size)] for j in range(self.board_size)]
        self.steps = 0
        self.game_count = 0
        self.last_game_info = None
        self.win_games = 0
        # Keep track of avg steps and remaining spots
        self.total_steps = 0
        self.total_remaining_spots = 0
        # Keep track of the game progress
        self.moves_list = []
        self.reset_next_game = True

    def _add_mine(self, x, y):
        """
        This helper function add mines to the given position (x, y).
        Additionally, it will add 1 to the value of all surrounding spots.
            X X X
            X O X
            X X X
        O is the spot (x, y)
        X are spots whose values should be incremented
        """
        # Set mines to location (x, y)
        self.board[x][y].value = -1
        self.board[x][y].mine = True
        # Modify values for surrounding spots
        for i in range(x-1, x+2):
            # Make sure the index is within the board
            if 0 <= i < self.board_size:
                # For a spot that is not a mine, we need to change its value
                if y-1 >= 0 and not self.board[i][y-1].mine:
                    self.board[i][y-1].value += 1
                if y+1 < self.board_size and not self.board[i][y+1].mine:
                    self.board[i][y+1].value += 1
        # Make changes to the remaining two spots
        if x-1 >= 0 and not self.board[x-1][y].mine:
            self.board[x-1][y].value += 1
        if x+1 < self.board_size and not self.board[x+1][y].mine:
            self.board[x+1][y].value += 1

    def _make_lies(self):
        """
        This helper function adds lies to the mine game. It will add number of lies according to self.num_lies.
        To make the game easier, the lies will NOT be:  1 -> 0  or  0 -> 1
        """
        i = 0
        lie_list = []
        while i < self.num_lies:
            x = self.np_random.integers(0, self.board_size-1)
            y = self.np_random.integers(0, self.board_size-1)
            if self.board[x][y].value > 0 and (x, y) not in lie_list:
                # Subtractions is only valid for value >= 2
                if self.board[x][y].value > 1:
                    self.board[x][y].value += self.np_random.choice([-1, 1])
                else:
                    self.board[x][y].value += 1
                lie_list += [(x, y)]
                i += 1
        self.lie_list = lie_list

    def _make_move(self, x, y):
        """
        This helper function makes a move at (x, y)
        """
        self.board[x][y].selected = True
        self.selectable_spots -= 1
        if self.board[x][y].value == -1:
            # Hit a mine
            return False
        # On a 0 spot, automatically opens all surrounding spots
        if self.board[x][y].value == 0:
            for i in range(x-1, x+2):
                if 0 <= i < self.board_size:
                    if y-1 >= 0 and not self.board[i][y-1].selected:
                        self._make_move(i, y-1)
                    if y+1 < self.board_size and not self.board[i][y+1].selected:
                        self._make_move(i, y+1)
            if x-1 >= 0 and not self.board[x-1][y].selected:
                self._make_move(x-1, y)
            if x+1 < self.board_size and not self.board[x+1][y].selected:
                self._make_move(x+1, y)
            return True
        else:
            return True

    def _is_winner(self):
        # TODO: ==0 or <=0 ?
        return self.selectable_spots <= 0

    def _get_obs(self):
        # Get the observation
        observation = np.array([[spot.value if spot.selected else -1 for spot in row] for row in self.board])
        observation = np.transpose(observation)
        return observation

    def _get_info(self):
        # Get the information
        information = {
            'mines': self.num_mines,
            'lies': self.num_lies,
            'remaining spots': self.selectable_spots,
            'steps': self.steps,
            'successful games': self.win_games,
            'total games': self.game_count,
            'win rate': 0 if self.game_count == 0 else (self.win_games / self.game_count)
        }
        return information

    def _check_neighbor_spots(self, x, y):
        """
        For a spot (x, y), check whether at least one neighbor is selected.
        This function should be called before making move at (x, y)
        """
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                try:
                    if self.board[i][j].selected:
                        return True
                except:
                    # This happens on the boundary
                    pass

        # None of the neighbors are selected
        return False

    def _save_game(self, win):

        game_dir = 'games_mask'
        if win:
            game_dir = os.path.join(game_dir, 'win')
        else:
            game_dir = os.path.join(game_dir, 'lose')
        if not os.path.exists(game_dir):
            os.makedirs(game_dir)

        game_file_name = os.path.join(game_dir, f'game{int(time.time())}_{self.game_count}')
        print('Saving game history to', game_file_name)

        # Get the board
        board = np.array([[spot.value for spot in row] for row in self.board])
        board = np.transpose(board)

        # Save game replay video
        replay_game_video(board, self.moves_list, filename=game_file_name+'.mp4')

        # Save game to a file
        with open(game_file_name+'.txt', 'w') as file:

            # Saving the board
            np.savetxt(file, board, fmt='%d')
            file.write('# Moves\n')

            # Saving the list of ordered pairs
            for pair in self.moves_list:
                file.write(f'{pair[0]} {pair[1]}\n')

        # Return the file name
        return game_file_name

    def get_last_game_info(self):
        """
        This function returns the information of last game at the beginning of the next game.
        The callback function is not called at the end of an episode(i.e game). In order to track the progress
        of the games, the relevant information can only be passed at the beginning of the next game.
        """
        if self.steps < 1:
            return self.last_game_info
        else:
            return None

    def action_masks(self):
        mask = np.array([[not spot.selected for spot in row] for row in self.board])
        mask = np.transpose(mask)
        # TODO: check the consistency with action space
        mask = mask.reshape(-1)
        return mask

    def reset(self, seed=None, options=None):
        """
        The `done` variable is always set to true for training purposes, so the reset() function
        will be called every step. However, we start a new game only when the last game finishes.
        """
        if self.reset_next_game:
            # Clear the flag
            self.reset_next_game = False

            super().reset(seed=seed, options=options)
            # Reset the state of the environment to an initial state
            self.selectable_spots = self.board_size ** 2 - self.num_mines
            self.board = [[BoardSpot() for i in range(self.board_size)] for j in range(self.board_size)]
            self.steps = 0
            self.game_count += 1
            self.moves_list = []
            i = 0
            while i < self.num_mines:
                # Generate random x y coordinates for the mine position
                x = self.np_random.integers(0, self.board_size, dtype=np.int32)
                y = self.np_random.integers(0, self.board_size, dtype=np.int32)
                # Check if this spot is already a mine.
                if not self.board[x][y].mine:
                    # Call helper function to add a mine.
                    self._add_mine(x, y)
                    i += 1
            # Add lies to the mine game
            self._make_lies()
            # Return the current observation and information
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Execute one time step within the environment
        # TODO: check the consistency with action space
        # This works with 1D action space.
        x = action % self.board_size
        y = action // self.board_size

        '''
        # This works with 2D action space.
        x, y = action
        '''

        self.steps += 1
        self.moves_list += [(x, y)]

        # Punishment for going too far away
        reward = 0 if self._check_neighbor_spots(x, y) else -3.5

        # Lose if hit a mine (i.e. _make_move() returns False)
        loseGame = not self._make_move(x, y)
        winGame = False if loseGame else self._is_winner()
        terminated = loseGame or winGame
        if winGame:
            self.win_games += 1
        if terminated:
            # Set the flag
            self.reset_next_game = True
            # Save relevant information
            self.total_steps += self.steps
            self.total_remaining_spots += self.selectable_spots
            info = self._get_info()
            info['Average steps'] = self.total_steps / self.game_count
            info['Average remaining spots'] = self.total_remaining_spots / self.game_count
            # Save the current game with probability
            probability = 0.0001
            if random.random() < probability:
                name = self._save_game(win=winGame)
                # Only upload "good" games to wandb
                if self.selectable_spots < 5:
                    info['video'] = wandb.Video(name+'.mp4')
            self.last_game_info = info

        if winGame:
            # Win the game
            reward += 10
        elif loseGame:
            # Hit a mine and lose the game
            reward += -10
        else:
            # Clear a spot without hitting a mine
            reward += 1

        # This step should end immediately
        return self._get_obs(), reward, True, False, self._get_info()

    def render(self, mode='console'):
        returnString = " "
        divider = "\n---"

        for i in range(0, self.board_size):
            returnString += " | " + str(i)
            divider += "----"
        divider += "\n"

        returnString += divider
        for y in range(0, self.board_size):
            returnString += str(y)
            for x in range(0, self.board_size):
                if self.board[x][y].mine and self.board[x][y].selected:
                    returnString += " |" + str(self.board[x][y].value)
                elif self.board[x][y].selected:
                    returnString += " | " + str(self.board[x][y].value)
                else:
                    returnString += " |  "
            returnString += " |"
            returnString += divider
        print(returnString)

    def close(self):
        # Clean up when closing the environment
        pass
