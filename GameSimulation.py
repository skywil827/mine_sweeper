import numpy as np
from PIL import Image
import imageio
import cv2
import sys


class SimulationBoardSpot(object):

    def __init__(self, value):
        self.value = value
        self.selected = False
        self.mine = (value == -1)

    def __str__(self):
        return str(SimulationBoardSpot.value)


class SimulationBoardClass(object):
    def __init__(self, board):
        board_size = len(board)
        self.board_size = board_size
        self.board = [[SimulationBoardSpot(board[i][j]) for i in range(board_size)] for j in range(board_size)]

    def __str__(self):
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
        return returnString

    def print_complete_board(self):
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
                if self.board[x][y].mine:
                    returnString += " | " + "*"
                else:
                    returnString += " | " + str(self.board[x][y].value)
            returnString += " |"
            returnString += divider
        return returnString

    def make_move(self, x, y):
        self.board[x][y].selected = True
        if self.board[x][y].value == -1:
            return False
        if self.board[x][y].value == 0:
            for i in range(x-1, x+2):
                if 0 <= i < self.board_size:
                    if y-1 >= 0 and not self.board[i][y-1].selected:
                        self.make_move(i, y-1)
                    if y+1 < self.board_size and not self.board[i][y+1].selected:
                        self.make_move(i, y+1)
            if x-1 >= 0 and not self.board[x-1][y].selected:
                self.make_move(x-1, y)
            if x+1 < self.board_size and not self.board[x+1][y].selected:
                self.make_move(x+1, y)
            return True
        else:
            return True

    def get_board_image(self):
        # Load images for each cell state
        cell_images = {
            'hidden': Image.open("img/hidden.jpg"),
            'mine': Image.open("img/mine.jpg"),
            # Load images for numbers 0 to 8
            **{str(i): Image.open(f"img/{i}.jpg") for i in range(9)}
        }

        # Size of each cell in pixels
        cell_size = 32

        # Create a new blank image for the board
        board_image = Image.new('RGB', (self.board_size * cell_size, self.board_size * cell_size))

        for x in range(self.board_size):
            for y in range(self.board_size):
                cell = self.board[x][y]
                if cell.selected:
                    if cell.mine:
                        img = cell_images['mine']
                    else:
                        img = cell_images[str(cell.value)]
                else:
                    img = cell_images['hidden']

                # Paste the cell image into the correct position
                board_image.paste(img, (x * cell_size, y * cell_size))

        return board_image


def replay_game_console(board, moves_list):
    # Reconstruct the board
    new_board = SimulationBoardClass(board)
    print('The board is:')
    print(new_board.print_complete_board())
    for move in moves_list:
        x, y = move
        print('Move:', move, '\n')
        print(new_board)
        new_board.make_move(x, y)


def replay_game_video(board, moves_list, filename='game_replay.mp4'):
    # Reconstruct the board
    new_board = SimulationBoardClass(board)
    # Show initial image
    # new_board.get_board_image().save('x.jpg')
    frames = [new_board.get_board_image()]

    for move in moves_list:
        x, y = move
        new_board.make_move(x, y)
        frames += [new_board.get_board_image()]

    # Define the codec and create VideoWriter object
    height, width = frames[0].size

    # Using 'avc1' as the codec and specifying the .mp4 extension for the output file
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'avc1'), 1, (width, height))

    for frame in frames:
        # Convert PIL image to OpenCV image
        open_cv_image = np.array(frame)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        video.write(open_cv_image)

    video.release()


def load_game(game_file):
    # Load data from a file
    with open(game_file, 'r') as file:
        # Read the board
        board = []
        for line in file:
            if line.startswith('#'):
                break
            board.append(line)
        board = np.loadtxt(board, dtype=int)

        # Read the list of ordered pairs
        moves_list = []
        for line in file:
            if line.strip():  # Skip empty lines
                x, y = map(int, line.split())
                moves_list.append((x, y))

    return board, moves_list


if __name__ == "__main__":

# Get the first argument after the script name
    if len(sys.argv) > 1:
        game_file = sys.argv[1]
    else:
        game_file = r'games_mask\win\game1704859214_1300.txt'
    board, moves_list = load_game(game_file)
    video_file = game_file + ".mp4"
    replay_game_console(board, moves_list)
    replay_game_video(board, moves_list, filename=video_file)
    print(f"video replay for game {game_file} is saved to file [{video_file}].")
