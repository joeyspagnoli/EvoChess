import torch
import numpy as np
import pandas as pd
import chess
import re

#Chess columns are notated by a-h, but matrix is noted by #s. Will be used to translate chess columns to matrix equivalent
letter_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

#We utilize a CNN as it can process 3D information which is vital to both specify location and importance of a chess piece




def create_rep_layer(board, piece):

    #Isolates specified piece on a board layout and zeros out every other piece
    s = str(board)
    s = re.sub(f'[^{piece}{piece.upper()} \n]', '0', s)
    s = re.sub(f'{piece}', '-1', s)
    s = re.sub(f'{piece.upper()}', '1', s)

    #Convert to a numpy matrix
    board_mat = []
    for row in s.split('\n'): #Iterate through 1 row
        row = row.split(' ') #Seperate each value in the row
        row = [int(x) for x in row] #Convert all strings into ints
        board_mat.append(row) #Add to board rep

    return np.array(board_mat)


def board_to_matrix(board):
    pieces = ['p', 'r', 'k', 'q', 'n', 'b']
    layers = []

    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep.astype(float)


def move_to_rep(move, board):

    #Grabs representation of piece moving 'from' one square 'to' new square
    #in a 4 character setup ie d4e5 means a piece moved from d4 to e5
    board.push_san(move).uci()
    move = str(board.pop())


    from_output_layer = np.zeros((8,8))

    #Turn board rep into matrix rep
    from_row = 8 - int(move[1])
    from_column = letter_to_num[move[0]]


    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    to_column = letter_to_num[move[2]]

    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])






board = chess.Board()

print(board)
matrix_board = board_to_matrix(board)
print(matrix_board)




