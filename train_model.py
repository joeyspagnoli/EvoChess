import numpy as np
import pandas as pd
import chess
import re

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

def create_move_list(s):
    return re.sub(r'\d*\. ', '', s).split(' ')[:-1]

def test_board_methods():
    board = chess.Board()

    print(board)
    matrix_board = board_to_matrix(board)
    print(matrix_board)

    x = move_to_rep("e4", board)
    board.push_san("e4")

    print(x)

    print(board)

    moves_str1 = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"

    print(create_move_list(moves_str1))

    moves_str2 = "1. e4 c5 2. Nf3 d6 3. d4 cxd4"

    print(create_move_list(moves_str2))

#test_board_methods()

class ChessDataset(Dataset):

    def __init__(self, games, return_board=False):
        super(ChessDataset, self).__init__()
        self.games = games
        self.return_board = return_board

    def __len__(self):
        return 40_000

    def __getitem__(self, index):
        game_i = np.random.randint(len(self.games))
        random_game = chess_data['AN'].values[game_i]
        moves = create_move_list(random_game)
        game_state_i = np.random.randint(len(moves) - 1)
        next_move = moves[game_state_i]
        moves = moves[:game_state_i]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        x = board_to_matrix(board)
        y = move_to_rep(next_move, board)
        if game_state_i % 2 == 1:
            x *= -1
        if self.return_board:
            return x, y, board
        else:
            return x, y




#The CNN!

class module(nn.Module):

    def __init__(self, hidden_size):

        super(module, self).__init__()
        #Two convolutional layers
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)

        #Two batch normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        #Two SELU activation layers
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += x_input

        x = self.activation2(x)

        return x

class chessBot(nn.Module):

    def __init__(self, hidden_layers = 4, hidden_size=200):
        super(chessBot, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = F.relu(x)

        for module in self.module_list:
            x = module(x)

        x = self.output_layer(x)

        return x

#Helper functions to decide moves

def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        board.pop()

def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()

    probs = probs ** 3
    probs = probs / probs.sum()

    return probs

def pred_to_move(move, board):
#     move = move[0]

    legal_moves = list(board.legal_moves)

    check_mate = check_mate_single(board)

    if check_mate is not None:
        return check_mate

    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))

    for from_ in froms:
        row = 8 - int(from_[1])
        col = letter_to_num[from_[0]]
        val = move[0, row, col].detach().cpu().numpy()
        vals.append(val)

    probs = distribution_over_moves(vals)
    chosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        if from_ == chosen_from:
            to = str(legal_move)[2:]
            r = 8 - int(to[1])
            c = letter_to_num[to[0]]
            val = move[1, r, c].detach().cpu().numpy()
            vals.append(val)
        else:
            vals.append(0)

    chosen_move = legal_moves[np.argmax(vals)]
    return chosen_move

def custom_collate(batch):
    board_states, correct_moves, boards = zip(*batch)
    board_states = torch.stack([torch.tensor(x) for x in board_states])
    correct_moves = torch.stack([torch.tensor(x) for x in correct_moves])
    # Leave boards as a tuple of chess.Board objects
    return board_states, correct_moves, boards


def train_loop(dataloader, model, metric_from, metric_to, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (board_state, correct_move) in enumerate(dataloader):

        board_state, correct_move = board_state.to(device), correct_move.to(device)

        # Compute prediction and loss
        pred = model(board_state.float())

        loss = metric_from(pred[:,0,:,:], correct_move[:,0,:,:]) + metric_to(pred[:,1,:,:], correct_move[:,1,:,:])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * batch_size + len(board_state)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model):

    num_guesses = 0
    correct = 0
    batch_number = 0

    model.eval()

    with torch.no_grad():
        for board_state, correct_move, boards in dataloader:

            board_state, correct_move = board_state.to(device), correct_move.to(device)

            preds = model(board_state.float())

#             print("new batch of predictions, batch number: ", batch_number)
            for i, pred in enumerate(preds):
                board_obj = boards[i]
                UCI = pred_to_move(pred, board_obj)
                san_move = board_obj.san(UCI)

                matrix_move = move_to_rep(san_move, board_obj)   # get one prediction
                correct_move_np = correct_move[i].cpu().numpy()    # get the corresponding correct move

                if np.array_equal(matrix_move, correct_move_np):
                    correct += 1

                num_guesses += 1
#             print("num correct", correct)
#             print(" ")

    print("number of total guesses: ", num_guesses)
    print("number of correct guesses: ", correct)
    print("accuracy: ", (correct/num_guesses)*100)



def get_ai_move(board, model, device):
    mat_move = board_to_matrix(board)
    mat_move = np.expand_dims(mat_move, axis=0)  # Add a batch dimension
    mat_move_tensor = torch.tensor(mat_move)  # Convert to PyTorch tensor
    move = model(mat_move_tensor.float().to(device))
    uci = pred_to_move(move[0], board)
    return uci

def get_player_move(board):
    while True:
        user_move = input("Enter your move (in algebraic notation, e.g., e2e4): ")
        if user_move in [move.uci() for move in board.legal_moves]:
            return user_move
        else:
            print("Invalid move. Please enter a valid move.")


def game_loop():
    board = chess.Board()
    print(board)

    while not board.is_game_over():
        ai_move = get_ai_move(board, model)
        board.push_uci(str(ai_move))
        print("AI's move:", ai_move)
        print(board)

        if board.is_game_over():
            break

        # Player's move|
        player_move = get_player_move(board)
        board.push_uci(str(player_move))
        print("Player's move:", player_move)
        print(board)

def train():
    #read data
    chess_data = pd.read_csv('chess_games.csv', usecols=['AN', 'WhiteElo']) #Only columns needed

    #Filter the data to only games w/ players above 2000 elo, only save moves and of games where there are a lot of moves(?)
    chess_data = chess_data[chess_data['WhiteElo'] > 2000]
    chess_data = chess_data[['AN']]
    chess_data = chess_data[~chess_data['AN'].str.contains('{')]
    chess_data = chess_data[chess_data['AN'].str.len() > 20]

    train_size = int(0.8 * len(chess_data))
    test_size = len(chess_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(chess_data, [train_size, test_size])

    data_train = ChessDataset(train_dataset, return_board=False)
    data_test = ChessDataset(test_dataset, return_board=True)

    data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)
    data_test_loader = DataLoader(data_test, batch_size=32, shuffle=True, drop_last=True, collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = chessBot().to(device)

    learning_rate = 1e-2
    batch_size = 32

    metric_from = nn.CrossEntropyLoss()
    metric_to = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(data_train_loader, model, metric_from, metric_to, optimizer)
        test_loop(data_test_loader, model)
        if optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step()

    print("Done!")

    torch.save(model.state_dict(), 'model.pth')









