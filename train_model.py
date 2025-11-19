import numpy as np
import pandas as pd
import chess
import re

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow
import mlflow.pytorch

# Chess columns are notated by a-h, but matrix is noted by #s. Will be used to translate chess columns to matrix equivalent
letter_to_num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
num_to_letter = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}

# We utilize a CNN as it can process 3D information which is vital to both specify location and importance of a chess piece


def create_rep_layer(board, piece):
    # Isolates specified piece on a board layout and zeros out every other piece
    s = str(board)
    s = re.sub(f"[^{piece}{piece.upper()} \n]", "0", s)
    s = re.sub(f"{piece}", "-1", s)
    s = re.sub(f"{piece.upper()}", "1", s)

    # Convert to a numpy matrix
    board_mat = []
    for row in s.split("\n"):  # Iterate through 1 row
        row = row.split(" ")  # Seperate each value in the row
        row = [int(x) for x in row]  # Convert all strings into ints
        board_mat.append(row)  # Add to board rep

    return np.array(board_mat)


def board_to_matrix(board):
    pieces = ["p", "r", "k", "q", "n", "b"]
    layers = []

    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep.astype(float)


def move_to_rep(move, board):
    # Grabs representation of piece moving 'from' one square 'to' new square
    # in a 4 character setup ie d4e5 means a piece moved from d4 to e5
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8, 8))

    # Turn board rep into matrix rep
    from_row = 8 - int(move[1])
    from_column = letter_to_num[move[0]]

    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8, 8))
    to_row = 8 - int(move[3])
    to_column = letter_to_num[move[2]]

    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])


def create_move_list(s):
    return re.sub(r"\d*\. ", "", s).split(" ")[:-1]


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


# test_board_methods()


class ChessDataset(Dataset):
    def __init__(self, games, return_board=False):
        super(ChessDataset, self).__init__()
        self.games = games
        self.return_board = return_board

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        game_i = index

        max_attempts = 5
        for _ in range(max_attempts):
            random_game = self.games["AN"].iloc[game_i]
            moves = create_move_list(random_game)

            if len(moves) > 1:
                break  # good game, we can use this one

            # otherwise, move to the next game (wrap around at end)
            game_i = (game_i + 1) % len(self.games)
        else:
            # If we never found a valid game, raise an error
            raise ValueError("No games with at least 2 moves found in dataset.")

        game_state_i = np.random.randint(
            len(moves) - 1
        )  # Choose a random state from game

        next_move = moves[game_state_i]  # Grab what the next move was
        moves = moves[:game_state_i]  # Grab all prior moves

        board = chess.Board()

        # Build the board up to all moves prior to move CNN has to predict
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


# The CNN!


class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        # Two convolutional layers
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)

        # Two batch normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        # Two SELU activation layers
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
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(chessBot, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList(
            [module(hidden_size) for i in range(hidden_layers)]
        )
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for module in self.module_list:
            x = module(x)

        x = self.output_layer(x)

        return x


# Helper functions to decide moves


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

    probs = probs - probs.max()
    probs = np.exp(probs)
    probs = probs / probs.sum()

    probs = probs**3
    probs = probs / probs.sum()

    return probs


def pred_to_move(move, board):
    #     move = move[0]

    legal_moves = list(board.legal_moves)

    check_mate = check_mate_single(board)
    if check_mate is not None:
        return check_mate

    from_sqs = [str(legal_move)[:2] for legal_move in legal_moves]
    unique_froms = list(set(from_sqs))
    from_vals = []

    for frm in unique_froms:
        file_char, rank_char = frm[0], frm[1]
        row = 8 - int(rank_char)
        col = letter_to_num[file_char]
        val = move[0, row, col].detach().cpu().numpy()
        from_vals.append(val)

    probs = distribution_over_moves(from_vals)

    chosen_from = np.random.choice(unique_froms, size=1, p=probs)[0]

    candidate_moves = []
    candidate_vals = []

    for legal_move in legal_moves:
        uci_str = str(legal_move)
        from_sq = uci_str[:2]

        if from_sq == chosen_from:
            to_sq = uci_str[2:]
            file_char, rank_char = to_sq[0], to_sq[1]

            r = 8 - int(rank_char)
            c = letter_to_num[file_char]

            score = move[1, r, c].detach().cpu().item()  # to-channel score
            candidate_moves.append(legal_move)
            candidate_vals.append(score)

    # fallback in case nothing can be found
    if not candidate_moves:
        best_idx = None
        best_val = -np.inf
        for i, legal_move in enumerate(legal_moves):
            uci_str = str(legal_move)
            to_sq = uci_str[2:]
            file_char, rank_char = to_sq[0], to_sq[1]
            r = 8 - int(rank_char)
            c = letter_to_num[file_char]
            score = move[1, r, c].detach().cpu().item()
            if score > best_val:
                best_val = score
                best_idx = i
        return legal_moves[best_idx]

    # normal case: choose best candidate from chosen_from
    best_idx = int(np.argmax(candidate_vals))
    chosen_move = candidate_moves[best_idx]
    return chosen_move


def custom_collate(batch):
    board_states, correct_moves, boards = zip(*batch)
    board_states = torch.stack([torch.tensor(x) for x in board_states])
    correct_moves = torch.stack([torch.tensor(x) for x in correct_moves])
    # Leave boards as a tuple of chess.Board objects
    return board_states, correct_moves, boards


def train_loop(
    dataloader, model, metric_from, metric_to, optimizer, device, batch_size
):
    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.0
    num_batches = 0

    for batch, (board_state, correct_move) in enumerate(dataloader):
        board_state = board_state.to(device)
        correct_move = correct_move.to(device)

        # Compute prediction and loss
        pred = model(board_state.float())

        pred_from_map = pred[:, 0, :, :]
        pred_to_map = pred[:, 1, :, :]

        target_from_map = correct_move[:, 0, :, :]
        target_to_map = correct_move[:, 1, :, :]

        B = board_state.shape[0]

        pred_from_flat = pred_from_map.view(B, 64)
        pred_to_flat = pred_to_map.view(B, 64)

        target_from_flat = target_from_map.view(B, 64)
        target_to_flat = target_to_map.view(B, 64)

        target_from_idx = torch.argmax(target_from_flat, dim=1)
        target_to_idx = torch.argmax(target_to_flat, dim=1)

        loss_from = metric_from(pred_from_flat, target_from_idx)
        loss_to = metric_to(pred_to_flat, target_to_idx)
        loss = loss_from + loss_to

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if batch % 1000 == 0:
            loss = loss.item()
            current = batch * batch_size + len(board_state)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


def test_loop(dataloader, model, device, batch_size):
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

                matrix_move = move_to_rep(san_move, board_obj)  # get one prediction
                correct_move_np = (
                    correct_move[i].cpu().numpy()
                )  # get the corresponding correct move

                if np.array_equal(matrix_move, correct_move_np):
                    correct += 1

                num_guesses += 1
    #             print("num correct", correct)
    #             print(" ")
    accuracy = (correct / num_guesses) * 100
    print("number of total guesses: ", num_guesses)
    print("number of correct guesses: ", correct)
    print("accuracy: ", accuracy)

    return accuracy


def get_ai_move(board, model, device):
    mat_move = board_to_matrix(board)
    mat_move = np.expand_dims(mat_move, axis=0)  # Add a batch dimension
    mat_move_tensor = torch.tensor(mat_move)  # Convert to PyTorch tensor
    move = model(mat_move_tensor.float().to(device))
    uci = pred_to_move(move[0], board)
    return uci


def train(csv_path: str = "", model_out: str = "", df: pd.DataFrame | None = None):
    # read data
    if df is not None:
        chess_data = df.copy()
    else:
        if not csv_path:
            print("Invalid csv path")
            return

        try:
            chess_data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            return

    if chess_data.empty:
        print(f"No data found in {csv_path}")
        return

    if "AN" not in chess_data.columns or "WhiteElo" not in chess_data.columns:
        print(f"Data cols not found in {csv_path}")
        return

    if not chess_data["AN"].notna().any() or not chess_data["WhiteElo"].notna().any():
        print(
            f"Columns AN/WhiteElo are present but contain no valid data in {csv_path}"
        )
        return

    chess_data = chess_data[["AN", "WhiteElo"]]
    chess_data = chess_data.reset_index(drop=True)

    if not model_out:
        print("Invalid model out path")
        return

    # Filter the data to only games w/ players above 2000 elo, only save moves and of games where there are a lot of moves(?)
    # chess_data = chess_data[chess_data["WhiteElo"] > 2000]
    # chess_data = chess_data[["AN"]]
    # chess_data = chess_data[~chess_data["AN"].str.contains("{")]
    # chess_data = chess_data[chess_data["AN"].str.len() > 20]

    n = len(chess_data)
    train_idx = int(np.ceil(n * 0.8))
    train_dataset = chess_data.iloc[:train_idx]
    test_dataset = chess_data.iloc[train_idx:]

    data_train = ChessDataset(train_dataset, return_board=False)
    data_test = ChessDataset(test_dataset, return_board=True)

    data_train_loader = DataLoader(
        data_train, batch_size=32, shuffle=True, drop_last=True
    )
    data_test_loader = DataLoader(
        data_test,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = chessBot().to(device)

    learning_rate = 1e-2
    batch_size = 32
    epochs = 100

    metric_from = nn.CrossEntropyLoss()
    metric_to = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    best_test_accuracy = None

    with mlflow.start_run() as run:
        # Log static hyperparameters
        mlflow.log_param("csv_path", csv_path)
        mlflow.log_param("model_out", model_out)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden_layers", model.hidden_layers)
        mlflow.log_param("hidden_size", model.hidden_size)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("test_size", len(test_dataset))

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loss = train_loop(
                data_train_loader,
                model,
                metric_from,
                metric_to,
                optimizer,
                device,
                batch_size,
            )
            test_accuracy = test_loop(data_test_loader, model, device, batch_size)

            mlflow.log_metric("train_loss", train_loss, step=t)
            mlflow.log_metric("test_accuracy", test_accuracy, step=t)

            if best_test_accuracy is None or test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                # Save best model so far
                torch.save(model.state_dict(), model_out)

            if optimizer.param_groups[0]["lr"] > 1e-5:
                scheduler.step()

        print("Done!")

        torch.save(model.state_dict(), model_out)
        mlflow.pytorch.log_model(model, artifact_path="model")

        result = {
            "run_id": run.info.run_id,
            "final_test_accuracy": best_test_accuracy,
            "model_path": model_out,
        }

    return result
