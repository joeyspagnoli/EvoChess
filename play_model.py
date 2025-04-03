import torch
import train_model
import chess

def game_loop():
    board = chess.Board()
    print(board)

    while not board.is_game_over():
        ai_move = train_model.get_ai_move(board, model, device)
        board.push_uci(str(ai_move))
        print("AI's move:", ai_move)
        print(board)

        if board.is_game_over():
            break

        # Player's move|
        player_move = train_model.get_player_move(board)
        board.push_uci(str(player_move))
        print("Player's move:", player_move)
        print(board)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = train_model.chessBot()
model.load_state_dict(torch.load('model.pth'))
model.to(device)

game_loop()