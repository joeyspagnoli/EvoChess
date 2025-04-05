import torch
import train_model
import chess
import chess.svg
import pygame
import io
import cairosvg

def pixel_to_square(x, y, board_size=512):
    #Convert pixel coordinates to a chess square index.
     #  The board is drawn with rank 8 at the top and rank 1 at the bottom.
    square_size = board_size // 8
    col = x // square_size
    row = y // square_size
    # Pygame's (0,0) is top-left; rank 8 is top so we invert the row:
    rank = 7 - row
    return chess.square(col, rank)

def square_to_pixels(square, board_size=512):
    #Return the top-left pixel coordinates of the given chess square.
    square_size = board_size // 8
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    x = file * square_size
    y = (7 - rank) * square_size  # invert row to match drawing order
    return x, y

def update_board_surface(board, board_size=512):
    #Generate a new board surface from the current board state using chess.svg.
    svg_string = chess.svg.board(board=board, size=board_size)
    png_data = cairosvg.svg2png(bytestring=svg_string)
    png_bytes = io.BytesIO(png_data)
    return pygame.image.load(png_bytes)

def get_player_move_from_clicks(board, board_size=512):
    """
    Waits for two mouse clicks:
      - First click: select a square containing a piece (if it belongs to the player whose turn it is).
      - Second click: select a destination square.
    Returns a move string in UCI notation (e.g. "e2e4") if the move is legal.
    """
    selected_square = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()  # or return some signal that the user wants to quit

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()
                clicked_square = pixel_to_square(x, y, board_size)
                if selected_square is None:
                    # First click: select a square if there's a piece that belongs to the player.
                    piece = board.piece_at(clicked_square)
                    if piece is not None and piece.color == board.turn:
                        selected_square = clicked_square
                        # Optionally, you could provide visual feedback here
                else:
                    # Second click: destination square.
                    destination = clicked_square
                    move_str = chess.square_name(selected_square) + chess.square_name(destination)
                    # Check if the move is legal.
                    legal_moves = [move.uci() for move in board.legal_moves]
                    if move_str in legal_moves:
                        return move_str
                    else:
                        # Optionally, display an error or sound, then reset selection.
                        selected_square = None
                        print("Illegal move selected. Try again.")

pygame.init()
board_size = 512
screen = pygame.display.set_mode((board_size, board_size))
pygame.display.set_caption('Chess Board')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = train_model.chessBot()
model.load_state_dict(torch.load('model.pth'))
model.to(device)



def game_loop():
    board = chess.Board()
    running = True
    board_surface = update_board_surface(board, board_size)
    screen.blit(board_surface, (0, 0))

    while not board.is_game_over() and running:
        # Process Pygame events (e.g., to allow quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- AI Move ---
        ai_move = train_model.get_ai_move(board, model, device)
        board.push_uci(str(ai_move))
        print("AI's move:", ai_move)
        print(board)

        # Update the display after AI's move.
        board_surface = update_board_surface(board, board_size)
        screen.blit(board_surface, (0, 0))
        pygame.display.flip()
        pygame.time.wait(1000)  # Pause for 1 second so you can see the move

        # Check if game is over before player's move.
        if board.is_game_over():
            break

        # --- Player Move ---
        player_move = get_player_move_from_clicks(board, board_size)
        board.push_uci(player_move)
        print("Player's move:", player_move)
        print(board)
        board_surface = update_board_surface(board, board_size)
        screen.blit(board_surface, (0, 0))
        pygame.display.flip()
        pygame.time.wait(1000)

    pygame.quit()

game_loop()