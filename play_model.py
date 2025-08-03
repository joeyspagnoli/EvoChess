import torch
import train_model
import chess
import chess.svg
import pygame
import io
import cairosvg


def draw_promotion_overlay(screen, board_size, piece_color):
    """Draw promotion piece choices as an overlay using the same SVG assets as the main board"""
    # Create semi-transparent overlay
    overlay = pygame.Surface((board_size, board_size))
    overlay.set_alpha(200)  # Semi-transparent
    overlay.fill((0, 0, 0))  # Dark overlay
    screen.blit(overlay, (0, 0))

    # Promotion dialog dimensions
    dialog_width = 400
    dialog_height = 150
    dialog_x = (board_size - dialog_width) // 2
    dialog_y = (board_size - dialog_height) // 2

    # Draw dialog background
    dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
    pygame.draw.rect(screen, (240, 240, 240), dialog_rect)
    pygame.draw.rect(screen, (100, 100, 100), dialog_rect, 3)

    # Title text
    font = pygame.font.Font(None, 28)
    title_text = font.render("Choose promotion piece:", True, (0, 0, 0))
    title_rect = title_text.get_rect(center=(board_size // 2, dialog_y + 25))
    screen.blit(title_text, title_rect)

    # Piece data
    pieces = ["q", "r", "b", "n"]
    piece_names = ["Queen", "Rook", "Bishop", "Knight"]

    # Map piece letters to chess piece types
    piece_type_map = {
        "q": chess.QUEEN,
        "r": chess.ROOK,
        "b": chess.BISHOP,
        "n": chess.KNIGHT,
    }

    # Calculate piece positions
    piece_size = 70
    spacing = 90
    start_x = (
        dialog_x
        + (dialog_width - (len(pieces) * spacing - (spacing - piece_size))) // 2
    )
    piece_y = dialog_y + 50

    piece_rects = []
    label_font = pygame.font.Font(None, 16)

    for i, (piece, name) in enumerate(zip(pieces, piece_names)):
        x = start_x + i * spacing

        # Create a full 8x8 board with just this piece at A1
        temp_board = chess.Board()
        temp_board.clear()
        piece_obj = chess.Piece(piece_type_map[piece], piece_color)
        temp_board.set_piece_at(chess.A1, piece_obj)  # Place at A1 (bottom-left)

        # Generate full-size SVG
        board_svg_size = piece_size * 8
        svg_string = chess.svg.board(
            board=temp_board, size=board_svg_size, coordinates=False
        )

        # Convert SVG to pygame surface
        png_data = cairosvg.svg2png(bytestring=svg_string)
        png_bytes = io.BytesIO(png_data)
        full_board_surface = pygame.image.load(png_bytes)

        # Extract the A1 square (bottom-left of the board)
        # A1 is at coordinates (0, 7) in pygame terms (0-indexed, y-flipped)
        square_x = 0 * piece_size  # A file
        square_y = 7 * piece_size  # 1st rank (flipped)
        piece_surface = full_board_surface.subsurface(
            pygame.Rect(square_x, square_y, piece_size, piece_size)
        )

        # Create clickable rectangle
        piece_rect = pygame.Rect(x, piece_y, piece_size, piece_size)
        piece_rects.append((piece_rect, piece))

        # Draw piece background first
        if (i % 2) == 0:
            pygame.draw.rect(screen, (240, 217, 181), piece_rect)  # Light square
        else:
            pygame.draw.rect(screen, (181, 136, 99), piece_rect)  # Dark square

        # Draw the piece
        screen.blit(piece_surface, (x, piece_y))

        # Draw border around piece
        pygame.draw.rect(screen, (100, 100, 100), piece_rect, 2)

        # Draw piece name below
        label_text = label_font.render(name, True, (0, 0, 0))
        label_rect = label_text.get_rect(
            center=(x + piece_size // 2, piece_y + piece_size + 15)
        )
        screen.blit(label_text, label_rect)

    return piece_rects


def get_promotion_choice(current_screen, board_size, piece_color, current_board):
    """Handle promotion piece selection with overlay UI - no window resizing"""
    # Draw the current board with the pawn moved
    board_surface = update_board_surface(current_board, board_size)
    current_screen.blit(board_surface, (0, 0))

    # Draw promotion overlay on top
    piece_rects = draw_promotion_overlay(current_screen, board_size, piece_color)

    pygame.display.flip()

    # Wait for user to click on a piece
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()

                # Check if click is on any promotion piece
                for piece_rect, piece in piece_rects:
                    if piece_rect.collidepoint(x, y):
                        return piece

    # Fallback to queen if something goes wrong
    return "q"


def draw_win_modal(screen, board_size, winner):
    """Draw win modal overlay"""
    # Create semi-transparent overlay
    overlay = pygame.Surface((board_size, board_size))
    overlay.set_alpha(200)  # Semi-transparent
    overlay.fill((0, 0, 0))  # Dark overlay
    screen.blit(overlay, (0, 0))

    # Modal dimensions
    modal_width = 350
    modal_height = 200
    modal_x = (board_size - modal_width) // 2
    modal_y = (board_size - modal_height) // 2

    # Draw modal background
    modal_rect = pygame.Rect(modal_x, modal_y, modal_width, modal_height)
    pygame.draw.rect(screen, (240, 240, 240), modal_rect)
    pygame.draw.rect(screen, (100, 100, 100), modal_rect, 3)

    # Title text
    font_large = pygame.font.Font(None, 36)
    if winner == "white":
        title_text = "EvoChess Wins!"
        title_color = (220, 50, 50)  # Red color
    elif winner == "black":
        title_text = "You Win!"
        title_color = (50, 150, 50)  # Green color
    else:  # draw
        title_text = "Draw!"
        title_color = (100, 100, 200)  # Blue color

    title_surface = font_large.render(title_text, True, title_color)
    title_rect = title_surface.get_rect(center=(board_size // 2, modal_y + 60))
    screen.blit(title_surface, title_rect)

    # Play Again button
    button_width = 150
    button_height = 50
    button_x = (board_size - button_width) // 2
    button_y = modal_y + 120

    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, (100, 150, 255), button_rect)
    pygame.draw.rect(screen, (50, 50, 50), button_rect, 2)

    # Button text
    font_medium = pygame.font.Font(None, 24)
    button_text = font_medium.render("Play Again", True, (255, 255, 255))
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

    return button_rect


def show_win_modal(screen, board_size, winner):
    """Show win modal and wait for play again button click"""
    while True:
        button_rect = draw_win_modal(screen, board_size, winner)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()
                if button_rect.collidepoint(x, y):
                    return True  # Play again requested

        pygame.time.wait(50)  # Small delay to prevent excessive CPU usage


def pixel_to_square(x, y, board_size=512):
    # Convert pixel coordinates to a chess square index.
    #  The board is drawn with rank 8 at the top and rank 1 at the bottom.
    square_size = board_size // 8
    col = x // square_size
    row = y // square_size
    # Pygame's (0,0) is top-left; rank 8 is top so we invert the row:
    rank = 7 - row
    return chess.square(col, rank)


def square_to_pixels(square, board_size=512):
    # Return the top-left pixel coordinates of the given chess square.
    square_size = board_size // 8
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    x = file * square_size
    y = (7 - rank) * square_size  # invert row to match drawing order
    return x, y


def update_board_surface(board, board_size=512, selected_square=None):
    # Generate a new board surface from the current board state using chess.svg.
    # If a square is selected, highlight it with a different color
    if selected_square is not None:
        # Create a dictionary to specify the color for the selected square
        fill = {selected_square: "#90EE90"}  # Light green color
        svg_string = chess.svg.board(board=board, size=board_size, fill=fill)
    else:
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
    global screen
    selected_square = None

    # Initial board draw without highlights
    board_surface = update_board_surface(board, board_size)
    screen.blit(board_surface, (0, 0))
    pygame.display.flip()

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
                        # Draw board with highlighted selected square
                        board_surface = update_board_surface(
                            board, board_size, selected_square
                        )
                        screen.blit(board_surface, (0, 0))
                        pygame.display.flip()
                else:
                    # Check if clicking on the same square (deselect)
                    if clicked_square == selected_square:
                        selected_square = None
                        # Redraw board without highlights
                        board_surface = update_board_surface(board, board_size)
                        screen.blit(board_surface, (0, 0))
                        pygame.display.flip()
                        continue

                    # Check if clicking on another piece of the same color (reselect)
                    piece = board.piece_at(clicked_square)
                    if piece is not None and piece.color == board.turn:
                        # Reselect a different piece
                        selected_square = clicked_square
                        # Redraw board with new highlight
                        board_surface = update_board_surface(
                            board, board_size, selected_square
                        )
                        screen.blit(board_surface, (0, 0))
                        pygame.display.flip()
                        continue

                    # Second click: destination square.
                    destination = clicked_square
                    move_str = chess.square_name(selected_square) + chess.square_name(
                        destination
                    )

                    # Check if this is a pawn promotion move
                    piece = board.piece_at(selected_square)
                    is_promotion = False
                    if piece and piece.piece_type == chess.PAWN:
                        # Check if pawn is moving to the last rank
                        dest_rank = chess.square_rank(destination)
                        if (piece.color == chess.WHITE and dest_rank == 7) or (
                            piece.color == chess.BLACK and dest_rank == 0
                        ):
                            is_promotion = True

                    # Check if the move is legal.
                    legal_moves = [move.uci() for move in board.legal_moves]

                    if is_promotion:
                        # Create a temporary board showing the pawn move for the promotion UI
                        temp_board = board.copy()
                        temp_board.set_piece_at(
                            destination, piece
                        )  # Move pawn to destination
                        temp_board.remove_piece_at(
                            selected_square
                        )  # Remove pawn from origin

                        # Get promotion choice using pygame UI with pawn shown on destination
                        promotion_choice = get_promotion_choice(
                            screen, board_size, piece.color, temp_board
                        )
                        move_str_with_promotion = move_str + promotion_choice

                        if move_str_with_promotion in legal_moves:
                            return move_str_with_promotion
                        else:
                            # If the chosen promotion isn't legal, try queen as fallback
                            move_str_with_queen = move_str + "q"
                            if move_str_with_queen in legal_moves:
                                return move_str_with_queen
                            else:
                                selected_square = None
                                # Redraw board without highlights
                                board_surface = update_board_surface(board, board_size)
                                screen.blit(board_surface, (0, 0))
                                pygame.display.flip()
                                print("Illegal promotion move. Try again.")
                    else:
                        # Regular move (not promotion)
                        if move_str in legal_moves:
                            return move_str
                        else:
                            # Optionally, display an error or sound, then reset selection.
                            selected_square = None
                            # Redraw board without highlights
                            board_surface = update_board_surface(board, board_size)
                            screen.blit(board_surface, (0, 0))
                            pygame.display.flip()
                            print("Illegal move selected. Try again.")


pygame.init()
board_size = 512
screen = pygame.display.set_mode((board_size, board_size))
pygame.display.set_caption("Chess Board")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = train_model.chessBot()
model.load_state_dict(torch.load("model.pth"))
model.to(device)


def game_loop():
    while True:  # Main game loop that allows restarting
        board = chess.Board()
        running = True
        board_surface = update_board_surface(board, board_size)
        screen.blit(board_surface, (0, 0))
        pygame.display.flip()

        while not board.is_game_over() and running:
            # Process Pygame events (e.g., to allow quitting)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

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
            pygame.time.wait(1200)  # 1.2 second delay after user's move

        # Game is over, determine winner and show modal
        if not running:
            break  # User closed the window

        # Determine winner
        result = board.result()
        winner = None
        if result == "1-0":
            winner = "white"  # EvoChess wins
        elif result == "0-1":
            winner = "black"  # Player wins
        else:
            # Draw (1/2-1/2) or other result
            winner = "draw"

        print(f"Game over! Result: {result}")

        # Show win modal and wait for play again
        play_again = show_win_modal(screen, board_size, winner)

        if not play_again:
            break  # Exit the main loop if user doesn't want to play again

    pygame.quit()


if __name__ == "__main__":
    game_loop()
