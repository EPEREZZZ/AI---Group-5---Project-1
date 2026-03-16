import numpy as np
import socket, pickle, os
from reversi import reversi


# --- WEIGHT MAPPING LOGIC ---
# This maps the 10 values the AI learned back to the 8x8 board
def vector_to_matrix(v):
    m = np.zeros((8, 8))
    groups = [
        [(0, 0), (0, 7), (7, 0), (7, 7)],  # Corners
        [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)],  # C-Squares
        [(1, 1), (1, 6), (6, 1), (6, 6)],  # X-Squares
        [(0, 2), (2, 0), (0, 5), (5, 0), (2, 7), (7, 2), (5, 7), (7, 5)],  # A-Edges
        [(0, 3), (3, 0), (0, 4), (4, 0), (3, 7), (7, 3), (4, 7), (7, 4)],  # B-Edges
        [(1, 2), (2, 1), (1, 5), (5, 1), (2, 6), (6, 2), (5, 6), (6, 5)],
        [(1, 3), (3, 1), (1, 4), (4, 1), (3, 6), (6, 3), (4, 6), (6, 4)],
        [(2, 2), (2, 5), (5, 2), (5, 5)],
        [(2, 3), (3, 2), (2, 4), (4, 2), (3, 5), (5, 3), (4, 5), (5, 4)],
        [(3, 3), (3, 4), (4, 3), (4, 4)]  # Center
    ]
    for i, group in enumerate(groups):
        for r, c in group:
            m[r, c] = v[i]
    return m


# Load learned weights from your trainer
if os.path.exists('best_weights.npy'):
    LEARNED_MATRIX = vector_to_matrix(np.load('best_weights.npy'))
else:
    exit(0)

# Constants
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
MAX, MIN = float('inf'), float('-inf')


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))

    while True:
        data = game_socket.recv(4096)
        if not data: break
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        # --- YOUR ORIGINAL FUNCTIONS ---

        def find_available_moves(current_board: np.ndarray, turn: int) -> list:
            temp_board = reversi()
            temp_board.board = current_board
            moves = []
            for a in range(8):
                for b in range(8):
                    if temp_board.step(a, b, turn, False) > 0:
                        moves.append((a, b))
            return moves

        def use_turn(current_board: np.ndarray, move: tuple, turn: int) -> np.ndarray:
            game = reversi()
            game.board = np.copy(current_board)
            x, y = move
            game.step(x, y, turn, True)
            return game.board

        def board_score(current_board: np.ndarray, player: int) -> int:
            # INTEGRATED: Using learned weight matrix
            positional_score = np.sum(current_board * LEARNED_MATRIX * player)

            # Keep your original mobility score
            player_moves = len(find_available_moves(current_board, player))
            opponent_moves = len(find_available_moves(current_board, (-player)))
            mobility_score = (player_moves - opponent_moves) * 5

            return int(positional_score + mobility_score)

        def MM_Algorithm(board: np.ndarray, turn: int, depth: int, curr_player: int, alpha: float, beta: float) -> int:
            legal_moves = find_available_moves(board, turn)
            opponent = -turn

            if depth == 0:
                return board_score(board, curr_player)

            if len(legal_moves) == 0:
                opponent_moves = find_available_moves(board, opponent)
                if len(opponent_moves) == 0:
                    return board_score(board, curr_player)
                else:
                    return MM_Algorithm(board, opponent, depth - 1, curr_player, alpha, beta)

            if turn == curr_player:
                highest_score = float('-inf')
                for move in legal_moves:
                    new_board = use_turn(board, move, turn)
                    score = MM_Algorithm(new_board, opponent, depth - 1, curr_player, alpha, beta)
                    highest_score = max(highest_score, score)
                    alpha = max(alpha, highest_score)
                    if beta <= alpha:
                        break
                return highest_score
            else:
                lowest_score = float('inf')
                for move in legal_moves:
                    new_board = use_turn(board, move, turn)
                    score = MM_Algorithm(new_board, opponent, depth - 1, curr_player, alpha, beta)
                    lowest_score = min(lowest_score, score)
                    beta = min(beta, lowest_score)
                    if beta <= alpha:
                        break
                return lowest_score

        # Create vars
        x, y = -1, -1
        best_root_score = float('-inf')
        moves = find_available_moves(board, turn)

        # Tree depth
        depth = 3

        for move in moves:
            new_board = use_turn(board, move, turn)
            # Fixed your MIN/MAX order here
            score = MM_Algorithm(new_board, -turn, depth - 1, turn, MIN, MAX)
            if score >= best_root_score:
                best_root_score = score
                x, y = move

        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()