#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle, os
from reversi import reversi
import time

# ---------------------------------------------------------------------------
# Load trained weights
# ---------------------------------------------------------------------------
def vector_to_matrix(v):
    m = np.zeros((8, 8))
    groups = [
        [(0,0),(0,7),(7,0),(7,7)],
        [(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)],
        [(1,1),(1,6),(6,1),(6,6)],
        [(0,2),(2,0),(0,5),(5,0),(2,7),(7,2),(5,7),(7,5)],
        [(0,3),(3,0),(0,4),(4,0),(3,7),(7,3),(4,7),(7,4)],
        [(1,2),(2,1),(1,5),(5,1),(2,6),(6,2),(5,6),(6,5)],
        [(1,3),(3,1),(1,4),(4,1),(3,6),(6,3),(4,6),(6,4)],
        [(2,2),(2,5),(5,2),(5,5)],
        [(2,3),(3,2),(2,4),(4,2),(3,5),(5,3),(4,5),(5,4)],
        [(3,3),(3,4),(4,3),(4,4)],
    ]
    for i, group in enumerate(groups):
        for r, c in group:
            m[r, c] = v[i]
    return m

# Fall back to hand-tuned weights if no trained file exists
if os.path.exists('best_weights.npy'):
    _vec = np.load('best_weights.npy')
    if _vec.shape == (10,):
        WEIGHT_MATRIX = vector_to_matrix(_vec)
        print("Loaded trained weights from best_weights.npy")
    else:
        print("Warning: best_weights.npy is wrong shape, using default weights")
        WEIGHT_MATRIX = None
else:
    print("No best_weights.npy found, using default weights")
    WEIGHT_MATRIX = None

# Default hand-tuned fallback (used if no trained weights available)
DEFAULT_WEIGHTS = np.array([
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 10,  -5,   3,   2,   2,   3,  -5,  10],
    [  5,  -5,   2,   1,   1,   2,  -5,   5],
    [  5,  -5,   2,   1,   1,   2,  -5,   5],
    [ 10,  -5,   3,   2,   2,   3,  -5,  10],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [100, -20,  10,   5,   5,  10, -20, 100],
], dtype=float)

if WEIGHT_MATRIX is None:
    WEIGHT_MATRIX = DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------

CORNERS = [(0,0), (0, 7), (7, 0), (7, 7)]
MAX, MIN = float('inf'), float('-inf')

class TimeoutException(Exception):
    pass

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        #Debug info
        print(turn)
        print(board)

        # Minimax - Replace with your algorithm
        """NOTES:
            * 1 = white
            * -1 = black
        """
        start_time = time.time()
        TIME_LIMIT = 5

        def check_time():
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutException

        def find_available_moves(current_board: np.ndarray, turn: int) -> list:
            """Return a list of legal moves"""
            temp_board = reversi()
            temp_board.board = current_board

            moves = []
            for a in range(8):
                for b in range(8):
                    if temp_board.step(a, b, turn, False) > 0:
                        moves.append((a, b))
            return moves

        def use_turn(current_board: np.ndarray, move: tuple, turn: int) -> np.ndarray:
            """Return new board copy after making a move"""
            game = reversi()
            game.board = np.copy(current_board)
            x, y = move
            game.step(x, y, turn, True)
            return game.board

        def board_score(current_board: np.ndarray, player: int) -> int:
            """Give a score to a board using trained positional weights + mobility"""
            # Positional score using trained weight matrix
            positional_score = np.sum(current_board * WEIGHT_MATRIX * player)

            # Mobility score (unchanged from original)
            player_moves = len(find_available_moves(current_board, player))
            opponent_moves = len(find_available_moves(current_board, (-player)))
            mobility_score = player_moves - opponent_moves

            ms_mult = 5  # Mobility multiplier (unchanged)

            return int(positional_score + mobility_score * ms_mult)

        def MM_Algorithm(board: np.ndarray, turn: int, depth: int, curr_player: int, alpha: int, beta: int) -> int:
            """Recursive MM_Algorithm algorithm"""
            legal_moves = find_available_moves(board, turn)
            opponent = -turn

            # Depth limit reached
            if depth == 0:
                score = board_score(board, curr_player)
                return score

            # No moves
            if len(legal_moves) == 0:
                opponent_moves = find_available_moves(board, opponent)

                # Game over (Undo recursion)
                if len(opponent_moves) == 0:
                    score = board_score(board, curr_player)
                    return score

                # Skip curr_player's turn
                else:
                    return MM_Algorithm(board, opponent, depth-1, curr_player, alpha, beta)

            # Maximizing player
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

            # Minimizing player
            else:
                lowest_score = float('inf')

                for move in legal_moves:
                    new_board = use_turn(board, move, turn)
                    score = MM_Algorithm(new_board, opponent, depth-1, curr_player, alpha, beta)
                    lowest_score = min(lowest_score, score)
                    beta = min(beta, lowest_score)
                    if beta <= alpha:
                        break
                return lowest_score

        x = -1
        y = -1
        best_root_score = float('-inf')

        moves = find_available_moves(board, turn)
        p = len(moves)
        if p <= 3:
            depth = 7
        elif p <= 6:
            depth = 6
        elif p <= 7:
            depth = 5
        elif p <= 8:
            depth = 4
        else:
            depth = 3

        print(depth)

        # Default move so we always return something
        x, y = (-1, -1)
        if len(moves) > 0:
            x, y = moves[0]

        try:
            for move in moves:
                check_time()
                new_board = use_turn(board, move, turn)
                score = MM_Algorithm(new_board, -turn, depth-1, turn, MIN, MAX)

                if score > best_root_score:
                    best_root_score = score
                    x, y = move

        except TimeoutException:
            print(f"Search stopped due to time limit : p = {p}")
            game_socket.send(pickle.dumps([x, y]))

        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()