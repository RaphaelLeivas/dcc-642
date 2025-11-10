from typing import List, Tuple, Optional, Dict
import time
import math
import random
import csv
import pdb

ROWS, COLS = 6, 7
EMPTY, P1, P2 = 0, 1, 2
MAX_TIME_TOLERANCE = 250 # in ms

nodes_expanded = 0
start = 0
max_time_global = 0

# -----------------------------------------------------------------------------
# Utilidades de tabuleiro (PRONTAS)
# -----------------------------------------------------------------------------
def copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]

def valid_moves(board: List[List[int]]) -> List[int]:
    """Retorna as colunas ainda jogáveis (topo vazio)."""
    return [c for c in range(COLS) if board[0][c] == EMPTY]

def make_move(board: List[List[int]], col: int, player: int) -> Optional[List[List[int]]]:
    """Retorna um novo tabuleiro aplicando a gravidade na coluna col; None se inválido."""
    if col < 0 or col >= COLS or board[0][col] != EMPTY:
        return None
    nb = copy_board(board)
    for r in reversed(range(ROWS)):
        if nb[r][col] == EMPTY:
            nb[r][col] = player
            return nb
    return None

def winner(board: List[List[int]]) -> int:
    """0 se ninguém venceu; 1 ou 2 se há 4 em linha."""
    # Horizontais
    for r in range(ROWS):
        for c in range(COLS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                return x
    # Verticais
    for c in range(COLS):
        for r in range(ROWS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                return x
    # Diag ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                return x
    # Diag ↗
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            x = board[r][c]
            if x != EMPTY and x == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                return x
    return 0

def is_full(board: List[List[int]]) -> bool:
    return all(board[0][c] != EMPTY for c in range(COLS))

def terminal(board: List[List[int]]) -> Tuple[bool, int]:
    """(é_terminal, vencedor) com vencedor=0 para empate/indefinido."""
    w = winner(board)
    if w != 0:
        return True, w
    if is_full(board):
        return True, 0
    return False, 0

def other(player: int) -> int:
    return P1 if player == P2 else P2

# -----------------------------------------------------------------------------
# ÚNICO PONTO A SER IMPLEMENTADO PELOS ALUNOS
# -----------------------------------------------------------------------------
def evaluate_window(window, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    if window.count(piece) == 4:
        score += 10000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 100
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10

    # Block opponent threats
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 120

    return score

def score_position(board, piece):
    ROWS = len(board)
    COLS = len(board[0])
    score = 0

    # Score center column (control of center is strong in Connect 4)
    center_col = COLS // 2
    center_array = [board[r][center_col] for r in range(ROWS)]
    score += center_array.count(piece) * 6

    # Horizontal score
    for r in range(ROWS):
        row = board[r]
        for c in range(COLS - 3):
            window = row[c:c+4]
            score += evaluate_window(window, piece)

    # Vertical score
    for c in range(COLS):
        col = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col[r:r+4]
            score += evaluate_window(window, piece)

    # Positive diagonal score
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Negative diagonal score
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def minimax(board: List[List[int]], player: int, depth: int) -> Tuple[int, int]:
    global nodes_expanded
    nodes_expanded += 1 

    if depth == 0:
        return score_position(board, player), None
    
    is_terminal, winner = terminal(board)
    if is_terminal:
        if winner == P1:
            return -math.inf, None
        if winner == P2:
            return math.inf, None
        else:
            return 0, None
    
    if player == P1: # minimizing player
        minValue = math.inf
        best_move = random.choice(valid_moves(board))

        temp_board = copy_board(board)
        for move in valid_moves(temp_board):
            child_board = make_move(temp_board, move, player)
            value = minimax(child_board, other(player), depth - 1)[0]

            if value < minValue:
                minValue = value
                best_move = move

        return minValue, best_move
    else: # maximizing player
        maxValue = -math.inf
        best_move = random.choice(valid_moves(board))

        temp_board = copy_board(board)
        for move in valid_moves(temp_board):
            child_board = make_move(temp_board, move, player)
            value = minimax(child_board, other(player), depth - 1)[0]

            if value > maxValue:
                maxValue = value
                best_move = move
           
        return maxValue, best_move
    
def minimax_alphabeta(board: List[List[int]], player: int, depth: int, alpha: int, beta: int) -> Tuple[int, int]:
    global nodes_expanded
    nodes_expanded += 1 

    if depth == 0:
        return score_position(board, player), None
    
    is_terminal, winner = terminal(board)
    if is_terminal:
        if winner == P1:
            return -math.inf, None
        if winner == P2:
            return math.inf, None
        else:
            return 0, None
        
    global start
    global max_time_global

    if player == P1: # minimizing player
        minValue = math.inf
        best_move = random.choice(valid_moves(board))

        temp_board = copy_board(board)
        for move in valid_moves(temp_board):
            if max_time_global > 0 and (time.time() - start) * 1000.0 >= (max_time_global - MAX_TIME_TOLERANCE):
                print("RAISED AT", (time.time() - start) * 1000.0)
                raise Exception("custom timeout")

            child_board = make_move(temp_board, move, player)
            value = minimax_alphabeta(child_board, other(player), depth - 1, alpha, beta)[0]

            if value < minValue:
                minValue = value
                best_move = move

            beta = min(beta, value)
            if alpha >= beta:
                break

        return minValue, best_move
    else: # maximizing player
        maxValue = -math.inf
        best_move = random.choice(valid_moves(board))

        temp_board = copy_board(board)
        for move in valid_moves(temp_board):
            if max_time_global > 0 and (time.time() - start) * 1000.0 >= (max_time_global - MAX_TIME_TOLERANCE):
                print("RAISED AT", (time.time() - start) * 1000.0)
                raise Exception("timeout")

            child_board = make_move(temp_board, move, player)
            value = minimax_alphabeta(child_board, other(player), depth - 1, alpha, beta)[0]

            if value > maxValue:
                maxValue = value
                best_move = move

            alpha = max(alpha, value)
            if alpha >= beta:
                break
           
        return maxValue, best_move
    
def iterative_deepening(board, player, max_depth):
    best_move = None
    depth_reached = 0

    for depth in range(2, 1000000):
        global start
        global max_time_global

        # vai aumentando a profundidade ate o tempo permitir
        if max_time_global > 0 and (time.time() - start) * 1000.0 >= (max_time_global - MAX_TIME_TOLERANCE):
            break

        global nodes_expanded
        nodes_expanded = 0

        try:
            move = minimax_alphabeta(board, player, depth, alpha=-math.inf, beta=math.inf)[1]
            best_move = move
            depth_reached = depth
        except Exception:
            pass

    return best_move, depth_reached

def choose_move(board: List[List[int]], player: int, config: Dict) -> Tuple[int, Dict]:
    """
    Decide a coluna (0..6) para jogar agora.

    Parâmetros:
      - board: matriz 6x7 com valores {0,1,2}
      - player: 1 ou 2
      - config: {"max_time_ms": int, "max_depth": int}

    Retorna:
      - col: int (0..6)
    """
    max_time_ms = int(config.get("max_time_ms"))
    max_depth = int(config.get("max_depth"))
    player = int(player)

    print(f"AI choose_move called with max_time_ms={max_time_ms}, max_depth={max_depth}, player={player}")
    
    global start
    start = time.time()

    global max_time_global
    max_time_global = max_time_ms

    # Função auxiliar para checar tempo decorrido   
    def time_exceeded():
        return max_time_ms > 0 and (time.time() - start) * 1000.0 >= max_time_ms
    
    legal = valid_moves(board)

    move = 0
    if not legal:
        # Sem jogadas: devolve 0 por convenção (servidor lida com isso)
        return move
    
    global nodes_expanded
    nodes_expanded = 0

    random.seed(time.time())

    # if player == P1:
    #     move, depth = iterative_deepening(board, player, max_depth)

    #     f = open('ids.csv','a')
    #     f.write(f'{max_time_ms},{time.time() - start},{nodes_expanded},{depth}\n')
    #     f.close()
    # else:
    #     move = minimax_alphabeta(board, player, 5, alpha=-math.inf, beta=math.inf)[1]
    
    # VERSÃO INICIAL: escolhe aleatoriamente entre as jogadas legais
    # move = random.choice(legal)
    # move = minimax(board, player, max_depth)[1]
    move = minimax_alphabeta(board, player, max_depth, alpha=-math.inf, beta=math.inf)[1]
    # move, depth = iterative_deepening(board, player, max_depth)

    # print("player = ", player)

    # print("Expanded nodes = ", nodes_expanded)


    return move
