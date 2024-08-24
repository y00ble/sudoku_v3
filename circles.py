from sudoku import Board
from sudoku.constraints import GivenPossibles, NoX

board = Board()

board[1, 1] = 9
GivenPossibles(board, (1, 2), (6, 7, 8))
GivenPossibles(board, (1, 8), (2, 3))
board[1, 9] = 1
GivenPossibles(board, (2, 1), (6, 7))
GivenPossibles(board, (2, 2), (6, 7, 8))
board[2, 8] = 9
GivenPossibles(board, (2, 9), (2, 3))
board[8, 1] = 1
board[8, 2] = 2
board[8, 8] = 8
board[8, 9] = 9
board[9, 1] = 8
board[9, 2] = 9
board[9, 8] = 1
board[9, 9] = 7

NoX(board)

board.solve(with_terminal=True)
