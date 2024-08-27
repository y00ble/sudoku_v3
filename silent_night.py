from sudoku import Board, GermanWhisper, KillerCage

board = Board()

KillerCage(board, [(1, 1), (1, 2)], 11)

KillerCage(board, [(1, 8), (1, 9), (2, 9)], 18)

KillerCage(board, [(5, 1), (6, 1), (7, 1)], 18)

KillerCage(board, [(9, 1), (9, 2), (9, 3)], 16)

KillerCage(board, [(7, 5), (8, 4), (8, 5), (9, 4), (9, 5)], 18)

KillerCage(board, [(8, 6), (8, 7), (9, 6)], 18)

KillerCage(board, [(9, 7), (9, 8)], 10)

GermanWhisper(board, [(4, 1), (3, 1), (2, 1), (3, 2)])

GermanWhisper(board, [(4, 2), (3, 1)])

GermanWhisper(board, [(2, 3), (1, 4), (2, 5)])

GermanWhisper(board, [(3, 3), (2, 4), (3, 5)])

GermanWhisper(board, [(5, 2), (4, 3), (3, 4), (4, 5), (5, 6)])

GermanWhisper(board, [(6, 2), (5, 3), (4, 4), (5, 5), (6, 6)])

GermanWhisper(board, [(5, 4), (6, 4), (7, 4)])

GermanWhisper(board, [(6, 7), (5, 8), (6, 9)])

GermanWhisper(board, [(7, 7), (6, 8), (7, 9)])

GermanWhisper(board, [(7, 8), (8, 8)])

# board[7, 4] = 7
# board.prefered_bifurcations.update(range(
#     board._cell_start_index(7, 4), board._cell_start_index(7, 4) + 9))
board.solve(with_terminal=True)
