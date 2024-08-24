from sudoku import Board, GermanWhisper, KillerCage

board = Board()

KillerCage(board, [(2, 1), (2, 2)], 5)
KillerCage(board, [(1, 8), (2, 8)], 5)
KillerCage(board, [(8, 2), (9, 2)], 5)
KillerCage(board, [(8, 8), (8, 9)], 5)
KillerCage(board, [(4, 4), (4, 5)], 5)

KillerCage(board, [(2, 3), (2, 4)], 10)
KillerCage(board, [(2, 9), (3, 9)], 10)
KillerCage(board, [(3, 3), (4, 3)], 10)
KillerCage(board, [(3, 5), (4, 5)], 10)
KillerCage(board, [(4, 7), (5, 7)], 10)
KillerCage(board, [(5, 3), (6, 3)], 10)
KillerCage(board, [(6, 2), (7, 2)], 10)
KillerCage(board, [(7, 5), (8, 5)], 10)
KillerCage(board, [(9, 8), (9, 9)], 10)

GermanWhisper(board, [(2, 5), (3, 6), (3, 7), (4, 7), (5, 8), (6, 7), (7, 7), (7, 6), (8, 5), (7, 4), (7, 3), (6, 3), (5, 2), (4, 3), (3, 3), (3, 4), (2, 4)])

board.solve(with_terminal=True)
