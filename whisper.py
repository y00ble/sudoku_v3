# https://www.youtube.com/watch?v=nH3vat8z9uM&t=176s
from sudoku import Board, GermanWhisper

try:
    board = Board(do_terminal=True)

    GermanWhisper(
        board,
        [(8, 1), (7, 1), (7, 2), (8, 3), (9, 3), (9, 2)],
    )

    GermanWhisper(board, [(4, 5), (4, 6), (3, 7)])

    GermanWhisper(
        board,
        [(9, 6), (8, 7), (7, 7), (7, 8), (6, 9), (5, 8)],
    )

    GermanWhisper(
        board,
        [
            (6, 3),
            (5, 2),
            (4, 3),
            (3, 4),
            (2, 5),
            (1, 6),
            (1, 7),
            (2, 8),
            (3, 8),
            (4, 7),
            (5, 6),
            (6, 6),
            (7, 6),
            (8, 5),
            (7, 4),
        ],
    )

    board[1, 5] = 1
    board[2, 2] = 5
    board[5, 1] = 6
    board[5, 9] = 9
    board[7, 3] = 3
    board[8, 8] = 3
    board[9, 5] = 3
finally:
    pass
    # print(board)
