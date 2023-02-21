from sudoku.board import Board

board = Board()
givens = [
    [6, 0, 0, 0, 9, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0],
    [4, 0, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 9],
    [0, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 0, 0],
]

for i, digits in enumerate(givens):
    for j, digit in enumerate(digits):
        if digit != 0:
            row = i + 1
            column = j + 1
            board[row, column] = digit

try:
    board.solve(with_terminal=True)
finally:
    print(board)
# try:
#     board.solve()
# finally:
#     print(board)
