from sudoku import MadrigalBoard, AdjacentThermo

board = MadrigalBoard()

AdjacentThermo(board, [(3, 2), (2, 1)])
AdjacentThermo(board, [(2, 4), (2, 3)])
AdjacentThermo(board, [(2, 6), (1, 5)])

AdjacentThermo(board, [(1, 8), (2, 7)])
AdjacentThermo(board, [(3, 8), (4, 8)])
AdjacentThermo(board, [(5, 9), (6, 8)])

AdjacentThermo(board, [(7, 8), (8, 9)])
AdjacentThermo(board, [(8, 6), (8, 7)])
AdjacentThermo(board, [(8, 4), (9, 5)])

AdjacentThermo(board, [(5, 1), (4, 2)])
AdjacentThermo(board, [(7, 2), (6, 2)])
AdjacentThermo(board, [(9, 2), (8, 3)])

AdjacentThermo(board, [(5, 3), (5, 4)])
AdjacentThermo(board, [(4, 5), (3, 5)])
AdjacentThermo(board, [(5, 7), (5, 6)])
AdjacentThermo(board, [(6, 5), (7, 5)])

FREE_CELLS = {(1, 1), (1, 3), (1, 9), (3, 9), (7, 1), (9, 1), (9, 7), (9, 9)}

for i in range(1, 10):
    for j in range(1, 10):
        box = ((i - 1) // 3) * 3 + ((j - 1) // 3) + 1
        if box in {2, 4, 5, 6, 8}:
            continue

        if (i, j) not in FREE_CELLS:
            if box in {1, 9}:
                board[i, j] = [2, 3, 4, 5, 6, 7, 8, 9]
            else:
                board[i, j] = [1, 2, 3, 4, 5, 6, 7, 8]

board[1, 1] = 1
board[1, 9] = 9
board[9, 1] = 9
board[9, 9] = 1
board[1, 3] = 3
board[3, 9] = 3
board[9, 7] = 7
board[7, 1] = 7

board.solve(with_terminal=True)
