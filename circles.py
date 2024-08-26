from collections import defaultdict

from sudoku import Board
from sudoku.constraints import GivenPossibles, NoX

board = Board()

board[1, 1] = 8
board[1, 2] = 9
board[1, 8] = 1
board[1, 9] = 7
board[2, 1] = 1
GivenPossibles(board, (2, 2), (6, 7))
GivenPossibles(board, (2, 8), (2, 3))
board[2, 9] = 9
board[8, 1] = 2
GivenPossibles(board, (8, 2), (6, 7))
board[8, 8] = 9
board[8, 9] = 8
board[9, 1] = 9
board[9, 2] = 8
GivenPossibles(board, (9, 8), (2, 3))
board[9, 9] = 1

board[1, 3] = 5
board[1, 4] = 6
board[1, 5] = 2
board[1, 6] = 3
board[1, 7] = 4
# board[7, 6] = 6

NoX(board)

board.solve(with_terminal=True)

grouped_by_circle_pattern = defaultdict(list)
for solution_idx, solution in enumerate(board.solutions):
    circle_cells = []
    for row in range(3, 8):
        for col in range(3, 8):
            circle = False
            for digit in range(5, 9):
                index = board.possible_index(row, col, digit)
                if solution[index]:
                    circle = True

            if circle:
                circle_cells.append((row, col))
    grouped_by_circle_pattern[tuple(circle_cells)].append(solution_idx)

print("Solutions grouped by circle pattern:")
for pattern, indices in grouped_by_circle_pattern.items():
    print(indices, ":", pattern)
