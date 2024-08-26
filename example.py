from collections import defaultdict

from sudoku.board import Board

board = Board()
givens = [
    [0, 2, 0, 0, 0, 6, 0, 8, 0],
    [0, 9, 6, 0, 1, 5, 0, 0, 2],
    [5, 0, 7, 0, 3, 0, 4, 0, 0],
    [0, 3, 0, 5, 0, 0, 0, 0, 4],
    [2, 0, 1, 4, 0, 8, 9, 0, 3],
    [8, 0, 0, 0, 0, 9, 0, 1, 0],
    [0, 0, 5, 0, 9, 0, 2, 0, 8],
    [9, 0, 0, 1, 8, 0, 3, 5, 0],
    [0, 6, 0, 2, 0, 0, 0, 9, 0],
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
