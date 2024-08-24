import time
import itertools
from tqdm import tqdm as tq
import numpy as np

from sudoku import Board, AdjacentThermo, SudokuContradiction

REDS = sorted([(i, j) for i in [2, 4, 6, 8] for j in [1, 3, 5, 7, 9]])
ORANGES = sorted([(i, j) for j in [2, 4, 6, 8] for i in [1, 3, 5, 7, 9]])

adjacencies = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 1, 5, 6],
    [1, 2, 6, 7],
    [2, 3, 7, 8],
    [3, 4, 8, 9],
    [5, 6, 10, 11],
    [6, 7, 11, 12],
    [7, 8, 12, 13],
    [8, 9, 13, 14],
    [10, 11, 15, 16],
    [11, 12, 16, 17],
    [12, 13, 17, 18],
    [13, 14, 18, 19],
    [15, 16],
    [16, 17],
    [17, 18],
    [18, 19],
]

all_elbows = []

for orange_idx, red_idxs in enumerate(adjacencies):
    orange = ORANGES[orange_idx]
    elbows = []
    for red_idx in red_idxs:
        red = REDS[red_idx]
        print(red, orange)
        elbows.append(((red[0], orange[1]), red))
        elbows.append(((orange[0], red[1]), red))
    all_elbows.append(elbows)


def iter_matchings():
    queue = [tuple()]
    while queue:
        assigned = queue.pop()
        assigned_set = set(assigned)
        pivot = len(assigned)
        if pivot < 20:
            possibles = set(adjacencies[pivot]) - assigned_set
            for possible in possibles:
                queue.append(assigned + (possible,))
        else:
            yield assigned


for matching in tq(iter_matchings()):
    for skip_indices in itertools.combinations(list(range(20)), 3):
        skip_oranges = [ORANGES[i] for i in skip_indices]
        skip_reds = [REDS[matching[i]] for i in skip_indices]
        orange_xs = {x for x, y in skip_oranges}
        orange_ys = {y for x, y in skip_oranges}
        red_xs = {x for x, y in skip_reds}
        red_ys = {y for x, y in skip_reds}

        if 1 not in {len(orange_xs), len(orange_ys)} or 1 not in {len(red_xs), len(red_ys)}:
            continue

        for bent_thermo in range(20):
            if bent_thermo in skip_indices:
                continue

            for elbow in all_elbows[bent_thermo]:
                board = Board()
                for orange_idx, red_idx in enumerate(matching):
                    orange = ORANGES[orange_idx]
                    red = REDS[red_idx]
                    if orange_idx in skip_indices:
                        board[orange] = [7, 8, 9]
                        board[red] = [1, 2, 3]
                        board.prefered_bifurcations.update(range(
                            board._cell_start_index(*red),
                            board._cell_start_index(*red) + 3))
                    elif orange_idx == bent_thermo:
                        AdjacentThermo(board, (orange,) + elbow)
                    else:
                        AdjacentThermo(board, (orange, red))
                try:
                    board.solve(with_terminal=True)
                    print(matching, skip_indices)
                    exit
                except SudokuContradiction:
                    pass
