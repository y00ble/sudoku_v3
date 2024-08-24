import itertools
from sudoku import Board, MiniConsecutiveRun, MiniSandwich, MiniSkyscraper, MiniXSum, SudokuContradiction
from tqdm import tqdm
from joblib import Parallel, delayed

LEFT_BRANCH = [(5, 1), (5, 2), (5, 3), (5, 4)]
RIGHT_BRANCH = [(5, 6), (5, 7), (5, 8), (5, 9)]
BOTTOM_BRANCH = [(6, 5), (7, 5), (8, 5), (9, 5)]
TOP_BRANCH = [(4, 5), (3, 5), (2, 5), (1, 5)]


def iter_boards():
    for sandwich_pos_name in {"side", "bottom"}:
        for top_x_pos in itertools.combinations([6, 7, 8, 9], 2):
            if sandwich_pos_name == "bottom" and 6 in top_x_pos:
                continue
            for top_skyscraper_pos in itertools.combinations([1, 2, 3, 4], 2):
                bottom_run_pos = [
                    i for i in [1, 2, 3, 4] if i not in top_skyscraper_pos]
                for side_skyscraper_y in itertools.combinations([7, 8, 9], 1):
                    for side_x_y in itertools.combinations([7, 8, 9], 1):
                        if side_x_y == side_skyscraper_y:
                            continue
                        for side_run_y in itertools.combinations([1, 2, 3, 4], 1):
                            if sandwich_pos_name == "side" and side_run_y in {1, 4}:
                                continue
                            board = Board()
                            if sandwich_pos_name == "side":
                                MiniSandwich(board, (1, 5), [
                                    (1, 6), (1, 7), (1, 8), (1, 9)])
                                MiniSandwich(board, (4, 5), [
                                    (4, 6), (4, 7), (4, 8), (4, 9)])
                            else:
                                MiniSandwich(board, (5, 6), [
                                    (4, 6), (3, 6), (2, 6), (1, 6)])
                                MiniSandwich(board, (5, 9), [
                                    (4, 9), (3, 9), (2, 9), (1, 9)])

                            for x in top_x_pos:
                                MiniXSum(board, (5, x), [
                                    (6, x), (7, x), (8, x), (9, x)])

                            for x in top_skyscraper_pos:
                                MiniSkyscraper(board, (5, x), [
                                    (6, x), (7, x), (8, x), (9, x)])

                            for x in bottom_run_pos:
                                MiniConsecutiveRun(
                                    board, (5, x), [(4, x), (3, x), (2, x), (1, x)])

                            for y in side_run_y:
                                MiniConsecutiveRun(
                                    board, (y, 5), [
                                        (y, 4), (y, 3), (y, 2), (y, 1)]
                                )

                            for y in side_skyscraper_y + (6,):
                                MiniSkyscraper(
                                    board, (y, 5), [
                                        (y, 4), (y, 3), (y, 2), (y, 1)]
                                )

                            for y in side_x_y:
                                MiniXSum(
                                    board, (y, 5), [
                                        (y, 6), (y, 7), (y, 8), (y, 9)]
                                )

                            yield board


# board_count = sum((1 for _ in tqdm(iter_boards())))
board_count = 1296


def solve(board):
    try:
        board.solve()
        return board.solutions
    except SudokuContradiction:
        return None


solutions = Parallel(n_jobs=4)(delayed(solve)(board)
                               for board in tqdm(iter_boards(), total=board_count))

for i, board in enumerate(solutions):
    if board is None:
        continue
    print("BOARD {}".format(i).center(20, "-"))
    for j, solution in enumerate(board.solutions):
        print("SOLUTION {}".format(j).center(20, "-"))
        board.simple_draw(solution)
