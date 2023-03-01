import copy
from collections import defaultdict, namedtuple
import itertools

from tqdm import tqdm as tq

from sudoku import Board, MiniConsecutiveRun, MiniSandwich, MiniSkyscraper, MiniSumRun, SudokuContradiction


def true_cell_iter(TlRule, TrRule, BrRule, BlRule):
    for l1, l2, l3, l4 in itertools.product(
        itertools.combinations([0, 1, 2, 3], 2),
        itertools.combinations([4, 5, 6, 7], 2),
        itertools.combinations([8, 9, 10, 11], 2),
        itertools.combinations([12, 13, 14, 15], 2),
    ):
        rules_to_indices = defaultdict(list)

        rules_to_indices[TlRule].extend(l1)
        rules_to_indices[TlRule].extend([15 - i for i in l3])

        rules_to_indices[TrRule].extend(
            [i - 8 for i in range(8, 12) if i not in l3])
        rules_to_indices[TrRule].extend(l2)

        rules_to_indices[BrRule].extend(
            [7 - i for i in range(4, 8) if i not in l2])
        rules_to_indices[BrRule].extend(
            [i - 8 for i in range(12, 16) if i not in l4])

        rules_to_indices[BlRule].extend([15 - i for i in l4])
        rules_to_indices[BlRule].extend(
            [7 - i for i in range(4) if i not in l1])

        if MiniSandwich in rules_to_indices:
            indices = rules_to_indices[MiniSandwich]
            if tuple(sorted(indices)) != (0, 3, 4, 7):
                continue

        yield l1 + l2 + l3 + l4


def rule_permutations():
    return [[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 4, 2]]


best_board_index = None
best_score = 9 * 9 * 9


def iter_boards():
    four_rules = [
        MiniSumRun,
        MiniSkyscraper,
        MiniSandwich,
        MiniConsecutiveRun,
    ]
    for permutation in rule_permutations():
        TlRule = four_rules[permutation[0] - 1]
        TrRule = four_rules[permutation[1] - 1]
        BrRule = four_rules[permutation[2] - 1]
        BlRule = four_rules[permutation[3] - 1]
        for true_cells in true_cell_iter(TlRule, TrRule, BrRule, BlRule):
            cell_flags = [i in true_cells for i in range(16)]

            board = Board()

            # Left branch
            for col in tq(range(1, 5), leave=False):
                if cell_flags[col - 1]:
                    TlRule(
                        board,
                        (5, col),
                        [
                            (4, col),
                            (3, col),
                            (2, col),
                            (1, col),
                        ],
                    )
                else:
                    BlRule(
                        board,
                        (5, col),
                        [
                            (6, col),
                            (7, col),
                            (8, col),
                            (9, col),
                        ],
                    )

            # Right branch
            for col in tq(range(6, 10), leave=False):
                if cell_flags[col - 2]:
                    TrRule(
                        board,
                        (5, col),
                        [
                            (4, col),
                            (3, col),
                            (2, col),
                            (1, col),
                        ],
                    )
                else:
                    BrRule(
                        board,
                        (5, col),
                        [
                            (6, col),
                            (7, col),
                            (8, col),
                            (9, col),
                        ],
                    )

            # Top branch
            for row in tq(range(1, 5), leave=False):
                if cell_flags[row + 7]:
                    TlRule(
                        board,
                        (row, 5),
                        [
                            (row, 4),
                            (row, 3),
                            (row, 2),
                            (row, 1),
                        ],
                    )
                else:
                    TrRule(
                        board,
                        (row, 5),
                        [
                            (row, 6),
                            (row, 7),
                            (row, 8),
                            (row, 9),
                        ],
                    )

            # Bottom branch
            for row in tq(range(6, 10), leave=False):
                if cell_flags[row + 6]:
                    BlRule(
                        board,
                        (row, 5),
                        [
                            (row, 4),
                            (row, 3),
                            (row, 2),
                            (row, 1),
                        ],
                    )
                else:
                    BrRule(
                        board,
                        (row, 5),
                        [
                            (row, 6),
                            (row, 7),
                            (row, 8),
                            (row, 9),
                        ],
                    )

            yield board


multi_solution_boards = []
unsolved_boards = []

Score = namedtuple("Score", "interesting possibles outcome")


def map_board_to_scores(board):
    try:
        board.solve(with_terminal=True)
        return Score(True, board.possibles.sum(), "Puzzle solved")
    except SudokuContradiction:
        return Score(False, board.possibles.sum(), "Broke")
    except KeyboardInterrupt:
        raise
    finally:
        print(board)


# boards_copy = copy.deepcopy(boards)

scores = [map_board_to_scores(board) for board in tq(iter_boards())]
# scores = Parallel(n_jobs=-1)(delayed(map_board_to_scores)(board) for board in tq(boards))


def summarise_board(board):
    for constraint in board.constraints:
        print(constraint)

    try:
        board.solve()
    except:
        pass
    print(board)


# print("Best board")

# board, score = min(zip(boards, scores), key=lambda x: x[1][1])
# summarise_board(copy.deepcopy(board))

# print("All board summary")
# for board, score in zip(boards, scores):
#     print(score)
#     print(board)
