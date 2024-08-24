from __future__ import annotations


import collections
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from .board import Board

DIGITS = list(range(1, 10))


class Constraint(ABC):

    def __init__(self, board: Board, cells: list[tuple]):
        self.board = board
        self.cells = np.array(cells)
        self.add_contradictions()
        self.restrict_possibles()

    @abstractmethod
    def add_contradictions(self):
        """
        Add entries to a board's contradictions indicating
        which values break this constraint.
        """
        ...

    @abstractmethod
    def restrict_possibles(self):
        """
        Mark some of the board's numbers as impossible.
        """
        ...


class NoRepeatsConstraint(Constraint):

    def add_contradictions(self):
        for c1, c2 in itertools.combinations(self.cells, 2):
            for value in DIGITS:
                i1 = self.board.possible_index(*c1, value)
                i2 = self.board.possible_index(*c2, value)
                self.board.add_contradiction(i1, i2)

    def restrict_possibles(self):
        pass


class Row(NoRepeatsConstraint):

    def __init__(self, board, row):
        super().__init__(board, [(row, i) for i in DIGITS])


class Column(NoRepeatsConstraint):

    def __init__(self, board, column):
        super().__init__(board, [(i, column) for i in DIGITS])


class Box(NoRepeatsConstraint):

    def __init__(self, board, box):
        rows = [((box - 1) // 3) * 3 + i + 1 for i in range(3)]
        cols = [((box - 1) % 3) * 3 + i + 1 for i in range(3)]
        super().__init__(board, [(i, j) for i in rows for j in cols])


class GivenPossibles(Constraint):

    def __init__(self, board, cell, possibles):
        self.possibles = possibles
        super().__init__(board, [cell])

    def restrict_possibles(self):
        indices = [
            self.board.possible_index(*self.cells[0], i)
            for i in DIGITS
            if i not in self.possibles
        ]
        self.board._remove_possibles(indices)

    def add_contradictions(self):
        pass


class GermanWhisper(Constraint):

    def restrict_possibles(self):
        indices = [
            self.board.possible_index(row, col, 5) for row, col in self.cells
        ]
        self.board._remove_possibles(indices)

    def add_contradictions(self):
        for (r1, c1), (r2, c2) in zip(self.cells[:-1], self.cells[1:]):
            for d1, d2 in itertools.product(DIGITS, repeat=2):
                if abs(d1 - d2) < 5:
                    i1 = self.board.possible_index(r1, c1, d1)
                    i2 = self.board.possible_index(r2, c2, d2)
                    self.board.add_contradiction(i1, i2)


def powerset(iterable):
    s = list(iterable)
    output_iterable = itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )
    return [tuple(sorted(subset)) for subset in output_iterable]


class IsValidConstraint(Constraint):

    contradiction_cache = {}

    @abstractmethod
    def is_valid(self, values): ...

    @abstractmethod
    def initial_possibles(self): ...

    @property
    def _instance_key(self):
        return type(self), tuple([tuple(cell) for cell in self.cells])

    def add_contradictions(self):
        if self._instance_key in IsValidConstraint.contradiction_cache:
            for contradiction in IsValidConstraint.contradiction_cache[
                self._instance_key
            ]:
                self.board.add_contradiction(*contradiction)
            return

        validity_counts = collections.defaultdict(int)
        for assignment in itertools.product(*self.initial_possibles()):
            if self.is_valid(assignment):
                indices = [
                    self.board.possible_index(*cell, value)
                    for cell, value in zip(self.cells, assignment)
                ]
                for valid_subset in powerset(indices):
                    validity_counts[tuple(sorted(valid_subset))] += 1

        contradictions = set()
        for contradiction_size in range(len(self.cells) + 1):
            for cell_indices in itertools.combinations(
                list(range(len(self.cells))), contradiction_size
            ):
                cells = [self.cells[i] for i in cell_indices]
                for assignment in itertools.product(
                    *[self.initial_possibles()[i] for i in cell_indices]
                ):
                    indices = tuple(
                        sorted(
                            [
                                self.board.possible_index(*cell, value)
                                for cell, value in zip(cells, assignment)
                            ]
                        )
                    )
                    if not validity_counts[indices]:
                        if all(
                            [
                                validity_counts[subset]
                                for subset in powerset(indices)
                                if subset != indices
                            ]
                        ):
                            contradictions.add(indices)

        IsValidConstraint.contradiction_cache[self._instance_key] = (
            contradictions
        )
        for contradiction in contradictions:
            self.board.add_contradiction(*contradiction)

    def restrict_possibles(self):
        to_remove = []
        for cell, possibles in zip(self.cells, self.initial_possibles()):
            for digit in DIGITS:
                index = self.board.possible_index(*cell, digit)
                if digit not in possibles:
                    to_remove.append(index)

        self.board._remove_possibles(to_remove)


class KillerCage(IsValidConstraint):

    def __init__(self, board, cells, total):
        self.total = total
        super().__init__(board, cells)

    def initial_possibles(self):
        return [DIGITS] * len(self.cells)

    def is_valid(self, values):
        return sum(values) == self.total and len(set(values)) == len(values)


class MiniSandwich(IsValidConstraint):

    def __init__(self, board, sum_cell, summand_cells):
        if len(summand_cells) != 4:
            raise ValueError("Must be exactly 4 summand cells")
        super().__init__(board, [sum_cell] + list(summand_cells))

    def initial_possibles(self):
        return [
            [5, 6, 7, 8],
            [1, 9],
            [2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6],
            [1, 9],
        ]

    def is_valid(self, values):
        return (
            len(set(values)) == len(values)
            and values[2] + values[3] == values[0]
        )


class MiniSkyscraper(IsValidConstraint):

    def __init__(self, board, count_cell, subject_cells):
        if len(subject_cells) != 4:
            raise ValueError("Must be exactly 4 subject cells")
        super().__init__(board, [count_cell] + subject_cells)
        self.board.prefered_bifurcations.add(
            self.board.possible_index(*count_cell, 4)
        )

    def initial_possibles(self):
        return [[1, 2, 3, 4], DIGITS, DIGITS, DIGITS, DIGITS]

    def is_valid(self, values):
        current_height = 0
        visible_count = 0
        for i in values[1:]:
            if i > current_height:
                current_height = i
                visible_count += 1

        return visible_count == values[0]


class MiniConsecutiveRun(IsValidConstraint):

    def __init__(self, board, count_cell, subject_cells):
        if len(subject_cells) != 4:
            raise ValueError("Must be exactly 4 subject cells")
        super().__init__(board, [count_cell] + subject_cells)
        self.board.prefered_bifurcations.add(
            self.board.possible_index(*count_cell, 4)
        )

    def initial_possibles(self):
        return [[1, 2, 3, 4], DIGITS, DIGITS, DIGITS, DIGITS]

    def is_valid(self, values):
        current_run = 0
        previous_value = None
        best_run = 1
        for i in values[1:]:
            if previous_value is None:
                current_run = 1
                previous_value = i
            elif abs(previous_value - i) == 1:
                current_run += 1
                if current_run > best_run:
                    best_run = current_run
                previous_value = i
            else:
                current_run = 0
                previous_value = None

        return best_run == values[0] and len(set(values)) == len(values)


class MiniSumRun(IsValidConstraint):

    def __init__(self, board, count_cell, subject_cells):
        if len(subject_cells) != 4:
            raise ValueError("Must be exactly 4 subject cells")
        super().__init__(board, [count_cell] + subject_cells)

    def initial_possibles(self):
        # return [[3, 4, 5, 6, 7, 8, 9]] + [DIGITS] * 4
        return [[5, 6, 7, 8, 9]] + [DIGITS] * 4

    def is_valid(self, values):
        if len(set(values)) != len(values):
            return False

        for start, end in itertools.product(range(1, 5), repeat=2):
            if sum(values[start:end]) == values[0]:
                return True

        return False


class MiniXSum(IsValidConstraint):

    def __init__(self, board, count_cell, subject_cells):
        if len(subject_cells) != 4:
            raise ValueError("Must be exactly 4 subject cells")
        super().__init__(board, [count_cell] + subject_cells)

    def initial_possibles(self):
        # return [[3, 4, 5, 6, 7, 8, 9]] + [DIGITS] * 4
        return [[5, 6, 7, 8, 9], [2, 3]] + [DIGITS] * 3

    def is_valid(self, values):
        if len(set(values)) != len(values):
            return False

        return sum(values[1 : 1 + values[1]]) == values[0]


class AdjacentThermo(IsValidConstraint):
    def __init__(self, board, cells):
        super().__init__(board, cells)

    def initial_possibles(self):
        return [
            list(range(i + 1, 11 - len(self.cells) + i))
            for i in range(len(self.cells))
        ]

    def is_valid(self, values):
        for i in range(len(self.cells) - 1):
            if values[i + 1] != values[i] + 1:
                return False
        return True


class NoX(Constraint):
    """No adjacent cells sum to 10."""

    def __init__(self, board):
        cells = [(i, j) for i in range(1, 10) for j in range(1, 10)]
        super().__init__(board, cells)

    def restrict_possibles(self):
        pass

    def add_contradictions(self):
        for low_summand in range(1, 5):
            for col in DIGITS:
                for row in DIGITS:
                    # Right side of cell
                    if col != max(DIGITS):
                        i1 = self.board.possible_index(row, col, low_summand)
                        i2 = self.board.possible_index(
                            row, col + 1, 10 - low_summand
                        )
                        self.board.add_contradiction(i1, i2)

                        i1 = self.board.possible_index(
                            row, col, 10 - low_summand
                        )
                        i2 = self.board.possible_index(
                            row, col + 1, low_summand
                        )
                        self.board.add_contradiction(i1, i2)

                    if row < max(DIGITS):
                        i1 = self.board.possible_index(row, col, low_summand)
                        i2 = self.board.possible_index(
                            row + 1, col, 10 - low_summand
                        )
                        self.board.add_contradiction(i1, i2)

                        i1 = self.board.possible_index(
                            row, col, 10 - low_summand
                        )
                        i2 = self.board.possible_index(
                            row + 1, col, low_summand
                        )
                        self.board.add_contradiction(i1, i2)
