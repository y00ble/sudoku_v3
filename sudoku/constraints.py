import itertools
from abc import ABC, abstractmethod

import numpy as np


DIGITS = list(range(1, 10))


class Constraint(ABC):

    def __init__(self, board, cells):
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
                i1 = self.board._possible_index(*c1, value)
                i2 = self.board._possible_index(*c2, value)
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


class GermanWhisper(Constraint):

    def restrict_possibles(self):
        indices = [self.board._possible_index(
            row, col, 5) for row, col in self.cells]
        self.board._remove_possibles(indices)

    def add_contradictions(self):
        for (r1, c1), (r2, c2) in zip(self.cells[:-1], self.cells[1:]):
            for d1, d2 in itertools.product(DIGITS, repeat=2):
                if abs(d1 - d2) < 5:
                    i1 = self.board._possible_index(r1, c1, d1)
                    i2 = self.board._possible_index(r2, c2, d2)
                    self.board.add_contradiction(i1, i2)
