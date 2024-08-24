import itertools

from .board import DIGITS, Board
from .constraints import Box, NoRepeatsConstraint


class MadrigalRow(NoRepeatsConstraint):

    def __init__(self, board, index):
        super().__init__(board, [(index + 2 - (i - 1) // 3, i) for i in DIGITS])


class MadrigalColumn(NoRepeatsConstraint):

    def __init__(self, board, index):
        super().__init__(board, [(i, index + (i - 1) // 3) for i in DIGITS])


class MadrigalBoard(Board):

    def _init_grid_constraints(self):
        for constraint_cls in [Box, MadrigalRow, MadrigalColumn]:
            for i in DIGITS:
                if i > 7 and constraint_cls is not Box:
                    break
                constraint = constraint_cls(self, i)
                print(type(constraint), constraint.cells)
                for j in DIGITS:
                    indices = [
                        self.possible_index(*cell, j)
                        for cell in constraint.cells
                    ]
                    self.add_coveree(indices)

        for row in DIGITS:
            for col in DIGITS:
                indices = [self.possible_index(row, col, j) for j in DIGITS]
                self.add_coveree(indices)

                for d1, d2 in itertools.product(DIGITS, repeat=2):
                    if d1 == d2:
                        continue

                    i1 = self.possible_index(row, col, d1)
                    i2 = self.possible_index(row, col, d2)
                    self.add_contradiction(i1, i2)
