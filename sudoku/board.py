import collections
import itertools

import numpy as np

from .constraints import DIGITS, Row, Column, Box
from .exceptions import SudokuContradiction

MAX_COVEREE_SIZE = 9


class Board:

    def __init__(self):
        self.possibles = np.ones(9 * 9 * 9 + 1).astype(bool)
        self.possibles[-1] = False
        self.finalised = np.zeros(9 * 9 * 9).astype(bool)
        self.contradictions = collections.defaultdict(list)
        self.coveree_index = collections.defaultdict(list)
        self._coverees = []

        for i in DIGITS:
            for constraint_cls in [Row, Column, Box]:
                constraint = constraint_cls(self, i)
                for j in DIGITS:
                    indices = [self._possible_index(
                        *cell, j) for cell in constraint.cells]
                    self._register_coveree(indices)

        for row in DIGITS:
            for col in DIGITS:
                indices = [self._possible_index(row, col, j) for j in DIGITS]
                self._register_coveree(indices)

                for d1, d2 in itertools.product(DIGITS, repeat=2):
                    if d1 == d2:
                        continue

                    i1 = self._possible_index(row, col, d1)
                    i2 = self._possible_index(row, col, d2)
                    self.contradictions[i1].append(i2)
                    self.contradictions[i2].append(i1)

        self.coverees = []
        for coveree in self._coverees:
            self.coverees.append(
                coveree + [-1] * (MAX_COVEREE_SIZE - len(coveree)))

        self.coverees = np.array(self.coverees)

    def __setitem__(self, key, value):
        if not isinstance(value, collections.abc.Iterable):
            value = [value]

        row, column = key

        for digit in DIGITS:
            index = self._possible_index(row, column, digit)
            if digit not in value:
                self._remove_possibles(index)

    def __str__(self):
        output = ""
        for row in DIGITS:
            for col in DIGITS:
                possibles = [
                    str(i) for i in DIGITS if self.possibles[self._possible_index(row, col, i)]]
                output += "".join(possibles).ljust(len(DIGITS))
                if col in {3, 6}:
                    output += " | "
                elif col != 9:
                    output += " "
            if row in {3, 6}:
                output += "\n" + "-" * (9 * 9 + 6 + 2 * 3) + "\n"
            elif row != 9:
                output += "\n"

        return output

    def finalise(self, indices):
        to_remove = []
        for index in indices:
            if self.finalised[index]:
                continue
            to_remove.extend(self.contradictions[index])

        self.finalised[indices] = True
        if to_remove:
            self._remove_possibles(to_remove)

    @staticmethod
    def _cell_start_index(row, col):
        return (row - 1) * 81 + (col - 1) * 9

    @staticmethod
    def _possible_index(row, col, possible):
        return Board._cell_start_index(row, col) + (possible - 1)

    @staticmethod
    def _possible_index_to_cell_index(index):
        return index // 9

    @staticmethod
    def _possible_index_to_digit(index):
        return index % 9 + 1

    @staticmethod
    def _describe_cell_index(index):
        return f"R{index // 9 + 1}C{index % 9 + 1}"

    def _min_possible_index(self, cell_index):
        cell_start_index = cell_index * 9
        cell_possibles = self.possibles[
            cell_start_index:cell_start_index+9
        ]
        return cell_start_index + min([i for i, possible in enumerate(cell_possibles) if possible])

    def _remove_possibles(self, indices):
        self.possibles[indices] = False

        self.finalise(self.get_singleton_coverees())

    def get_singleton_coverees(self):
        coveree_status = self.possibles[self.coverees]
        coveree_counts = coveree_status.sum(axis=1)
        singleton_coverees_idx = (coveree_counts == 1)
        if np.any(coveree_counts == 0):
            raise SudokuContradiction(
                "At least one coveree cannot be covered!")
        singleton_coverees = (self.coverees[singleton_coverees_idx] *
                              coveree_status[singleton_coverees_idx]).sum(axis=1)
        return singleton_coverees

    def _register_coveree(self, cell_indices):
        if len(cell_indices) > MAX_COVEREE_SIZE:
            raise ValueError(
                "Coveree is too large! Max coveree size is {}".format(MAX_COVEREE_SIZE))
        index = len(self._coverees)
        self._coverees.append(cell_indices)
        for cell in cell_indices:
            self.coveree_index[cell].append(index)
