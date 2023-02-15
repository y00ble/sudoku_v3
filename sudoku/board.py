import collections
import inspect

import numpy as np
import pandas as pd

from .constraints import DIGITS, Row, Column, Box, GivenDigit
from .exceptions import SudokuContradiction


class Board:

    def __init__(self):
        self.possibles = np.ones(9 * 9 * 9).astype(bool)
        self.contradictions = collections.defaultdict(list)
        self.coveree_index = collections.defaultdict(list)
        self.coverees = []

        for i in DIGITS:
            for constraint_cls in [Row, Column, Box]:
                constraint = constraint_cls(self, i)
                for j in DIGITS:
                    indices = [self._possible_index(
                        *cell, j) for cell in constraint.cells]
                    self._register_coveree(indices)

        self.coveree_counts = np.array(
            [len(coveree) for coveree in self.coverees])

        self.cell_possibles = (np.ones(81) * 0b111111111).astype(int)

    def __setitem__(self, key, value):
        if not isinstance(value, collections.abc.Iterable):
            value = [value]

        row, column = key

        for digit in DIGITS:
            index = self._possible_index(row, column, digit)
            if digit not in value:
                self._remove_possible(index)

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

    def finalise(self, index):
        for contradictory_index in self.contradictions[index]:
            self._remove_possible(contradictory_index)

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

    def _remove_possible(self, index):
        # TODO vectorise this
        if not self.possibles[index]:
            return

        cell_index = self._possible_index_to_cell_index(index)
        digit = self._possible_index_to_digit(index)
        self.possibles[index] = False
        self.cell_possibles[cell_index] = self.cell_possibles[cell_index] & (0b111111111 -
                                                                             (1 << (digit - 1)))
        if self.cell_possibles[cell_index] == 0:
            raise SudokuContradiction("No possibles remaing in {}".format(
                self._describe_cell_index(cell_index)
            ))
        elif self.cell_possibles[cell_index] & (self.cell_possibles[cell_index] - 1) == 0:
            self.finalise(cell_index * 9 +
                          int(np.log2(self.cell_possibles[cell_index])))
        coverees = self.coveree_index[index]
        self.coveree_counts[coverees] -= 1

        # TODO finalise based on coverees

    def _register_coveree(self, cell_indices):
        index = len(self.coverees)
        self.coverees.append(cell_indices)
        for cell in cell_indices:
            self.coveree_index[cell].append(index)
