import collections
import curses
import itertools
import random
import time

import numpy as np

from .constraints import DIGITS, Row, Column, Box
from .exceptions import SudokuContradiction

MAX_COVEREE_SIZE = 9

WINDOW_WIDTH = 9 * 9 + 6 + 2 * 3 + 1
WINDOW_HEIGHT = 9 + 2
FRAME_RATE = 1

Bifurcation = collections.namedtuple(
    "Bifurcation", "index possibles finalised")


class Board:

    def __init__(self):
        self.possibles = np.ones(9 * 9 * 9 + 1).astype(bool)
        self.possibles[-1] = False
        self.finalised = np.zeros(9 * 9 * 9).astype(bool)
        self.unbifurcated_possibles = self.possibles
        self.unbifurcated_finalised = self.finalised
        self.contradictions = collections.defaultdict(list)
        self.coveree_index = collections.defaultdict(list)
        self._coverees = []
        self.solving = False
        self.screen = None
        self.last_frame_time = 0
        self.bifurcations = []

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

    def solve(self, with_terminal=False):
        # TODO detect multiple solutions
        # TODO error handling for impossible puzzles
        # TODO contradictions hypergraph
        if with_terminal:
            curses.wrapper(self._solve)
        else:
            self._solve()

        print(self)

    def _solve(self, screen=None):
        self.screen = screen
        self._init_colors()
        self.solving = True

        self.finalise(self.get_singleton_coverees())
        to_remove = None
        while not np.all(self.finalised):
            try:
                if to_remove is not None:
                    self._remove_possibles([to_remove])
                    to_remove = None
                else:
                    self._do_bifurcation()
            except SudokuContradiction:
                self._refresh_screen()
                failed_bifurcation = self.bifurcations.pop()
                self._rewind_bifurcation(failed_bifurcation)
                to_remove = failed_bifurcation.index

    def finalise(self, indices):
        to_remove = []
        for index in indices:
            if self.finalised[index]:
                continue
            to_remove.extend(self.contradictions[index])

        self.finalised[indices] = True
        if to_remove:
            self._remove_possibles(to_remove)

        self._refresh_screen()

    def _do_bifurcation(self):
        index = self._select_bifurcation_index()
        self.bifurcations.append(Bifurcation(
            index, np.copy(self.possibles), np.copy(self.finalised)))
        self.finalise([index])

    def _select_bifurcation_index(self):
        # TODO with the following line on the german whispers example, why
        # does it bifurcate on 1s in box 2 when there's a given 1 there?
        # return min(np.where(self.possibles[:-1] & ~self.finalised)[0])
        return max(np.where(self.possibles[:-1] & ~self.finalised)[0], key=lambda x: len(self.contradictions[x]))

    def _rewind_bifurcation(self, bifurcation):
        self.possibles = bifurcation.possibles
        self.finalised = bifurcation.finalised

    def _init_colors(self):
        if self.screen is None:
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)

    def _refresh_screen(self):
        if self.screen is None:
            return
        if time.time() - self.last_frame_time < 1 / FRAME_RATE:
            return
        self.last_frame_time = time.time()

        self.screen.erase()
        for i, line in enumerate(str(self).splitlines()):
            self.screen.addstr(i, 0, line)

        for bifurcation in self.bifurcations:
            # TODO pull out into another function
            index = bifurcation.index
            row = index // 81
            line_number = row + row // 3
            column = index // 9 % 9
            cursor_column = 10 * column + 2 * (column // 3)
            digit = index % 9
            # TODO more colors (probably highlight non-bifurcated stuff)
            self.screen.addstr(line_number, cursor_column,
                               str(digit), curses.color_pair(1))

        self.screen.refresh()

    @ staticmethod
    def _cell_start_index(row, col):
        return (row - 1) * 81 + (col - 1) * 9

    @ staticmethod
    def _possible_index(row, col, possible):
        return Board._cell_start_index(row, col) + (possible - 1)

    @ staticmethod
    def _possible_index_to_cell_index(index):
        return index // 9

    @ staticmethod
    def _possible_index_to_digit(index):
        return index % 9 + 1

    @ staticmethod
    def _describe_cell_index(index):
        return f"R{index // 9 + 1}C{index % 9 + 1}"

    def _min_possible_index(self, cell_index):
        cell_start_index = cell_index * 9
        cell_possibles = self.possibles[
            cell_start_index: cell_start_index+9
        ]
        return cell_start_index + min([i for i, possible in enumerate(cell_possibles) if possible])

    def _remove_possibles(self, indices):
        self.possibles[indices] = False
        self.finalised[indices] = True
        if not self.solving:
            return
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
