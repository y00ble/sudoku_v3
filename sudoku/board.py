import collections
import curses
import itertools
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
        self.possibles = np.ones(9 * 9 * 9 + 2).astype(bool)
        self.possibles[-1] = False
        self.possibles[-2] = True
        self.finalised = np.zeros(9 * 9 * 9 + 2).astype(bool)
        self.finalised[-1] = True
        self.finalised[-2] = True

        self.unbifurcated_possibles = self.possibles
        self.unbifurcated_finalised = self.finalised
        self.successful_bifurcations = set()
        self.solutions = set()

        self.coveree_index = collections.defaultdict(list)
        self.coverees = []

        self.contradiction_index = collections.defaultdict(list)
        self.contradictions = []

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
                    self.add_coveree(indices)

        for row in DIGITS:
            for col in DIGITS:
                indices = [self._possible_index(row, col, j) for j in DIGITS]
                self.add_coveree(indices)

                for d1, d2 in itertools.product(DIGITS, repeat=2):
                    if d1 == d2:
                        continue

                    i1 = self._possible_index(row, col, d1)
                    i2 = self._possible_index(row, col, d2)
                    self.add_contradiction(i1, i2)

    def __setitem__(self, key, value):
        if not isinstance(value, collections.abc.Iterable):
            value = [value]

        row, column = key

        for digit in DIGITS:
            index = self._possible_index(row, column, digit)
            if digit not in value:
                self._remove_possibles([index])

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
        if with_terminal:
            curses.wrapper(self._solve)
        else:
            self._solve()

        print("{} {} found!".format(len(self.solutions),
              "solution" if len(self.solutions) == 1 else "solutions"))

    def _solve(self, screen=None):
        self.screen = screen
        self._init_colors()
        self.solving = True

        self._finalise_constraints()

        to_finalise = self._get_singleton_coverees()
        if len(to_finalise):
            self.finalise(to_finalise)
        to_remove = None
        while not np.all(self.finalised) or self.bifurcations:
            try:
                if to_remove is not None:
                    self._remove_possibles([to_remove])
                    to_remove = None
                elif not np.all(self.finalised):
                    self._do_bifurcation()
                else:
                    self.successful_bifurcations.update((
                        bifurcation.index for bifurcation in self.bifurcations
                    ))
                    self.solutions.add(tuple(self.possibles))
                    while self.bifurcations:
                        self._pop_bifurcation()

            except SudokuContradiction:
                self._refresh_screen()
                popped = self._pop_bifurcation()
                to_remove = popped.index

        self.solutions.add(tuple(self.possibles))

    def _finalise_constraints(self):
        new_coverees = []
        for coveree in self.coverees:
            new_coverees.append(
                coveree + [-1] * (MAX_COVEREE_SIZE - len(coveree)))
        self.coverees = np.array(new_coverees)

        new_contradictions = []
        for contradiction in self.contradictions:
            new_contradictions.append(
                contradiction + [-2] * (MAX_COVEREE_SIZE - len(contradiction)))
        self.contradictions = np.array(new_contradictions)
        self.contradiction_counts = np.zeros(len(self.contradictions))

    def _pop_bifurcation(self):
        popped_bifurcation = self.bifurcations.pop()
        self.possibles = popped_bifurcation.possibles
        self.finalised = popped_bifurcation.finalised
        return popped_bifurcation

    def finalise(self, indices):
        self.finalised[indices] = True
        to_remove = self._get_forced_contradictions()
        if len(to_remove):
            self._remove_possibles(to_remove)

        self._refresh_screen()

    def _do_bifurcation(self):
        index = self._select_bifurcation_index()
        self.bifurcations.append(Bifurcation(
            index, self.possibles, self.finalised))
        self.possibles = np.copy(self.possibles)
        self.finalised = np.copy(self.finalised)
        self.finalise([index])

    def _select_bifurcation_index(self):
        unfinalised = np.where(self.possibles[:-2] & ~self.finalised[:-2])[0]
        unfinalised_and_unbifurcated = [
            i for i in unfinalised if i not in self.successful_bifurcations]
        return min(unfinalised_and_unbifurcated, key=self._count_contradictions)

    # TODO coveree and contradiction masks if already satisfied
    # TODO are simple criteria actually better or just faster? Count bifurcations.
    def _count_contradictions(self, index):
        return 1 / self.contradiction_counts[self.contradiction_index[index]].sum()
        cell_index = index // 9
        total_to_bifurcate = 0
        possible_count = self.possibles.sum()
        for i in range(9):
            possible_index = 9 * cell_index + i
            contradictions = self.contradictions[self.contradiction_index[possible_index]]

            if self.possibles[possible_index]:
                total_to_bifurcate += possible_count - \
                    self.possibles[list(
                        self.contradictions[possible_index])].sum()

        return total_to_bifurcate, possible_count - self.possibles[list(self.contradictions[possible_index])].sum()

    def _init_colors(self):
        if self.screen is None:
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, 12, -1)  # Primary bifurcation
        curses.init_pair(4, curses.COLOR_GREEN, -1)  # finalised
        curses.init_pair(2, 88, -1)  # Non-primary bifurcation
        curses.init_pair(3, 0, -1)  # Unbifurcated possible

    def _refresh_screen(self):
        if self.screen is None:
            return
        if time.time() - self.last_frame_time < 1 / FRAME_RATE:
            return

        self.last_frame_time = time.time()
        self.screen.erase()

        for i in range(11):
            self.screen.addstr(i, 0, " | ".join([" " * 29] * 3))

        for i in [3, 7]:
            self.screen.addstr(i, 0, "-" * ((9 * 9) + 6 + 2 * 3))

        draw_coords = self._possibles_to_draw_coords(
            self.unbifurcated_possibles)

        bifurcation_indices = [
            bifurcation.index for bifurcation in self.bifurcations]
        for index in range(9 ** 3):
            digit = str(index % 9 + 1)
            coords = tuple(draw_coords[index])
            if self.unbifurcated_possibles[index]:
                self.screen.addstr(
                    *coords, digit, curses.color_pair(3))

            if self.possibles[index]:
                self.screen.addstr(*coords, digit)
                if self.unbifurcated_finalised[index]:
                    self.screen.addstr(*coords, digit, curses.color_pair(4))

            if index in bifurcation_indices:
                first_bifurcation = index == bifurcation_indices[0]
                self.screen.addstr(
                    *coords, digit, curses.color_pair(1 if first_bifurcation else 2))

        self.screen.refresh()

    @staticmethod
    def _possibles_to_draw_coords(possibles):
        reshaped_possibles = possibles[:-2].reshape((81, 9))
        index_offests = np.cumsum(reshaped_possibles, axis=1).reshape(9 ** 3)
        cell_start_xs = ((np.arange(9 ** 3) // 9) % 9) * 10
        cell_start_xs += 2 * (cell_start_xs >= 30) + 2 * (cell_start_xs >= 60)
        x_pos = cell_start_xs + index_offests

        y_pos = np.arange(9 ** 3) // 9 ** 2
        y_pos += (y_pos >= 3).astype(int) + (y_pos >= 6).astype(int)

        return np.stack([y_pos, x_pos]).T

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
            cell_start_index: cell_start_index+9
        ]
        return cell_start_index + min([i for i, possible in enumerate(cell_possibles) if possible])

    def _remove_possibles(self, indices):
        self.possibles[indices] = False
        self.finalised[indices] = True
        if not self.solving:
            return

        to_finalise = self._get_singleton_coverees()
        if len(to_finalise):
            self.finalise(to_finalise)

    def _get_singleton_coverees(self):
        coveree_status = self.possibles[self.coverees]
        coveree_counts = coveree_status.sum(axis=1)
        singleton_coverees_idx = (coveree_counts == 1)
        if np.any(coveree_counts == 0):
            raise SudokuContradiction(
                "At least one coveree cannot be covered!")
        singleton_coverees = (self.coverees[singleton_coverees_idx] *
                              coveree_status[singleton_coverees_idx]).sum(axis=1)

        unfinalised_singletons = singleton_coverees[~self.finalised[singleton_coverees]]
        return unfinalised_singletons

    def _get_forced_contradictions(self):
        contradiction_status = 1 - (self.possibles[self.contradictions] *
                                    self.finalised[self.contradictions])
        self.contradiction_counts = contradiction_status.sum(axis=1)
        forced_contradictions_idx = (
            self.contradiction_counts == 1)
        if np.any(self.contradiction_counts == 0):
            raise SudokuContradiction(
                "At least one contradiction has occurred!")
        forced_contradictions = (self.contradictions[forced_contradictions_idx] *
                                 contradiction_status[forced_contradictions_idx]).sum(axis=1)
        return forced_contradictions

    def add_coveree(self, cell_indices):
        if len(cell_indices) > MAX_COVEREE_SIZE:
            raise ValueError(
                "Coveree is too large! Max coveree size is {}".format(MAX_COVEREE_SIZE))
        index = len(self.coverees)
        self.coverees.append(cell_indices)
        for cell in cell_indices:
            self.coveree_index[cell].append(index)

    def add_contradiction(self, *args):
        args = list(args)
        if len(args) > MAX_COVEREE_SIZE:
            raise ValueError(
                "Contradiction is too large! Max contradiction size is {}".format(MAX_COVEREE_SIZE))
        index = len(self.contradictions)
        self.contradictions.append(args)
        for cell in args:
            self.contradiction_index[cell].append(index)
