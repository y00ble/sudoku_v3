from __future__ import annotations

import collections
import curses
import itertools
import logging
import time

import numpy as np
from click import option

from .constraints import DIGITS, Box, Column, Row
from .exceptions import (
    DuplicateBifurcation,
    NoBifurcations,
    SudokuContradiction,
)

MAX_COVEREE_SIZE = 9

WINDOW_WIDTH = 9 * 9 + 6 + 2 * 3 + 1
WINDOW_HEIGHT = 9 + 2
FRAME_RATE = 1

Bifurcation = collections.namedtuple(
    "Bifurcation", "index possibles finalised pivot_coveree"
)

logging.basicConfig(
    filename="solve.log",
    filemode="w",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.ERROR,
)
_file_logger = logging.getLogger(__name__)

COVEREE_FILL = -1
CONTRADICTIONS_FILL = -2


class Board:

    def __init__(self):
        self.possibles = np.ones(9 * 9 * 9 + 2).astype(bool)
        self.possibles[COVEREE_FILL] = False
        self.possibles[CONTRADICTIONS_FILL] = True
        self.finalised = np.zeros(9 * 9 * 9 + 2).astype(bool)
        self.finalised[COVEREE_FILL] = True
        self.finalised[CONTRADICTIONS_FILL] = True

        # Indices that have been seen in at least one valid solution -
        # deprioritise these for bifurcation, but can't completely discount them
        # if trying to enumerate all solutions.
        self.in_valid_solutions = np.zeros(9 * 9 * 9 + 2)

        self.unbifurcated_possibles = self.possibles
        self.unbifurcated_finalised = self.finalised
        self.completed_bifurcations = set()
        self.solutions = set()

        self.prefered_bifurcations = set()
        self.coverees = []
        self.pivot_coveree = None

        self.contradictions = []

        self.solving = False
        self.screen = None
        self.last_frame_time = 0
        self.bifurcations = []

        self._init_grid_constraints()

    def _init_grid_constraints(self):
        for i in DIGITS:
            for constraint_cls in [Row, Column, Box]:
                constraint = constraint_cls(self, i)
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

    def __setitem__(self, key, value):
        if not isinstance(value, collections.abc.Iterable):
            value = [value]

        row, column = key

        for digit in DIGITS:
            index = self.possible_index(row, column, digit)
            if digit not in value:
                self._remove_possibles([index])

    def __str__(self):
        return self.simple_draw(self.possibles)

    def simple_draw(self, state):
        output = ""
        for row in DIGITS:
            for col in DIGITS:
                possibles = [
                    str(i)
                    for i in DIGITS
                    if state[self.possible_index(row, col, i)]
                ]
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
        try:
            if with_terminal:
                curses.wrapper(self._solve)
            else:
                self._solve()
        finally:
            print(
                "{} {} found!".format(
                    len(self.solutions),
                    "solution" if len(self.solutions) == 1 else "solutions",
                )
            )
            for i, solution in enumerate(self.solutions):
                print(f"SOLUTION {i}".center(30, "-"))
                print(self.simple_draw(solution))

    def possible_bifurcations(self, prefered_only=False):
        current_bifurcations = self.sorted_bifurcation_indices
        finalised_digits = {
            i for i in range(9**3) if self.finalised[i] and self.possibles[i]
        }
        for i in range(9**3):
            if not self.possibles[i]:
                continue
            if self.finalised[i]:
                continue
            # TODO this should be a subset check
            # If any existing bifurcations are a subset of this one, we don't
            # need to check it.
            # Moreover, we only need to check existing bifurcations involving
            # `i`, as others will have been checked at higher layers of
            # recursion.
            # Also, we should include all finalised digits into our proposed
            # bifurcation, as they may have been bifurcated on before.
            proposed_bifurcation = (
                {i}.union(current_bifurcations).union(finalised_digits)
            )
            relevant_completed_bifurcations = [
                set(b) for b in self.completed_bifurcations if i in b
            ]
            if any(
                b <= proposed_bifurcation
                for b in relevant_completed_bifurcations
            ):
                continue
            if prefered_only and i not in self.prefered_bifurcations:
                continue
            yield i

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
                    bifurcation_index = self._select_bifurcation_index()
                    if bifurcation_index is None:
                        # All bifurcations already tried, and led to solutions
                        # Erroneous bifurcations should've been removed by now.
                        if not self.bifurcations:
                            return

                        self._pop_bifurcation()
                        continue

                    self._do_bifurcation(bifurcation_index)
                else:
                    self._add_solution()
                    self._pop_bifurcation()
            except SudokuContradiction:
                if not self.bifurcations:
                    raise
                self._refresh_screen()
                popped = self._pop_bifurcation()
                to_remove = popped.index
            except DuplicateBifurcation:
                self._refresh_screen()
                if self.bifurcations:
                    self._pop_bifurcation()

        self._add_solution()

    def _add_solution(self):
        self.in_valid_solutions = np.logical_or(
            self.in_valid_solutions, self.possibles
        )
        _file_logger.info("%sSolution found!", " " * len(self.bifurcations))
        self.solutions.add(tuple(self.possibles))

    def _finalise_constraints(self):
        new_coverees = []
        for coveree in self.coverees:
            new_coverees.append(
                coveree + [COVEREE_FILL] * (MAX_COVEREE_SIZE - len(coveree))
            )
        self.coverees = np.array(new_coverees)

        new_contradictions = []
        for contradiction in self.contradictions:
            new_contradictions.append(
                contradiction
                + [CONTRADICTIONS_FILL]
                * (MAX_COVEREE_SIZE - len(contradiction))
            )
        self.contradictions = np.array(new_contradictions)

    def _pop_bifurcation(self):
        self.completed_bifurcations.add(self.sorted_bifurcation_indices)
        popped_bifurcation = self.bifurcations.pop()
        self.possibles = popped_bifurcation.possibles
        self.finalised = popped_bifurcation.finalised
        self.pivot_coveree = popped_bifurcation.pivot_coveree
        return popped_bifurcation

    def finalise(self, indices):
        self.finalised[indices] = True

        finalised_digits = self.finalised_digits
        if any(set(b) <= finalised_digits for b in self.completed_bifurcations):
            raise DuplicateBifurcation("Duplicate bifurcation!")

        to_remove = self._get_forced_contradictions()
        if len(to_remove):
            self._remove_possibles(to_remove)

        self._refresh_screen()

    @property
    def finalised_digits(self):
        return {
            i for i in range(9**3) if self.finalised[i] and self.possibles[i]
        }

    @property
    def sorted_bifurcation_indices(self) -> tuple[int]:
        return tuple(
            sorted(bifurcation.index for bifurcation in self.bifurcations)
        )

    def _do_bifurcation(self, index):
        # Debug
        cell_idx = self.possible_index_to_cell_index(index)
        row = cell_idx // 9 + 1
        col = cell_idx % 9 + 1
        digit = self.possible_index_to_digit(index)
        if (row, col) in {(5, 2), (5, 9), (6, 1), (6, 9), (7, 1), (7, 2)}:
            if digit in {3, 5}:
                self._refresh_screen()
                a = 0
        # End debug
        self.bifurcations.append(
            Bifurcation(
                index, self.possibles, self.finalised, self.pivot_coveree
            )
        )
        self.possibles = np.copy(self.possibles)
        self.finalised = np.copy(self.finalised)
        self.pivot_coveree = None
        self.finalise([index])

    def _select_bifurcation_index(self):
        if self.pivot_coveree is not None:
            try:
                output = next(self.pivot_coveree)
            except StopIteration:
                return None

            if not self.finalised[output]:
                return output

        options = list(self.possible_bifurcations(True))
        if not options:
            options = list(self.possible_bifurcations())
            if not options:
                return None

        live_coverees = self._live_coverees()
        live_coverees[np.isin(live_coverees, options, invert=True)] = (
            COVEREE_FILL
        )
        coveree_lengths = (live_coverees != COVEREE_FILL).sum(axis=1)
        min_coveree_length = coveree_lengths.min()
        min_length_coverees = live_coverees[
            coveree_lengths == min_coveree_length
        ]
        min_coveree_indices = np.unique(min_length_coverees)

        live_contradictions = self._live_contradictions()
        contradiction_counts = dict(
            zip(*np.unique(live_contradictions, return_counts=True))
        )
        bifurcation_scores = {
            k: contradiction_counts.get(k, 0) for k in min_coveree_indices
        }

        coveree_scores = np.vectorize(bifurcation_scores.__getitem__)(
            min_length_coverees
        ).sum(axis=1)
        selected_coveree = min_length_coverees[np.argmax(coveree_scores)]
        self.pivot_coveree = iter(
            [i for i in selected_coveree if i != COVEREE_FILL]
        )
        return next(self.pivot_coveree)

    def _init_colors(self):
        if self.screen is None:
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, 12, -1)  # Primary bifurcation
        curses.init_pair(4, 28, -1)  # in a valid solution
        curses.init_pair(
            5, 72, -1
        )  # in a valid solution, but not on bifurcation
        curses.init_pair(2, 88, -1)  # Non-primary bifurcation
        curses.init_pair(3, 0, -1)  # Unbifurcated possible

    def _refresh_screen(self):
        if self.screen is None:
            return
        if time.time() - self.last_frame_time < 1 / FRAME_RATE:
            return

        self.last_frame_time = time.time()
        self.screen.erase()

        self.screen.addstr(
            0,
            0,
            "==={} SOLUTIONS, BIFURCATION STATUS {}".format(
                len(self.solutions),
                ",".join([str(b.possibles.sum()) for b in self.bifurcations]),
            ).ljust((9 * 9) + 6 + 2 * 3, "="),
        )

        for i in range(11):
            self.screen.addstr(i + 1, 0, " | ".join([" " * 29] * 3))

        for i in [3, 7]:
            self.screen.addstr(i + 1, 0, "-" * ((9 * 9) + 6 + 2 * 3))

        draw_coords = self._possibles_to_draw_coords(
            self.unbifurcated_possibles
        )

        bifurcation_indices = [
            bifurcation.index for bifurcation in self.bifurcations
        ]
        for index in range(9**3):
            digit = str(index % 9 + 1)
            coords = tuple(draw_coords[index])
            if self.unbifurcated_possibles[index]:
                self.screen.addstr(
                    *coords,
                    digit,
                    curses.color_pair(
                        5 if self.in_valid_solutions[index] else 3
                    ),
                )

            if self.possibles[index]:
                self.screen.addstr(*coords, digit)
                if self.in_valid_solutions[index]:
                    self.screen.addstr(*coords, digit, curses.color_pair(4))

            if index in bifurcation_indices:
                first_bifurcation = index == bifurcation_indices[0]
                self.screen.addstr(
                    *coords,
                    digit,
                    curses.color_pair(1 if first_bifurcation else 2),
                )

        self.screen.refresh()

    @staticmethod
    def _possibles_to_draw_coords(possibles):
        reshaped_possibles = possibles[:-2].reshape((81, 9))
        index_offests = np.cumsum(reshaped_possibles, axis=1).reshape(9**3)
        cell_start_xs = ((np.arange(9**3) // 9) % 9) * 10
        cell_start_xs += 2 * (cell_start_xs >= 30) + 2 * (cell_start_xs >= 60)
        x_pos = cell_start_xs + index_offests - 1

        y_pos = np.arange(9**3) // 9**2
        y_pos += (y_pos >= 3).astype(int) + (y_pos >= 6).astype(int)

        return np.stack([y_pos + 1, x_pos]).T

    @staticmethod
    def _cell_start_index(row, col):
        return (row - 1) * 81 + (col - 1) * 9

    @staticmethod
    def possible_index(row, col, possible):
        return Board._cell_start_index(row, col) + (possible - 1)

    @staticmethod
    def possible_index_to_cell_index(index):
        return index // 9

    @staticmethod
    def possible_index_to_digit(index):
        return index % 9 + 1

    @staticmethod
    def _describe_cell_index(index):
        return f"R{index // 9 + 1}C{index % 9 + 1}"

    @staticmethod
    def _describe_possible_index(index):
        return f"R{index // 81 + 1}C{(index // 9) % 9 + 1} = {index % 9 + 1}"

    def _minpossible_index(self, cell_index):
        cell_start_index = cell_index * 9
        cell_possibles = self.possibles[cell_start_index : cell_start_index + 9]
        return cell_start_index + min(
            [i for i, possible in enumerate(cell_possibles) if possible]
        )

    def _remove_possibles(self, indices):
        _file_logger.info(
            "%sRemoving %s",
            " " * len(self.bifurcations),
            ",".join(map(str, indices)),
        )
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
        singleton_coverees_idx = coveree_counts == 1
        if np.any(coveree_counts == 0):
            raise SudokuContradiction("At least one coveree cannot be covered!")
        singleton_coverees = (
            self.coverees[singleton_coverees_idx]
            * coveree_status[singleton_coverees_idx]
        ).sum(axis=1)

        unfinalised_singletons = singleton_coverees[
            ~self.finalised[singleton_coverees]
        ]
        return unfinalised_singletons

    def _live_contradictions(self):
        """Contradictions that can still occur."""
        contradictions_that_could_still_happen = (1 - self.possibles)[
            self.contradictions
        ].sum(axis=1) == 0
        relevant_contradictions = self.contradictions[
            contradictions_that_could_still_happen
        ]
        # Remove digits that are already true and finalized - they're no longer
        # relevant to the contradictions
        irrelevant_index_mask = (self.possibles * self.finalised)[
            relevant_contradictions
        ]
        return (
            relevant_contradictions * (1 - irrelevant_index_mask)
            + CONTRADICTIONS_FILL * irrelevant_index_mask
        )

    def _live_coverees(self):
        """Coverees that still need covering."""
        still_need_covering = (self.possibles * self.finalised)[
            self.coverees
        ].sum(axis=1) == 0
        relevant_coverees = self.coverees[still_need_covering]
        # Remove any digits that are alerady not possible
        irrelevant_index_mask = self.possibles[relevant_coverees] == 0
        return (
            1 - irrelevant_index_mask
        ) * relevant_coverees + COVEREE_FILL * irrelevant_index_mask

    def _get_forced_contradictions(self):
        # 1 if an index can still be flipped to prevent a contradiction
        # 0 if that index is already finalised
        contradiction_status = 1 - (
            self.possibles[self.contradictions]
            * self.finalised[self.contradictions]
        )
        remaining_contradiction_outs = contradiction_status.sum(axis=1)
        forced_contradictions_idx = remaining_contradiction_outs == 1
        if np.any(remaining_contradiction_outs == 0):
            raise SudokuContradiction(
                "At least one contradiction has occurred!"
            )
        # Select the singleton indices that can still be flipped to prevent
        # contradictions.
        forced_contradictions = (
            self.contradictions[forced_contradictions_idx]
            * contradiction_status[forced_contradictions_idx]
        ).sum(axis=1)
        return forced_contradictions

    def add_coveree(self, cell_indices):
        if len(cell_indices) > MAX_COVEREE_SIZE:
            raise ValueError(
                "Coveree is too large! Max coveree size is {}".format(
                    MAX_COVEREE_SIZE
                )
            )
        index = len(self.coverees)
        self.coverees.append(cell_indices)

    def add_contradiction(self, *args):
        args = list(args)
        if len(args) > MAX_COVEREE_SIZE:
            raise ValueError(
                "Contradiction is too large! Max contradiction size is {}".format(
                    MAX_COVEREE_SIZE
                )
            )
        index = len(self.contradictions)
        self.contradictions.append(args)
