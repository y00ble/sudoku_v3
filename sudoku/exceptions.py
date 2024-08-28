class SudokuContradiction(Exception):
    """Indicates a puzzle (or bifurcation) is not solvable."""


class NoBifurcations(Exception):
    """Indicates there are no valid bifurcations."""


class DuplicateBifurcation(Exception):
    """Indicates that this route has already been explored."""
