class SudokuContradiction(Exception):
    """Indicates a puzzle (or bifurcation) is not solvable."""


class NoBifurcations(Exception):
    """Indicates there are no valid bifurcations."""
