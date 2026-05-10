"""Farm grid representation for pest spread simulation.

Defines the cell state enum and the FarmGrid class that holds
the 2-D grid used by the spread model and intervention strategies.
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum
from typing import Tuple


class CellState(IntEnum):
    """Possible health states for a single farm grid cell.

    Susceptible: healthy crop, can become infected.
    Infected:    diseased crop, actively spreads to neighbours.
    Removed:     treated or harvested cell, no longer infectious.
    """
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2


# Human-readable colour map used by the dashboard (Lab 8)
CELL_STATE_COLOURS: dict[CellState, str] = {
    CellState.SUSCEPTIBLE: "#2ecc71",   # green
    CellState.INFECTED:    "#e74c3c",   # red
    CellState.REMOVED:     "#95a5a6",   # grey
}


class FarmGrid:
    """2-D grid representing a farm field for SIR-based simulation.

    Each cell holds a CellState value.  The grid is initialised
    as fully Susceptible and then seeded with an initial infection
    derived from the ML model detection result.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        random_seed: Seed for reproducible stochastic operations.
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        random_seed: int = 42,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

        # All cells start healthy
        self.grid: np.ndarray = np.full(
            (rows, cols), CellState.SUSCEPTIBLE, dtype=np.int8
        )

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_infection(
        self,
        num_initial_infections: int = 3,
        centre: bool = True,
    ) -> None:
        """Place initial infected cells on the grid.

        Args:
            num_initial_infections: How many cells to infect at start.
            centre: If True, seed near the grid centre (realistic field
                    scenario); if False, seed at random positions.
        """
        if centre:
            centre_row = self.rows // 2
            centre_col = self.cols // 2
            offsets = self.rng.integers(-2, 3, size=(num_initial_infections, 2))
            for offset_row, offset_col in offsets:
                row = int(np.clip(centre_row + offset_row, 0, self.rows - 1))
                col = int(np.clip(centre_col + offset_col, 0, self.cols - 1))
                self.grid[row, col] = CellState.INFECTED
        else:
            for _ in range(num_initial_infections):
                row = int(self.rng.integers(0, self.rows))
                col = int(self.rng.integers(0, self.cols))
                self.grid[row, col] = CellState.INFECTED

    # ------------------------------------------------------------------
    # State counts — used for logging and DSS scoring
    # ------------------------------------------------------------------

    def count_state(self, state: CellState) -> int:
        """Return the number of cells in a given state.

        Args:
            state: The CellState to count.

        Returns:
            Integer count of matching cells.
        """
        return int(np.sum(self.grid == state))

    def state_summary(self) -> dict[str, int]:
        """Return a dict with counts for all three states.

        Returns:
            Dictionary with keys 'susceptible', 'infected', 'removed'.
        """
        return {
            "susceptible": self.count_state(CellState.SUSCEPTIBLE),
            "infected":    self.count_state(CellState.INFECTED),
            "removed":     self.count_state(CellState.REMOVED),
        }

    def total_cells(self) -> int:
        """Return total number of cells in the grid."""
        return self.rows * self.cols

    def copy(self) -> "FarmGrid":
        """Return a deep copy of this grid (for parallel scenario runs).

        Returns:
            New FarmGrid with identical state.
        """
        new_grid = FarmGrid(
            rows=self.rows,
            cols=self.cols,
            random_seed=self.random_seed,
        )
        new_grid.grid = self.grid.copy()
        new_grid.rng = np.random.default_rng(seed=self.random_seed)
        return new_grid

    # ------------------------------------------------------------------
    # Neighbours
    # ------------------------------------------------------------------

    def get_susceptible_neighbours(self, row: int, col: int) -> list[Tuple[int, int]]:
        """Return coordinates of Susceptible neighbours (4-connectivity).

        Args:
            row: Row index of the cell.
            col: Column index of the cell.

        Returns:
            List of (row, col) tuples for neighbouring Susceptible cells.
        """
        neighbours: list[Tuple[int, int]] = []
        for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour_row = row + delta_row
            neighbour_col = col + delta_col
            if (
                0 <= neighbour_row < self.rows
                and 0 <= neighbour_col < self.cols
                and self.grid[neighbour_row, neighbour_col] == CellState.SUSCEPTIBLE
            ):
                neighbours.append((neighbour_row, neighbour_col))
        return neighbours


if __name__ == "__main__":
    grid = FarmGrid(rows=10, cols=10)
    grid.seed_infection(num_initial_infections=3)
    print("Initial state summary:", grid.state_summary())
    assert grid.count_state(CellState.INFECTED) == 3
    print("FarmGrid smoke test passed.")