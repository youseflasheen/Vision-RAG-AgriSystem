"""SIR-based pest spread model for a single timestep.

Implements the transition rules:
  Susceptible  -> Infected   with probability beta  (per infected neighbour)
  Infected     -> Removed    with probability gamma (natural recovery / die-off)

Each call to step() advances the simulation by one day.
"""

from __future__ import annotations

import numpy as np
from src.lab7_simulation.farm_grid import FarmGrid, CellState


# Default disease parameters derived from PlantVillage literature estimates.
# beta:  transmission probability per infected neighbour per day
# gamma: recovery/removal probability per infected cell per day
DEFAULT_BETA: float = 0.3
DEFAULT_GAMMA: float = 0.05


class SpreadModel:
    """Stochastic SIR spread model operating on a FarmGrid.

    At each timestep:
    1. Every Infected cell attempts to infect each Susceptible neighbour
       with probability beta.
    2. Every Infected cell transitions to Removed with probability gamma.

    Updates are computed on a copy of the grid state so that all
    transitions within a single step are based on the same snapshot
    (avoids order-of-update bias).

    Args:
        beta: Transmission probability per infected neighbour per day.
        gamma: Recovery probability per infected cell per day.
        random_seed: Seed for reproducible stochastic transitions.
    """

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        random_seed: int = 42,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed=random_seed)

    def step(self, farm_grid: FarmGrid) -> dict[str, int]:
        """Advance the simulation by one timestep in-place.

        Args:
            farm_grid: The FarmGrid to update.

        Returns:
            State summary dict after the step
            (keys: 'susceptible', 'infected', 'removed').
        """
        # Work from a snapshot so all transitions see the same grid state
        snapshot = farm_grid.grid.copy()
        new_grid = snapshot.copy()

        infected_positions = list(zip(*np.where(snapshot == CellState.INFECTED)))

        for row, col in infected_positions:
            row, col = int(row), int(col)

            # --- S -> I: attempt to infect each susceptible neighbour ---
            for neighbour_row, neighbour_col in farm_grid.get_susceptible_neighbours(row, col):
                # Check snapshot state (not new_grid) to avoid cascade within one step
                if snapshot[neighbour_row, neighbour_col] == CellState.SUSCEPTIBLE:
                    if self.rng.random() < self.beta:
                        new_grid[neighbour_row, neighbour_col] = CellState.INFECTED

            # --- I -> R: infected cell may recover/be removed ---
            if self.rng.random() < self.gamma:
                new_grid[row, col] = CellState.REMOVED

        farm_grid.grid = new_grid
        return farm_grid.state_summary()


if __name__ == "__main__":
    grid = FarmGrid(rows=15, cols=15, random_seed=0)
    grid.seed_infection(num_initial_infections=5)
    model = SpreadModel(beta=0.3, gamma=0.05, random_seed=0)

    print("Step 0:", grid.state_summary())
    for day in range(1, 6):
        summary = model.step(grid)
        print(f"Step {day}:", summary)

    assert grid.count_state(CellState.SUSCEPTIBLE) + \
           grid.count_state(CellState.INFECTED) + \
           grid.count_state(CellState.REMOVED) == grid.total_cells()
    print("SpreadModel smoke test passed.")