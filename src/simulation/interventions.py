"""Intervention strategies that can be applied to a FarmGrid mid-simulation.

Each strategy modifies the grid and/or the SpreadModel parameters to
simulate a real-world treatment decision.  The simulator runs each
strategy independently so the DSS (Lab 9) can compare outcomes.

Strategies:
  1. PesticideIntervention  — reduces beta (transmission rate) for all cells
  2. QuarantineIntervention — forcibly marks infected cells as Removed
                              within a radius, simulating crop removal
  3. NoIntervention         — baseline, no changes applied
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from farm_grid import FarmGrid, CellState
from spread_model import SpreadModel


class BaseIntervention(ABC):
    """Abstract base class for all intervention strategies.

    Each subclass must implement apply(), which modifies either the
    FarmGrid state or the SpreadModel parameters (or both).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in dashboard and DSS report."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description of what this intervention does."""
        ...

    @abstractmethod
    def apply(self, farm_grid: FarmGrid, spread_model: SpreadModel) -> None:
        """Apply the intervention to the grid and/or model.

        Args:
            farm_grid:    The FarmGrid to modify in-place.
            spread_model: The SpreadModel whose parameters may be adjusted.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class NoIntervention(BaseIntervention):
    """Baseline: no treatment is applied.

    Used as the reference scenario in the DSS comparison.
    """

    @property
    def name(self) -> str:
        return "No Intervention"

    @property
    def description(self) -> str:
        return "Disease spreads without any treatment — baseline scenario."

    def apply(self, farm_grid: FarmGrid, spread_model: SpreadModel) -> None:
        """No-op: leaves grid and model unchanged."""
        pass


class PesticideIntervention(BaseIntervention):
    """Apply pesticide across the entire field.

    Reduces the transmission probability (beta) to simulate the
    protective effect of a broad-spectrum fungicide or pesticide.
    Does not immediately remove infected cells — models preventative use.

    Args:
        beta_reduction_factor: Multiplier applied to beta (0 < factor < 1).
                               Default 0.3 means beta drops to 30% of original.
    """

    def __init__(self, beta_reduction_factor: float = 0.3) -> None:
        if not 0.0 < beta_reduction_factor < 1.0:
            raise ValueError(
                f"beta_reduction_factor must be in (0, 1), got {beta_reduction_factor}"
            )
        self.beta_reduction_factor = beta_reduction_factor

    @property
    def name(self) -> str:
        return "Pesticide Application"

    @property
    def description(self) -> str:
        return (
            f"Broad-spectrum pesticide reduces transmission rate to "
            f"{int(self.beta_reduction_factor * 100)}% of baseline."
        )

    def apply(self, farm_grid: FarmGrid, spread_model: SpreadModel) -> None:
        """Reduce spread_model.beta by the reduction factor.

        Args:
            farm_grid:    Unused — pesticide affects transmission, not current state.
            spread_model: Beta is multiplied by beta_reduction_factor.
        """
        spread_model.beta = spread_model.beta * self.beta_reduction_factor


class QuarantineIntervention(BaseIntervention):
    """Remove all infected cells within a quarantine radius.

    Simulates physical removal (harvesting or destroying) of diseased
    plants to create a firebreak that stops further spread.

    Args:
        quarantine_radius: Manhattan distance radius around each infected
                           cell within which all cells are marked Removed.
    """

    def __init__(self, quarantine_radius: int = 2) -> None:
        if quarantine_radius < 1:
            raise ValueError(f"quarantine_radius must be >= 1, got {quarantine_radius}")
        self.quarantine_radius = quarantine_radius

    @property
    def name(self) -> str:
        return "Quarantine & Removal"

    @property
    def description(self) -> str:
        return (
            f"Infected plants and all crops within radius {self.quarantine_radius} "
            f"are removed to prevent further spread."
        )

    def apply(self, farm_grid: FarmGrid, spread_model: SpreadModel) -> None:
        """Mark infected cells and their neighbours as Removed.

        Args:
            farm_grid:    Grid is modified in-place — infected zone removed.
            spread_model: Unused — quarantine acts on grid state, not parameters.
        """
        import numpy as np

        cells_to_remove: list[tuple[int, int]] = []
        infected_positions = list(zip(*np.where(farm_grid.grid == CellState.INFECTED)))

        for row, col in infected_positions:
            row, col = int(row), int(col)
            for delta_row in range(-self.quarantine_radius, self.quarantine_radius + 1):
                for delta_col in range(-self.quarantine_radius, self.quarantine_radius + 1):
                    # Manhattan distance check
                    if abs(delta_row) + abs(delta_col) <= self.quarantine_radius:
                        target_row = row + delta_row
                        target_col = col + delta_col
                        if 0 <= target_row < farm_grid.rows and 0 <= target_col < farm_grid.cols:
                            cells_to_remove.append((target_row, target_col))

        for target_row, target_col in cells_to_remove:
            farm_grid.grid[target_row, target_col] = CellState.REMOVED


# Registry used by the simulator — maps strategy name to instance
INTERVENTION_REGISTRY: dict[str, BaseIntervention] = {
    "no_intervention": NoIntervention(),
    "pesticide":       PesticideIntervention(beta_reduction_factor=0.3),
    "quarantine":      QuarantineIntervention(quarantine_radius=2),
}


if __name__ == "__main__":
    from farm_grid import FarmGrid
    from spread_model import SpreadModel

    for strategy_name, strategy in INTERVENTION_REGISTRY.items():
        grid = FarmGrid(rows=10, cols=10, random_seed=1)
        grid.seed_infection(num_initial_infections=4)
        model = SpreadModel(beta=0.3, gamma=0.05, random_seed=1)
        before = grid.state_summary()
        strategy.apply(grid, model)
        after = grid.state_summary()
        print(f"[{strategy.name}] before={before}  after={after}  beta={model.beta:.4f}")

    print("Interventions smoke test passed.")