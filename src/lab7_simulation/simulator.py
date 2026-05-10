"""Main simulation runner for pest spread scenario comparison.

Runs the SIR spread model under each available intervention strategy
and returns a structured results dictionary consumed by:
  - Lab 9 (DSS) for ranking intervention strategies
  - Lab 8 (Dashboard) for visualising spread over time

Usage from other modules:
    from src.lab7_simulation.simulator import run_simulation
    results = run_simulation(disease_name="Tomato___Septoria_leaf_spot")
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.lab7_simulation.farm_grid import FarmGrid, CellState
from src.lab7_simulation.spread_model import SpreadModel, DEFAULT_BETA, DEFAULT_GAMMA
from src.lab7_simulation.interventions import INTERVENTION_REGISTRY, BaseIntervention

# Simulation defaults
DEFAULT_GRID_ROWS: int = 20
DEFAULT_GRID_COLS: int = 20
DEFAULT_SIMULATION_DAYS: int = 30
DEFAULT_INITIAL_INFECTIONS: int = 3

# Disease-specific beta overrides based on known aggressiveness
DISEASE_BETA_OVERRIDES: dict[str, float] = {
    "Tomato___Late_blight":              0.45,
    "Tomato___Early_blight":             0.30,
    "Tomato___Septoria_leaf_spot":       0.28,
    "Corn_(maize)___Northern_Leaf_Blight": 0.35,
    "Potato___Late_blight":              0.50,
    "Apple___Apple_scab":                0.25,
    "Corn_(maize)___Common_rust_":       0.32,
}


def _get_beta_for_disease(disease_name: str) -> float:
    """Return the transmission rate for a specific disease.

    Falls back to DEFAULT_BETA if the disease is not in the override table.

    Args:
        disease_name: PlantVillage-format disease class name.

    Returns:
        Float beta transmission probability.
    """
    return DISEASE_BETA_OVERRIDES.get(disease_name, DEFAULT_BETA)


def _run_single_scenario(
    disease_name: str,
    intervention: BaseIntervention,
    num_days: int,
    grid_rows: int,
    grid_cols: int,
    initial_infections: int,
    random_seed: int,
) -> dict[str, Any]:
    """Run one intervention scenario and collect daily state counts.

    Args:
        disease_name:        Disease class name (used to set beta).
        intervention:        The intervention strategy to apply.
        num_days:            Number of simulation days to run.
        grid_rows:           Rows in the farm grid.
        grid_cols:           Columns in the farm grid.
        initial_infections:  Number of cells infected on day 0.
        random_seed:         Seed for reproducibility.

    Returns:
        Dictionary with scenario results including daily counts.
    """
    beta = _get_beta_for_disease(disease_name)

    grid = FarmGrid(rows=grid_rows, cols=grid_cols, random_seed=random_seed)
    grid.seed_infection(num_initial_infections=initial_infections, centre=True)

    model = SpreadModel(beta=beta, gamma=DEFAULT_GAMMA, random_seed=random_seed)

    # Apply intervention before simulation starts (day 0 treatment)
    intervention.apply(grid, model)

    daily_counts: list[dict[str, int]] = []
    peak_infected: int = 0

    # Day 0 snapshot
    day_zero_summary = grid.state_summary()
    day_zero_summary["day"] = 0
    daily_counts.append(day_zero_summary)
    peak_infected = max(peak_infected, day_zero_summary["infected"])

    # Advance day by day
    for day in range(1, num_days + 1):
        daily_summary = model.step(grid)
        daily_summary["day"] = day
        daily_counts.append(daily_summary)
        peak_infected = max(peak_infected, daily_summary["infected"])

    final_summary = grid.state_summary()
    total_crop_loss = final_summary["removed"] + final_summary["infected"]
    spread_contained = final_summary["infected"] == 0

    return {
        "intervention_name": intervention.name,
        "description":       intervention.description,
        "daily_counts":      daily_counts,
        "final_summary":     final_summary,
        "peak_infected":     peak_infected,
        "total_crop_loss":   total_crop_loss,
        "spread_contained":  spread_contained,
    }


def run_simulation(
    disease_name: str = "Tomato___Septoria_leaf_spot",
    num_days: int = DEFAULT_SIMULATION_DAYS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
    initial_infections: int = DEFAULT_INITIAL_INFECTIONS,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run all intervention scenarios and return a structured results dict.

    This is the primary function called by Lab 9 (DSS) and Lab 8 (Dashboard).

    Args:
        disease_name:       Disease predicted by the ResNet-50 model (Lab 5).
        num_days:           Number of days to simulate.
        grid_rows:          Rows in the farm grid.
        grid_cols:          Columns in the farm grid.
        initial_infections: Number of cells infected on day 0.
        random_seed:        Seed for reproducibility across scenarios.

    Returns:
        Dictionary with simulation results and ranked strategies.
    """
    total_cells = grid_rows * grid_cols
    scenarios: list[dict[str, Any]] = []

    for strategy_name, intervention in INTERVENTION_REGISTRY.items():
        scenario_result = _run_single_scenario(
            disease_name=disease_name,
            intervention=intervention,
            num_days=num_days,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            initial_infections=initial_infections,
            random_seed=random_seed,
        )
        scenario_result["strategy_key"] = strategy_name
        scenario_result["crop_loss_percent"] = round(
            (scenario_result["total_crop_loss"] / total_cells) * 100, 2
        )
        scenarios.append(scenario_result)

    # Rank scenarios: lowest total crop loss = best strategy
    ranked = sorted(scenarios, key=lambda scenario: scenario["total_crop_loss"])
    ranked_names = [scenario["intervention_name"] for scenario in ranked]

    return {
        "disease_name":          disease_name,
        "grid_size":             [grid_rows, grid_cols],
        "num_days":              num_days,
        "total_cells":           total_cells,
        "scenarios":             scenarios,
        "ranked_by_crop_loss":   ranked_names,
        "best_strategy":         ranked[0]["intervention_name"],
        "best_crop_loss_percent": ranked[0]["crop_loss_percent"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pest spread simulation.")
    parser.add_argument(
        "--disease", type=str, default="Tomato___Septoria_leaf_spot",
        help="PlantVillage disease class name.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_SIMULATION_DAYS)
    parser.add_argument("--rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_GRID_COLS)
    args = parser.parse_args()

    results = run_simulation(
        disease_name=args.disease,
        num_days=args.days,
        grid_rows=args.rows,
        grid_cols=args.cols,
    )

    print(f"\nSimulation complete for: {results['disease_name']}")
    print(f"Grid: {results['grid_size'][0]}x{results['grid_size'][1]}  |  Days: {results['num_days']}")
    print(f"\nStrategy ranking (best to worst crop loss):")
    for rank, scenario in enumerate(
        sorted(results["scenarios"], key=lambda s: s["total_crop_loss"]), start=1
    ):
        status = "✓ contained" if scenario["spread_contained"] else "✗ still spreading"
        print(
            f"  {rank}. {scenario['intervention_name']:30s} "
            f"crop loss: {scenario['crop_loss_percent']:5.1f}%  "
            f"peak infected: {scenario['peak_infected']:4d}  "
            f"{status}"
        )
    print(f"\nBest strategy: {results['best_strategy']}")