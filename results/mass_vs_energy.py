from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import wandb
from tqdm import tqdm

from settings import TrainingSettings


def fetch_mass_vs_energy_data(
        wandb_path: str,
        experiment_label: str,
        metric_name: str,
):
    api = wandb.Api()
    runs = api.runs(wandb_path)

    # (mass, energy_coeff) -> list of full metric histories, one list per run
    data: dict[tuple, list[list[float]]] = defaultdict(list)

    for run in tqdm(runs):
        if run.state != "finished":
            continue

        training_id = run.config.get("TrainingSettings", {}).get("TRAINING_ID")
        if training_id != experiment_label:
            continue

        mass = run.config.get("RenderingSettings", {}).get("ROBOT_MASS")
        energy_coeff = run.config.get("TrainingSettings", {}).get("ENERGY_COEFF")

        if energy_coeff > 1:
            continue

        history = run.history(
            keys=[metric_name], x_axis="episode", samples=10_000, pandas=False
        )
        history = [row[metric_name] for row in history if row.get(metric_name) is not None]

        data[(mass, energy_coeff)].append(history)

    return data


def aggregate_window_data(
        data: dict,
        end_episode: int,
        last_n: int,
):
    aggregated = {}

    for key, run_histories in data.items():
        values = []
        for history in run_histories:
            window = history[end_episode - last_n:end_episode]
            values.extend(window)

        if values:
            aggregated[key] = sum(values) / len(values)

    return aggregated


def create_heatmap(data, title: str = "Mass vs Energy Heatmap", legend: str = "Return") -> go.Figure:
    """Create a heatmap from (mass, energy_coeff) -> average_return data."""
    masses = sorted(set(mass for mass, _ in data.keys()))
    energy_coeffs = sorted(set(energy for _, energy in data.keys()))

    # Build z_values: rows are masses, columns are energy coeffs
    z_values = []
    for mass in masses:
        row = []
        for energy in energy_coeffs:
            value = data.get((mass, energy), None)
            row.append(value)
        z_values.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=energy_coeffs,
            y=masses,
            colorscale="electric",
            colorbar={"title": legend},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Energy coefficient",
        yaxis_title="Robot mass",
    )
    return fig


def get_testing_data(*args, **kwargs):
    return {
        (50, 0): [list(range(1000, 1400)), list(range(1100, 1500))],
        (50, 0.25): [list(range(1200, 1600)), list(range(1250, 1650))],
        (50, 0.5): [list(range(900, 1300)), list(range(950, 1350))],
        (75, 0): [list(range(1100, 1500)), list(range(1150, 1550))],
        (75, 0.25): [list(range(1300, 1700)), list(range(1350, 1750))],
        (75, 0.5): [list(range(950, 1350)), list(range(1000, 1400))],
    }


def main(metric_name="train/return", legend="Return") -> None:
    experiment_label = "Mass vs Energy"
    last_n_returns = 5
    output_path = Path(__file__).resolve().parent / "out" / legend.lower().replace(" ", "_")
    checkpoints = [10, 20, 30, 40]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = fetch_mass_vs_energy_data(
        wandb_path="iamtypingsomethingrandom/Robots-On-Ice",
        experiment_label=experiment_label,
        metric_name=metric_name,
    )

    log_freq = TrainingSettings().LOG_FREQ
    for checkpoint in checkpoints:
        window_data = aggregate_window_data(data, checkpoint, last_n_returns)
        fig = create_heatmap(
            window_data,
            title=f"Mass vs Energy Heatmap (episodes {(checkpoint - last_n_returns) * log_freq} to {checkpoint * log_freq})",
            legend=legend,
        )
        # fig.write_html(output_path.with_name(f"{output_path.name}_{checkpoint}").with_suffix(".html"))
        fig.write_image(output_path.with_name(f"{output_path.name}_{checkpoint}").with_suffix(".png"))


if __name__ == "__main__":
    main()
    main("train/targets_collected", legend="Targets Collected")
