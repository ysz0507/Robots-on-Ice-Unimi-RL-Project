import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pygame
import torch
import tqdm

from agents.agent import Transition
from agents.soft_actor_critic import SACAgent
from buffer import ReplayBuffer
from environment import ScaledIceEnv
from settings import RenderingSettings, TrainingSettings

os.environ["WANDB_SILENT"] = "true"

# --- Optimization budget (reduce for faster search) ---
N_TRIALS = 50
OPTUNA_EPISODES = 400


def objective(trial: optuna.Trial) -> float:
    # Reset singletons so each trial gets a fresh configuration
    TrainingSettings.clear_instance()
    RenderingSettings.clear_instance()

    # ── Training hyperparameters ──────────────────────────────────────────────
    TrainingSettings().DISCOUNT_FACTOR = trial.suggest_float("discount_factor", 0.980, 0.999)
    TrainingSettings().ACTOR_LEARNING_RATE = trial.suggest_float("actor_lr", 1e-5, 1e-2, log=True)
    TrainingSettings().CRITIC_LEARNING_RATE = trial.suggest_float("critic_lr", 1e-5, 1e-2, log=True)
    TrainingSettings().ALPHA_LEARNING_RATE = trial.suggest_float("alpha_lr", 1e-5, 1e-2, log=True)
    TrainingSettings().HIDDEN_ACTOR_NODES = trial.suggest_categorical("hidden_actor_nodes", [64, 128, 256])
    TrainingSettings().HIDDEN_CRITIC_NODES = trial.suggest_categorical("hidden_critic_nodes", [64, 128, 256])
    TrainingSettings().TAU = trial.suggest_float("tau", 1e-3, 2e-2, log=True)
    TrainingSettings().INIT_ALPHA = trial.suggest_float("init_alpha", 0.05, 0.50, log=True)
    TrainingSettings().BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    TrainingSettings().BUFFER_SIZE = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000, 500_000])
    TrainingSettings().ENERGY_COEFF = trial.suggest_float("energy_coeff", 0.0, 1.0)
    TrainingSettings().WARMUP_STEPS = trial.suggest_categorical("warmup_steps", [2_000, 5_000, 10_000, 20_000])

    # ── Environment hyperparameters ───────────────────────────────────────────
    RenderingSettings().ICE_FRICTION = trial.suggest_float("ice_friction", 0.01, 0.10, log=True)
    RenderingSettings().ROBOT_MASS = trial.suggest_int("robot_mass", 50, 100)

    # Reproducibility
    seed = TrainingSettings().SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pygame.init()
    pygame.display.set_mode(
        (RenderingSettings().WIDTH, RenderingSettings().HEIGHT),
        flags=pygame.HIDDEN,
    )

    try:
        env = ScaledIceEnv()
        agent = SACAgent()
        replay_buffer = ReplayBuffer(TrainingSettings().BUFFER_SIZE)

        # ── Warmup ────────────────────────────────────────────────────────────
        with tqdm.tqdm(total=TrainingSettings().WARMUP_STEPS, desc=f"Trial {trial.number} | Warmup",
                       leave=False) as pbar:
            while len(replay_buffer) < TrainingSettings().WARMUP_STEPS:
                state, done = env.reset(), False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    replay_buffer.store(Transition(state, action, reward, next_state, done))
                    pbar.update(1)
                    state = next_state

        # ── Training ──────────────────────────────────────────────────────────
        episode_returns: list[float] = []

        for episode in tqdm.tqdm(range(OPTUNA_EPISODES), desc=f"Trial {trial.number} | Train", leave=False):
            episode_return, done = 0.0, False
            state = env.reset()

            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                replay_buffer.store(Transition(state, action, reward, next_state, done))
                episode_return += reward
                state = next_state
                agent.train(*replay_buffer.sample(TrainingSettings().BATCH_SIZE))

            episode_returns.append(episode_return)

            # Report for pruning (rolling mean over last 10 episodes)
            trial.report(float(np.mean(episode_returns[-10:])), episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Objective: mean return over the final 20 episodes
        return float(np.mean(episode_returns[-20:]))

    finally:
        pygame.quit()


def run_optimization() -> optuna.Study:
    study_name = "robots-on-ice-hpo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path("optimization_results") / f"{study_name}_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / "study.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        storage=f"sqlite:///{db_path.resolve()}",
        load_if_exists=False,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ── Print results ─────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n=== Optuna Best Trial ===")
    print(f"  Trial  : {best.number}")
    print(f"  Return : {best.value:.4f}")
    print("  Params :")
    for key, value in best.params.items():
        print(f"    {key:<30s} = {value}")

    # ── Persist results ───────────────────────────────────────────────────────
    best_trial_path = study_dir / "best_trial.json"
    with open(best_trial_path, "w") as f:
        json.dump(
            {
                "best_trial": best.number,
                "best_value": best.value,
                "best_params": best.params,
                "n_trials": len(study.trials),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nBest parameters saved to {best_trial_path}")
    print(f"Study database saved to {db_path}")
    return study


if __name__ == "__main__":
    run_optimization()
