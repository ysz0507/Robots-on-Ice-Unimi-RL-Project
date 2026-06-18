from abc import ABCMeta
from dataclasses import dataclass
from typing import Any


class Settings(metaclass=ABCMeta):
    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}

@dataclass(frozen=True)
class TrainingSettings(Settings):
    WARMUP_STEPS = 50_000
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 256
    SEED = 47

    LOG_FREQ = 10
    VIDEO_FREQ = 50

    EPISODES = 10_000
    ENERGY_COEFF = 0.1  # 0-1

    DISCOUNT_FACTOR = 0.995
    ACTOR_LEARNING_RATE = 1e-3  # Smaller than critic
    CRITIC_LEARNING_RATE = 1.5e-3
    ALPHA_LEARNING_RATE = 3e-4

    HIDDEN_ACTOR_NODES = 128
    HIDDEN_CRITIC_NODES = 128

    TAU = 0.005
    INIT_ALPHA = 0.2
    TARGET_ENTROPY = -2  # -action_dim for automatic entropy tuning


@dataclass(frozen=True)
class RenderingSettings(Settings):
    ICE_FRICTION = 0.1  # Between 0.1 and 0.01 for the assignment
    WIDTH = 16 * 100  # 16m
    HEIGHT = 12 * 100  # 12m

    FPS = 20  # 10 for the assignment
    DT = 0.1  # physics timestep

    ROBOT_WIDTH = 100
    TARGET_WIDTH = 60
    COLLECT_DISTANCE = 60

    MAX_FORCE = 100  # Newtons should be 100 for the assignment
    ROBOT_MASS = 75  # kg, 50-100

    EPISODE_TIME = 20  # 20 for the assignment seconds
    MAX_STEPS = int(EPISODE_TIME * 1 / DT)
