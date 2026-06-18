from abc import ABCMeta
from dataclasses import dataclass
from typing import Any


class Settings(metaclass=ABCMeta):
    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}

@dataclass(frozen=True)
class TrainingSettings(Settings):
    SEED = 47
    LOG_FREQ = 100
    VIDEO_FREQ = 2000

    EPISODES = 100000
    ENERGY_COEFF = 0.1

    DISCOUNT_FACTOR = 0.995
    ACTOR_LEARNING_RATE = 1e-3  # Smaller than critic
    CRITIC_LEARNING_RATE = 1.5e-3


@dataclass(frozen=True)
class RenderingSettings(Settings):
    ICE_FRICTION = 0.02  # Between 0.1 and 0.01 for the assignment
    WIDTH = 1504  # Use multiples of 16 for better mp4 rendering
    HEIGHT = 1008

    FPS = 10  # 10 for the assignment
    DT = 1.0 / FPS  # physics timestep

    ROBOT_WIDTH = 100
    TARGET_WIDTH = 80
    COLLECT_DISTANCE = 60

    MAX_FORCE = 40e3  # Newtons
    ROBOT_MASS = 75  # kg

    EPISODE_TIME = 20  # 20 for the assignment seconds
    MAX_STEPS = int(EPISODE_TIME * FPS)
