import time
from abc import ABCMeta
from dataclasses import dataclass
from typing import Any


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        class_name = cls.__name__
        if class_name not in cls._instances:
            cls._instances[class_name] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[class_name]

    def clear_instance(self):
        self._instances.pop(self.__name__, None)


class SingletonABCMeta(ABCMeta, Singleton):
    pass


class Settings(metaclass=SingletonABCMeta):
    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}


@dataclass(frozen=False)
class TrainingSettings(Settings):
    WARMUP_STEPS = 50_000
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 256
    SEED = 47

    LOG_FREQ = 10
    VIDEO_FREQ = 50

    EPISODES = 1_000
    ENERGY_COEFF = 0.7  # 0-1
    COLLECTED_REWARD = 1e6
    METEORITE_REWARD = 0  # -1e6

    DISCOUNT_FACTOR = 0.995
    ACTOR_LEARNING_RATE = 1e-3  # Smaller than critic
    CRITIC_LEARNING_RATE = 1.5e-3
    ALPHA_LEARNING_RATE = 3e-4

    HIDDEN_ACTOR_NODES = 128
    HIDDEN_CRITIC_NODES = 128

    TAU = 0.005
    INIT_ALPHA = 0.2
    TARGET_ENTROPY = -2  # -action_dim for automatic entropy tuning
    TRAINING_ID = int(time.time())


@dataclass(frozen=False)
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

    MAX_STEPS_PER_TARGET = int(20 / DT)
    MAX_STEPS_PER_EPISODE = int(60 / DT)

    ENABLE_METEORITE = False
    METEORITE_WIDTH = 200
    MAX_METEORITE_SPEED = 2  # m/s


if __name__ == "__main__":
    # Create singleton
    RenderingSettings()
    RenderingSettings.clear_instance()
