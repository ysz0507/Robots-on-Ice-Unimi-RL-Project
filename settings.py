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
    BUFFER_SIZE = 200_000
    BATCH_SIZE = 1024
    SEED = 47

    LOG_FREQ = 10
    VIDEO_FREQ = 25

    EPISODES = 400
    ENERGY_COEFF = 0.9729714646106935  # 0-1
    COLLECTED_REWARD = 1e5  # 100 80 60 40 20 0
    METEORITE_REWARD = 0  # -1e6

    DISCOUNT_FACTOR = 0.9746667947051576
    ACTOR_LEARNING_RATE = 0.0005119333593712603  # Smaller than critic
    CRITIC_LEARNING_RATE = 0.005682048734624498
    ALPHA_LEARNING_RATE = 0.0005497678324141152

    HIDDEN_ACTOR_NODES = 265
    HIDDEN_CRITIC_NODES = 265

    TAU = 0.040688995907630504
    INIT_ALPHA = 0.36452787081040844
    TARGET_ENTROPY = -2  # -action_dim for automatic entropy tuning
    TRAINING_ID = int(time.time())


@dataclass(frozen=False)
class RenderingSettings(Settings):
    ICE_FRICTION = 0.02028359393854171  # Between 0.1 and 0.01 for the assignment
    WIDTH = 16 * 100  # 16m
    HEIGHT = 12 * 100  # 12m

    FPS = 20  # 10 for the assignment
    DT = 0.1  # physics timestep

    ROBOT_WIDTH = 100
    TARGET_WIDTH = 60
    COLLECT_DISTANCE = 60

    MAX_FORCE = 100  # Newtons should be 100 for the assignment
    ROBOT_MASS = 52  # kg, 50-100

    MAX_STEPS_PER_TARGET = int(20 / DT)
    MAX_STEPS_PER_EPISODE = int(60 / DT)

    ENABLE_METEORITE = False
    METEORITE_WIDTH = 200
    MAX_METEORITE_SPEED = 2  # m/s


if __name__ == "__main__":
    # Create singleton
    RenderingSettings()
    RenderingSettings.clear_instance()
