from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    WIDTH = 1500
    HEIGHT = 1000

    FPS = 10  # 10 decisions per second
    DT = 0.1  # physics timestep

    ROBOT_WIDTH = 80
    TARGET_WIDTH = 60

    MAX_FORCE = 600.0  # Newtons
    ROBOT_MASS = 4.0  # kg
    ENERGY_COEFF = 0.1

    EPISODE_TIME = 20  # seconds
    MAX_STEPS = int(EPISODE_TIME * FPS)
