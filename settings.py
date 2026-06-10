from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    WIDTH = 1000
    HEIGHT = 700

    FPS = 10  # 10 decisions per second
    DT = 0.1  # physics timestep

    ROBOT_RADIUS = 15
    TARGET_RADIUS = 12

    MAX_FORCE = 500.0  # Newtons
    ROBOT_MASS = 30.0  # kg
    ENERGY_COEFF = 0.1

    EPISODE_TIME = 20  # seconds
    MAX_STEPS = int(EPISODE_TIME * FPS)

    WHITE = (255, 255, 255)
    BLUE = (50, 120, 255)
    RED = (255, 80, 80)
    BLACK = (20, 20, 20)
