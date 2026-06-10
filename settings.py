from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    ICE_FRICTION = 0.02  # Between 0.1 and 0.01 for the assignment
    WIDTH = 1500
    HEIGHT = 1000

    FPS = 30  # 10 for the assignment
    DT = 1.0 / FPS  # physics timestep

    ROBOT_WIDTH = 80
    TARGET_WIDTH = 60
    COLLECT_DISTANCE = 60

    MAX_FORCE = 1000.0  # Newtons
    ROBOT_MASS = 2.0  # kg
    ENERGY_COEFF = 0.1

    EPISODE_TIME = 20  # seconds
    MAX_STEPS = int(EPISODE_TIME * FPS)
