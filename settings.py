from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSettings:
    SEED = 47
    EPISODES = 1000
    BATCH_SIZE = 20
    ENERGY_COEFF = 0.1
    BUFFER_SIZE = 1000

@dataclass(frozen=True)
class RenderingSettings:
    ICE_FRICTION = 0.02  # Between 0.1 and 0.01 for the assignment
    WIDTH = 1500
    HEIGHT = 1000

    FPS = 30  # 10 for the assignment
    DT = 1.0 / FPS  # physics timestep

    ROBOT_WIDTH = 100
    TARGET_WIDTH = 80
    COLLECT_DISTANCE = 60

    MAX_FORCE = 40e3  # Newtons
    ROBOT_MASS = 75  # kg

    EPISODE_TIME = 20  # seconds
    MAX_STEPS = int(EPISODE_TIME * FPS)
