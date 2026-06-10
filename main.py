import pygame

from agents import TrainedAgent, HumanAgent
from environment import IceEnv

from settings import Settings


def main():
    pygame.init()

    screen = pygame.display.set_mode((Settings.WIDTH, Settings.HEIGHT))
    pygame.display.set_caption("Robots on Ice")

    clock = pygame.time.Clock()

    env = IceEnv()
    agent = HumanAgent()

    state = env.reset()

    running = True

    while running:

        # -------------------------------------------------
        # EVENTS
        # -------------------------------------------------

        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
                running = False


        # -------------------------------------------------
        # AGENT ACTION
        # -------------------------------------------------

        action = agent.select_action(state)

        # -------------------------------------------------
        # ENVIRONMENT STEP
        # -------------------------------------------------

        next_state, reward, done = env.step(action)

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        # Store transition here
        # agent.remember(...)

        # Optional online update
        # agent.train()

        state = next_state

        # -------------------------------------------------
        # RENDER
        # -------------------------------------------------

        env.draw(screen)

        pygame.display.flip()

        # -------------------------------------------------
        # RESET EPISODE
        # -------------------------------------------------

        if done:
            print("Episode finished")
            state = env.reset()

        clock.tick(Settings.FPS)

    pygame.quit()

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
