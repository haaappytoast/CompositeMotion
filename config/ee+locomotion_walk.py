import numpy as np

env_cls = "ICCGANHumanoidEE"
env_params = dict(
    episode_length = 500,
    motion_file = "assets/motions/clips_walk.yaml",
    goal_reward_weight = [0.5],
)

training_params = dict(
    max_epochs = 100000,
    save_interval = 10000,
    terminate_reward = -25
)

discriminators = {
    "walk/full": dict(
        parent_link = None,
    )
}