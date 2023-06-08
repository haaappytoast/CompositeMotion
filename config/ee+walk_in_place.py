import numpy as np

env_cls = "ICCGANHumanoidEE"
env_params = dict(
    episode_length = 500,
    motion_file = "assets/motions/gym/chest_open+walk_in_place.json",
    goal_reward_weight = [0.5]

)

training_params = dict(
    max_epochs = 100000,
    save_interval = 10000,
    terminate_reward = -25
)

discriminators = {
    "front_jumping_jack/upper": dict(
        motion_file = "assets/motions/gym/front_jumping_jack.json",
        key_links = ["torso", "head", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link = "pelvis",
        local_pos = True,
        replay_speed = lambda n: np.random.uniform(0.8, 1.2, size=(n,))
        # ob_horizon = 2,
        # weight=0.2
    ),
    "walk_in_place/lower": dict(
        key_links = ["pelvis", "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"],
        parent_link = None
    )
}
