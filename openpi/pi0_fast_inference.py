from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example
example = {
    "observation/exterior_image_1_left": None,
    "prompt": "pick up the box"
}

action_chunk = policy.infer(example)["actions"]