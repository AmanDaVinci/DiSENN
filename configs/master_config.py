""" Master Configuration

The master configuration dict will be imported by other configs
and updated with the new information defined for each config
"""

CONFIG = {
    "seed": 42,
    "debug": False,
    "device": 'cuda',
    "num_workers": 4,
    "train": True,
    "download": False,
    "valid_freq": 50,
    "save_freq": 100,
    "anneal_step": 1
}