""" Master Configuration

The master configuration dict will be imported by other configs
and updated with the new information defined for each config
"""

CONFIG = {
    "seed": 42,
    "debug": True,
    "device": 'cpu',
    "train": True,
    "download": False,
    "valid_freq": 2,
    "save_freq": 2
}