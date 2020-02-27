from torch.utils.data import DataLoader
from .celeba_dataset import CelebA

def get_dataloader(config):
    """Generate the train and valid dataloaders given the dataset in config

    Parameters
    ----------
    config : dict
        dictionary must have the "data" key with value among the following:
        ["celeba"]
    
    Returns
    -------
    (train_dl, valid_dl): torch.utils.data.DataLoader
        train and valid dataloaders
    """
    assert config["data"] in ["celeba"], "supports CelebA dataset only"
    path = "data/"+config["data"]
    pin_memory = False if config['device'] is 'cpu' else True
    num_workers = config['num_workers']
    if config['data'] == 'celeba':
        train_dataset = CelebA(split="train", data_path=path, download=config['download'], target=config["target"])
        valid_dataset = CelebA(split="valid", data_path=path, download=False, target=config["target"])
        train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        valid_dl = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        return train_dl, valid_dl
    else:
        return None