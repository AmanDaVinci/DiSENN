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

    if config.dataloader.lower() == 'celeba':
        train_dataset = CelebA(split="train", data_path=path, download=True, target=config["target"])
        valid_dataset = CelebA(split="valid", data_path=path, download=False, target=config["target"])
        train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        valid_dl = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
        return train_dl, valid_dl
    else:
        return None