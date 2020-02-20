import argparse
import importlib
from disenn.trainer import DiSENN_Trainer


def main():
    """ Runs the trainer based on the given experiment configuration """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs.celeba_test", help='experiment configuration dict')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    trainer = DiSENN_Trainer(config_module.CONFIG)
    trainer.run()


if __name__ == "__main__":
    main()
