from .master_config import CONFIG

config = {
    "exp_name": 'celeba_default',
    "epochs": 1,
    "data": 'celeba',
    "target": 'Male',
    "conceptizer": 'VaeConceptizer',
    "num_concepts": 10,
    "beta": 4,
    "parameterizer": 'ConvParameterizer',
    "num_classes": 2,
    "robustness_reg": 1e-4,
    "aggregator": 'SumAggregator',
    "concept_loss": 'bvae_loss',
    "robustness_loss": 'celeba_robustness_loss',
    "learning_rate": 3e-4,
    "batch_size": 128,
    "pretrain_epochs": 1,
    "pre-beta": 1
}

CONFIG.update(config)