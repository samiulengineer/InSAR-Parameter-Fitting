import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg.db))

    cfg.est_out = 

    # with open(cfg.output_path, 'w') as f:
    #     json.dump(result, f)
    # log.info('result saved to {}'.format(cfg.output_path))

if __name__ == "__main__":
    try:
        main(None)
    except Exception as e:
        raise e
