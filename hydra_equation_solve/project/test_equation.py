import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

import json


log = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='test_config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # ------------
    # data
    # ------------

    data_module = hydra.utils.instantiate(cfg.data)

    # ------------
    # model
    # ------------

    model = hydra.utils.instantiate(cfg.model)

    trainer = pl.Trainer(**(cfg.pl_trainer))

    # ------------
    # testing
    # ------------

    result = trainer.test(model, datamodule=data_module)
    log.info(result)
    with open(f'{trainer.log_dir}/out', 'w') as f:
        f.write(json.dumps(str(result), indent=4))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        exit(1)
