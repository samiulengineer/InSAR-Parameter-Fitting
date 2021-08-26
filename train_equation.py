import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


log = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='train_config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # ------------
    # data
    # ------------

    data_module = hydra.utils.instantiate(cfg.data)
    # train_dataloader = data_module.train_dataloader()

    # ------------
    # model
    # ------------

    model = hydra.utils.instantiate(cfg.model)

    # ------------
    # training
    # ------------

    trainer = pl.Trainer(**(cfg.pl_trainer), checkpoint_callback=True)

    trainer.fit(model, datamodule=data_module)
    # trainer.predict(model, datamodule=data_module)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        exit(1)
