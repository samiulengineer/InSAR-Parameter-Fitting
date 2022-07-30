import logging

import pytorch_lightning as pl
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# logger = logging.getLogger(__name__)

torch.set_default_tensor_type(torch.FloatTensor)


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # ------------
    # data
    # ------------
    data_module = hydra.utils.instantiate(cfg.data)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # ------------
    # model
    # ------------
    model = hydra.utils.instantiate(cfg.model)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(**(cfg.pl_trainer),
                         checkpoint_callback=True)
    trainer.fit(model, train_dataloader=train_dataloader,
                val_dataloaders=[val_dataloader])

    # ------------
    # testing
    # ------------
    # TODO


if __name__ == '__main__':
    # try:
    main()
    # except Exception as e:
    # logger.error(e)
    # exit(1)
