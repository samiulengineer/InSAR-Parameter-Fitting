import torch
import glob
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Scripts.Data.dataloader import SpatialTemporalDataset

 
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Sequential(
                nn.Conv2d(2, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 2, 3, 1, 1),
                )

    def forward(self, x):
        return self.fcn(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs = train_batch.float()
        out = self.forward(inputs)
        loss = F.mse_loss(inputs, out)
        self.log('train_loss', loss.item())
        return loss

# data
all_paths = glob.glob('/mnt/hdd1/alvinsun/3vG-Parameter-Fitting-Data/*/fit_hr/def_fit_cmpy')

# d = ParameterDataset(all_paths, 256, 100, 1500)
sample_filt_dir = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/'
sample_filt_ext = '.diff.orb.statm_cor.natm.filt'

sample_coh_dir = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/'
sample_coh_ext = '.diff.orb.statm_cor.natm.filt.coh'

sample_bperp_dir = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/'
sample_bperp_ext = '.bperp'

sample_width = 1500
sample_height = 1500

sample_conv1 = -0.0110745533168
sample_conv2 = -0.00134047881374

sample_patch_size = 28
sample_stride = 0.5

ref_mr_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/def_fit_cmpy'
ref_he_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/hgt_fit_m'

sample_db = SpatialTemporalDataset(sample_filt_dir, sample_filt_ext, sample_bperp_dir, sample_bperp_ext, sample_coh_dir, sample_coh_ext, sample_conv1, sample_conv2, sample_width, sample_height,ref_mr_path, ref_he_path, sample_patch_size, sample_stride)

training_dataloader = DataLoader(sample_db, 4, shuffle=True, num_workers=4)

# d = ParameterDatasetCombineMandH(all_paths, 500, 100, 1500)

# train_loader = DataLoader(d, batch_size=16,
                    # shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=1)
trainer.fit(model, training_dataloader)