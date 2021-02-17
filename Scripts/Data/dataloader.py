# import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import glob
import tqdm
import logging
import re

from mrc_insar_common.data import data_reader
from torch.utils.data import DataLoader, Dataset
from datetime import datetime


log = logging.getLogger(__name__)


def get_delta_days(date_string):
    date_format = "%Y%m%d"
    tokens = re.split("_|\.", date_string)
    date1 = datetime.strptime(tokens[0], date_format)
    date2 = datetime.strptime(tokens[1], date_format)
    delta_days = np.abs((date2 - date1).days)
    return delta_days


class SpatialTemporalDataset(Dataset):

    def __init__(self,
                 filt_dir,
                 filt_ext,
                 bperp_dir,
                 bperp_ext,
                 coh_dir,
                 coh_ext,
                 conv1,
                 conv2,
                 width,
                 height,
                 ref_mr_path,
                 ref_he_path,
                 patch_size=38,
                 stride=0.5):
        self.filt_paths = sorted(glob.glob('{}/*{}'.format(filt_dir, filt_ext)))
        self.bperp_paths = sorted(glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path
        self.conv1 = conv1
        self.conv2 = conv2
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.stride = stride

        self.stack_size = len(self.filt_paths)

        self.ddays = np.zeros(self.stack_size)
        self.bperps = np.zeros(self.stack_size)

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]
        print('bp dday loaded')

        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride))
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

    def __len__(self):
        return len(self.all_sample_coords)

    def __getitem__(self, idx):
        coord = self.all_sample_coords[idx]

        #motion_rate and height_error
        mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
        he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
    
        filt_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])    # [N, h ,w] for a single training sample, 
        coh_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])    # [N, h ,w] for a single training sample

        for i in range(self.stack_size):
            # !! here is an example that only uses phase information 
            filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(coord[0], coord[1], self.patch_size, self.patch_size)))

            coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

        return {
            'input': filt_input,    
            'coh': coh_input,
            'mr': np.expand_dims(mr_target, 0),
            'he': np.expand_dims(he_target, 0),
            'ddays':
                self.
                ddays,    # ddays and bperps are shared for all training samples in a stack, it can be used in a more effecient way, here is just an example
            'bperps':
                self.bperps    # 
        }


if __name__ == "__main__":

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

    print('db length {}'.format(len(sample_db)))

    sample_dataloader = DataLoader(sample_db, 4, shuffle=True, num_workers=4)

    for batch_idx, batch in enumerate(sample_dataloader):
        print(batch_idx)
        print(batch['input'].shape)
        print(batch['coh'].shape)
        print(batch['mr'].shape)
        print(batch['he'].shape)
        print(batch['ddays'].shape)
        print(batch['bperps'].shape)
        break

    # visualize sample patchs in a batch
    fig, axs = plt.subplots(1,4, figsize=(8,2))
    input_shape = batch['input'][0].shape # first training example
    for i in range(input_shape[2]): # size of stack
        im = axs[i].imshow(batch['input'][0][i], cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()