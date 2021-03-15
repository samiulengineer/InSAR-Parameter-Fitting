import torch
from mrc_insar_common.data import data_reader 
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from mrc_insar_common.util.sim import gen_sim_3d
from mrc_insar_common.util.utils import wrap
import matplotlib.pyplot as plt
import numpy as np
import logging
import glob
import tqdm
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


def get_delta_days(date_string):
    date_format = "%Y%m%d"
    tokens = re.split("_|\.", date_string)
    date1 = datetime.strptime(tokens[0], date_format)
    date2 = datetime.strptime(tokens[1], date_format)
    delta_days = np.abs((date2 - date1).days)
    return delta_days


class SpatialTemporalTrainDataset(Dataset):

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
                 name,
                 sim,
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
        self.name = name
        self.sim = sim

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
        logger.info(f'stack {self.name} buffer data loaded')

        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride))
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

    def __len__(self):
        return len(self.all_sample_coords)

    def __getitem__(self, idx):
        coord = self.all_sample_coords[idx]

        mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
        he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

        filt_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])    # [N, h ,w] for a single training sample, 
        coh_input = np.zeros([self.stack_size, self.patch_size, self.patch_size])    # [N, h ,w] for a single training sample

        if (self.sim):
            unwrap_sim_phase, sim_ddays, sim_bperps, sim_conv1, sim_conv2 = gen_sim_3d(mr_target, he_target, self.stack_size) # [h, w, N] unwrapped phase
            wrap_sim_phase = wrap(unwrap_sim_phase)
            filt_input = np.transpose(wrap_sim_phase, [2,0,1]) # [N, h, w]
            coh_input += 1 # coh is 1 for simuluation data
            ddays = sim_ddays
            bperps = sim_bperps
            conv1 = sim_conv1
            conv2 = sim_conv2

        else:
            ddays = self.ddays
            bperps = self.bperps
            conv1 = self.conv1
            conv2 = self.conv2
            for i in range(self.stack_size):
                filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(coord[0], coord[1], self.patch_size, self.patch_size)))

                coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))

        return {
            'phase': filt_input.astype(np.float32),    
            'coh': coh_input.astype(np.float32),
            'mr': np.expand_dims(mr_target, 0).astype(np.float32),
            'he': np.expand_dims(he_target, 0).astype(np.float32),
            'ddays':ddays.astype(np.float32),
            'bperps': bperps.astype(np.float32),
            'conv1': float(conv1),
            'conv2': float(conv2)
        }

class SpatialTemporalTestDataset(Dataset):

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
                 name,
                 roi):
        self.filt_paths = sorted(glob.glob('{}/*{}'.format(filt_dir, filt_ext)))
        self.bperp_paths = sorted(glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path
        self.conv1 = conv1
        self.conv2 = conv2
        self.width = width
        self.height = height
        self.name = name
        self.roi = roi

        self.stack_size = len(self.filt_paths)

        self.ddays = np.zeros(self.stack_size)
        self.bperps = np.zeros(self.stack_size)

        coord = self.roi

        self.roi_height = self.roi[2] - self.roi[0]
        self.roi_width = self.roi[3] - self.roi[1]

        self.mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.roi_height, self.roi_width))
        self.mr_target = np.expand_dims(self.mr_target, 0)
        self.he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.roi_height, self.roi_width))
        self.he_target = np.expand_dims(self.he_target, 0)

        self.filt_input = np.zeros([self.stack_size, self.roi_height, self.roi_width])    # [N, h ,w] for a single training sample, 
        self.coh_input = np.zeros([self.stack_size, self.roi_height, self.roi_width])    # [N, h ,w] for a single training sample


        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(bperp_path, 1, 'float')[0][0]

            self.filt_input[idx] = np.angle(data_reader.readBin(self.filt_paths[idx], self.width, 'floatComplex', crop=(coord[0], coord[1], self.roi_height, self.roi_width)))

            self.coh_input[idx] = data_reader.readBin(self.coh_paths[idx], self.width, 'float', crop=(coord[0], coord[1], self.roi_height, self.roi_width))
        logger.info(f'stack {self.name} buffer data loaded')

    def __len__(self):
        return 1  # single ROI for each stack

    def __getitem__(self, idx):

        return {
            'phase': self.filt_input.astype(np.float32),    
            'coh': self.coh_input.astype(np.float32),
            'mr': self.mr_target.astype(np.float32),
            'he': self.he_target.astype(np.float32),
            'ddays':
                self.
                ddays.astype(np.float32),
            'bperps':
                self.bperps.astype(np.float32),    # 
            'conv1': float(self.conv1),
            'conv2': float(self.conv2)
        }


class RealInSARDataModule(LightningDataModule):
    def __init__(self, train_stacks, val_stacks, num_workers, patch_size: int=21, batch_size: int = 32, train_stride= 0.5,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.batch_size = batch_size


        self.train_dataset = torch.utils.data.ConcatDataset([SpatialTemporalTrainDataset(**stack, patch_size=patch_size, stride=train_stride) for stack in train_stacks])
        self.val_dataset = torch.utils.data.ConcatDataset([SpatialTemporalTestDataset(**stack) for stack in val_stacks])
        logger.info(f'len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}')



    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   drop_last=True,
                                                   pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.num_workers,
                                                 drop_last=True,
                                                 pin_memory=True)
        return val_loader
