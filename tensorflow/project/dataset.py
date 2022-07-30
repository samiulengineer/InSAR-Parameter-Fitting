from mrc_insar_common.data import data_reader
from mrc_insar_common.util.sim import gen_sim_3d
from mrc_insar_common.util.utils import wrap
import numpy as np
import logging
import glob
import tqdm
import math
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.utils import Sequence
import re
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def get_delta_days(date_string):
    date_format = "%Y%m%d"
    tokens = re.split("_|\.", date_string)
    date1 = datetime.strptime(tokens[0], date_format)
    date2 = datetime.strptime(tokens[1], date_format)
    delta_days = np.abs((date2 - date1).days)
    return delta_days


class SpatialTemporalTrainDataset(Sequence):

    def __init__(self,
                 filt_dir,
                 filt_ext,
                 bperp_dir,
                 bperp_ext,
                 coh_dir,
                 coh_ext,
                 conv1,
                 conv2,
                 batch,
                 width,
                 height,
                 ref_mr_path,
                 ref_he_path,
                 name,
                 sim,
                 patch_size=64,
                 stride=0.5):
        
        # extract data paths from folder and join with extension
        self.filt_paths = sorted(
            glob.glob('{}/*{}'.format(filt_dir, filt_ext))) 
        self.bperp_paths = sorted(
            glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))

        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path
        self.conv1 = conv1
        self.conv2 = conv2
        self.width = width
        self.batch_size = batch
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
            self.bperps[idx] = data_reader.readBin(
                bperp_path, 1, 'float')[0][0]

        logger.info(f'stack {self.name} buffer data loaded')
        
        # NAP Module
        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride))
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

    def __len__(self):

        # total data after created patch
        return math.ceil(len(self.all_sample_coords)/self.batch_size)

    def __getitem__(self, idx):

        # extract patch row and column indices
        coords = self.all_sample_coords[idx * self.batch_size:(idx + 1) *self.batch_size]

        filt_input_list = []
        coh_input_list = []
        mr_target_list = []
        he_target_list = []
        ddays_list = []
        bperps_list = []
        conv1_list = []
        conv2_list = []
        input_list = []
        target_list = []

        for coord in coords:
            # extract single patch from original sample
            mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(
                coord[0], coord[1], self.patch_size, self.patch_size))
            he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(
                coord[0], coord[1], self.patch_size, self.patch_size))

            # [N, h ,w] for a single training sample,
            filt_input = np.zeros(
                [self.stack_size, self.patch_size, self.patch_size])
            # [N, h ,w] for a single training sample
            coh_input = np.zeros(
                [self.stack_size, self.patch_size, self.patch_size])

            if (self.sim):
                unwrap_sim_phase, sim_ddays, sim_bperps, sim_conv1, sim_conv2 = gen_sim_3d(
                    mr_target, he_target, self.stack_size)  # [h, w, N] unwrapped phase
                wrap_sim_phase = wrap(unwrap_sim_phase)
                filt_input = np.transpose(wrap_sim_phase, [2, 0, 1])  # [N, h, w]
                coh_input += 1  # coh is 1 for simuluation data
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
                    filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(
                        coord[0], coord[1], self.patch_size, self.patch_size)))

                    coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(
                        coord[0], coord[1], self.patch_size, self.patch_size))

            filt_input_list.append(np.transpose(filt_input.astype(np.float32), [1,2,0]))
            coh_input_list.append(np.transpose(coh_input.astype(np.float32), [1,2,0]))
            mr_target_list.append(np.expand_dims(mr_target, -1).astype(np.float32))
            he_target_list.append(np.expand_dims(he_target, -1).astype(np.float32))
            ddays_list.append(ddays.astype(np.float32))
            bperps_list.append(bperps.astype(np.float32))
            conv1_list.append(float(conv1))
            conv2_list.append(float(conv2))
        
        filt_input_list = np.array(filt_input_list)
        coh_input_list = np.array(coh_input_list)
        mr_target_list = np.array(mr_target_list)
        he_target_list = np.array(he_target_list)
        ddays_list = np.array(ddays_list)
        bperps_list = np.array(bperps_list)


        [B, H, W, N] = filt_input_list.shape

        ddays = np.reshape(ddays_list, (B, 1, 1, N)) * np.ones((B, H, W, N))
        bperps = np.reshape(bperps_list, (B, 1, 1, N)) * np.ones((B, H, W, N))

        input_list = np.concatenate((filt_input_list, coh_input_list, ddays, bperps), axis=-1)
        
        return tf.convert_to_tensor(np.array(input_list)), tf.convert_to_tensor(np.array(mr_target_list))

        # return {
        #     'phase': , # complex data
        #     'coh': , # real data
        #     'mr': , # real data
        #     'he': , # real data
        #     'ddays': ,
        #     'bperps': ,
        #     'conv1': np.array(conv1_list),
        #     'conv2': np.array(conv2_list)
        # }
    def get_random_data(self, idx=-1):
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """

        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.all_sample_coords))
        
        # extract patch row and column indices
        coords = [self.all_sample_coords[idx]]

        filt_input_list = []
        coh_input_list = []
        mr_target_list = []
        he_target_list = []
        ddays_list = []
        bperps_list = []
        conv1_list = []
        conv2_list = []
        input_list = []
        target_list = []

        for coord in coords:
            # extract single patch from original sample
            mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(
                coord[0], coord[1], self.patch_size, self.patch_size))
            he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(
                coord[0], coord[1], self.patch_size, self.patch_size))

            # [N, h ,w] for a single training sample,
            filt_input = np.zeros(
                [self.stack_size, self.patch_size, self.patch_size])
            # [N, h ,w] for a single training sample
            coh_input = np.zeros(
                [self.stack_size, self.patch_size, self.patch_size])

            if (self.sim):
                unwrap_sim_phase, sim_ddays, sim_bperps, sim_conv1, sim_conv2 = gen_sim_3d(
                    mr_target, he_target, self.stack_size)  # [h, w, N] unwrapped phase
                wrap_sim_phase = wrap(unwrap_sim_phase)
                filt_input = np.transpose(wrap_sim_phase, [2, 0, 1])  # [N, h, w]
                coh_input += 1  # coh is 1 for simuluation data
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
                    filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(
                        coord[0], coord[1], self.patch_size, self.patch_size)))

                    coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(
                        coord[0], coord[1], self.patch_size, self.patch_size))

            filt_input_list.append(np.transpose(filt_input.astype(np.float32), [1,2,0]))
            coh_input_list.append(np.transpose(coh_input.astype(np.float32), [1,2,0]))
            mr_target_list.append(np.expand_dims(mr_target, -1).astype(np.float32))
            he_target_list.append(np.expand_dims(he_target, -1).astype(np.float32))
            ddays_list.append(ddays.astype(np.float32))
            bperps_list.append(bperps.astype(np.float32))
            conv1_list.append(float(conv1))
            conv2_list.append(float(conv2))
        
        filt_input_list = np.array(filt_input_list)
        coh_input_list = np.array(coh_input_list)
        mr_target_list = np.array(mr_target_list)
        he_target_list = np.array(he_target_list)
        ddays_list = np.array(ddays_list)
        bperps_list = np.array(bperps_list)


        [B, H, W, N] = filt_input_list.shape

        ddays = np.reshape(ddays_list, (B, 1, 1, N)) * np.ones((B, H, W, N))
        bperps = np.reshape(bperps_list, (B, 1, 1, N)) * np.ones((B, H, W, N))

        input_list = np.concatenate((filt_input_list, coh_input_list, ddays, bperps), axis=-1)
        
        return tf.convert_to_tensor(np.array(input_list)), tf.convert_to_tensor(np.array(mr_target_list)), idx


class SpatialTemporalTestDataset(Sequence):

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
        self.filt_paths = sorted(
            glob.glob('{}/*{}'.format(filt_dir, filt_ext)))
        self.bperp_paths = sorted(
            glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
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

        self.mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(
            coord[0], coord[1], self.roi_height, self.roi_width))
        self.mr_target = np.expand_dims(self.mr_target, 0)
        self.he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(
            coord[0], coord[1], self.roi_height, self.roi_width))
        self.he_target = np.expand_dims(self.he_target, 0)

        # [N, h ,w] for a single training sample,
        self.filt_input = np.zeros(
            [self.stack_size, self.roi_height, self.roi_width])
        # [N, h ,w] for a single training sample
        self.coh_input = np.zeros(
            [self.stack_size, self.roi_height, self.roi_width])

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('\\')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(
                bperp_path, 1, 'float')[0][0]

            self.filt_input[idx] = np.angle(data_reader.readBin(
                self.filt_paths[idx], self.width, 'floatComplex', crop=(coord[0], coord[1], self.roi_height, self.roi_width)))

            self.coh_input[idx] = data_reader.readBin(self.coh_paths[idx], self.width, 'float', crop=(
                coord[0], coord[1], self.roi_height, self.roi_width))
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

def get_dataloader(config):
    train = SpatialTemporalTrainDataset(filt_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
                                                filt_ext=".diff.orb.statm_cor.natm.filt",
                                                coh_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
                                                coh_ext=".diff.orb.statm_cor.natm.filt.coh",
                                                bperp_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
                                                bperp_ext=".bperp",
                                                ref_mr_path="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/fit_hr/def_fit_cmpy",
                                                ref_he_path="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/fit_hr/hgt_fit_m",
                                                batch=config['batch_size'],
                                                conv1=-0.0110745533168,
                                                conv2=-0.00122202886324,
                                                width=1500,
                                                height=1500,
                                                sim=False,
                                                name="cortez",
                                                patch_size=config["patch_size"])
    valid = SpatialTemporalTrainDataset(filt_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cerroverde.tsx.sm_dsc.2250.1750.1500.1500/ifg_hr/",
                                                filt_ext=".diff.orb.statm_cor.natm.filt",
                                                coh_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cerroverde.tsx.sm_dsc.2250.1750.1500.1500/ifg_hr/",
                                                coh_ext=".diff.orb.statm_cor.natm.filt.coh",
                                                bperp_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cerroverde.tsx.sm_dsc.2250.1750.1500.1500/ifg_hr/",
                                                bperp_ext=".bperp",
                                                ref_mr_path="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cerroverde.tsx.sm_dsc.2250.1750.1500.1500/fit_hr/def_fit_cmpy",
                                                ref_he_path="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cerroverde.tsx.sm_dsc.2250.1750.1500.1500/fit_hr/hgt_fit_m",
                                                batch=config['batch_size'],
                                                conv1=-0.0110745533168,
                                                conv2=-0.00122202886324,
                                                width=1500,
                                                height=1500,
                                                sim=False,
                                                name="cortez",
                                                patch_size=config["patch_size"])
    return train, valid


# if __name__ == "__main__":

#     train_dataset = SpatialTemporalTrainDataset(filt_dir="D:/CSML_workPlace/parameter_fitting/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
#                                                 filt_ext=".diff.orb.statm_cor.natm.filt",
#                                                 coh_dir="D:/CSML_workPlace/parameter_fitting/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
#                                                 coh_ext=".diff.orb.statm_cor.natm.filt.coh",
#                                                 bperp_dir="D:/CSML_workPlace/parameter_fitting/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
#                                                 bperp_ext=".bperp",
#                                                 ref_mr_path="D:/CSML_workPlace/parameter_fitting/cortez.tsx.sm_dsc.3100.500.1500.1500/fit_hr/def_fit_cmpy",
#                                                 ref_he_path="D:/CSML_workPlace/parameter_fitting/cortez.tsx.sm_dsc.3100.500.1500.1500/fit_hr/hgt_fit_m",
#                                                 batch=4,
#                                                 conv1=-0.0110745533168,
#                                                 conv2=-0.00122202886324,
#                                                 width=1500,
#                                                 height=1500,
#                                                 sim=False,
#                                                 name="cortez")

#     for batch_idx, batch in enumerate(train_dataset):

#         ''' if we want to return not in dictionary we can check in this way '''

#         # phase, coh, ddays, bperps, mr, he = batch

#         # print(input_filt.shape)

#         ''' if we want to return in dictionary we can check in this way '''

#         print('Batch Index \t = {}'.format(batch_idx))
#         print('Phase Shape \t = {}'.format(batch['phase'].shape))
#         print('Coh Shape \t = {}'.format(batch['coh'].shape))
#         print('mr Shape \t = {}'.format(batch['mr'].shape))
#         print('he Shape \t = {}'.format(batch['he'].shape))
#         print('ddays Shape \t = {}'.format(batch['ddays'].shape))
#         print('bperps Shape \t = {}'.format(batch['bperps'].shape))

#         break

#     ''' vsulize sample patches in a batch from train dataset'''

#     print('\n ---------------- mr ---------------- \n')
#     fig, axs = plt.subplots(1, 4, figsize=(8, 2))
#     # print (batch['mr'][0][0].shape)
#     for i in range(batch['mr'].shape[0]):  # size of stack
#         im = axs[i].imshow(np.squeeze(batch['mr'][i]), cmap='jet',
#                            vmin=-np.pi, vmax=np.pi)
#         fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)

#     fig.tight_layout()
#     plt.suptitle("Motion Rate Map", fontsize=14)
#     plt.savefig("mr_real.png", dpi=800)
#     plt.show()

#     print('\n ---------------- he ---------------- \n')
#     fig, axs = plt.subplots(1, 4, figsize=(8, 2))
#     # print (batch['he'][0][0].shape)
#     for i in range(batch['he'].shape[0]):  # size of stack
#         im = axs[i].imshow(np.squeeze(batch['he'][i]), cmap='jet',
#                            vmin=-np.pi, vmax=np.pi)
#         fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        
#     fig.tight_layout()
#     plt.suptitle("Height Error Map", fontsize=14)
#     plt.savefig("he_real.png", dpi=800)
#     plt.show()
