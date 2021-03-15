import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from mrc_insar_common.data import data_reader
from mrc_insar_common.util.pt.init import weight_init
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.utils.tensorboard as tensorboard
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DummyCNN(pl.LightningModule):

    def __init__(self, in_channels=1, lr=1e-3, loss_type='mse', *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.in_channels = in_channels
        self.loss_type = loss_type

        # build dummy cnn
        self.net = nn.Sequential(nn.Conv2d(self.in_channels, 64, 3, padding=1),
                                 nn.ReLU(True), nn.Conv2d(
                                     64, 64, 3, padding=1),
                                 nn.ReLU(True), nn.Conv2d(
                                     64, 64, 3, padding=1),
                                 nn.ReLU(True), nn.Conv2d(64, 2, 3, padding=1))

        # call ultimate weigth init
        self.apply(weight_init)

    def forward(self, filt_ifg_phase):
        # This is a quick demo case that only use phase information as input
        out = self.net(filt_ifg_phase)
        return out

    def training_step(self, batch, batch_idx):
        # demo purpose, and our input batch are noisy signals
        # [B, 2, W, H] for real and imag two channels
        filt_ifg_phase = batch['phase']  # [B, N, H, W]
        ddays = batch['ddays']  # [B, N]
        bperps = batch['bperps']  # [B, N]
        mr = batch['mr']  # [B, 1, H, W]
        he = batch['he']  # [B, 1, H, W]
        conv1 = batch['conv1']  # [B]
        conv2 = batch['conv2']  # [B]
        ref_out = torch.cat([mr, he], 1)  # [B, 2, H, W]

        out = self.forward(filt_ifg_phase)  # [B, 2, H, W] prediction

        # there are two types of baseline losses but they may not work

        if (self.loss_type == 'mse'):
            # 1# simple RSME with ground truth
            loss = F.mse_loss(out, ref_out)  # simple RMSE loss

        elif (self.loss_type == 'ri-mse'):
            # 2# RI-MSE between input and recon phase
            [B, N] = ddays.shape
            ddays = torch.reshape(ddays, [B, N, 1, 1])
            bperps = torch.reshape(bperps, [B, N, 1, 1])
            conv1 = torch.reshape(conv1, [B, 1, 1, 1])
            conv2 = torch.reshape(conv1, [B, 1, 1, 1])
            out_mr = out[:, 0, :, :].unsqueeze(1)  # [B, 1, H, W]
            out_he = out[:, 1, :, :].unsqueeze(1)  # [B, 1, H, W]
            recon_phase = ddays * out_mr * conv1 + bperps * out_he * conv2
            loss = torch.square(
                torch.cos(recon_phase) -
                torch.cos(filt_ifg_phase)) + torch.square(
                    torch.cos(recon_phase) - torch.cos(filt_ifg_phase))
            loss = loss.mean()

        self.log('step_loss', loss, prog_bar=True, logger=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            filt_ifg_phase = batch['phase']  # [B, N, H, W]
            mr = batch['mr'].cpu().data.numpy()  # [B, 1, H, W]
            he = batch['he'].cpu().data.numpy()  # [B, 1, H, W]
            ref_mr_path = f'{self.trainer.log_dir}/ref_mr.png'
            ref_he_path = f'{self.trainer.log_dir}/ref_he.png'
            out = self.forward(filt_ifg_phase)  # [B, 2, H, W]

            out = out.cpu().data.numpy()
            out_mr = out[:, 0, :, :]
            out_he = out[:, 1, :, :]
            mr_rmse = np.sqrt(np.square(mr - out_mr)).mean()
            he_rmse = np.sqrt(np.square(he - out_he)).mean()
            est_mr_example_path = f'{self.trainer.log_dir}/{self.trainer.current_epoch}:{self.global_step}:{batch_idx}:rmse-{mr_rmse:.3f}.mr.png'
            est_he_example_path = f'{self.trainer.log_dir}/{self.trainer.current_epoch}:{self.global_step}:{batch_idx}:rmse-{he_rmse:.3f}.he.png'
            plt.imsave(ref_mr_path,
                       mr.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            plt.imsave(ref_he_path,
                       he.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            plt.imsave(est_mr_example_path,
                       out_mr.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            plt.imsave(est_he_example_path,
                       out_he.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            logger.info(f'save inference mr example to {est_mr_example_path}')
            logger.info(f'save inference he example to {est_he_example_path}')
            self.log('mr_rmse', mr_rmse.mean(), logger=True)
            self.log('he_rmse', he_rmse.mean(), logger=True)
        self.train()


class DnCNN(DummyCNN):

    def __init__(self,
                 in_channels=1,
                 lr=1e-3,
                 loss_type='mse',
                 num_of_layers=17,
                 *args,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.in_channels = in_channels
        self.loss_type = loss_type
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features,
                          out_channels=features,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=features,
                      out_channels=2,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)

        # call ultimate weigth init
        self.apply(weight_init)

    def forward(self, filt_ifg_phase):
        # This is a quick demo case that only use phase information as input
        out = self.dncnn(filt_ifg_phase)
        return out


class NewCnn(pl.LightningModule):

    def __init__(self,
                 in_channels=1,
                 lr=1e-3,
                 loss_type='mse',
                 *args,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.in_channels = in_channels
        self.loss_type = loss_type
        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv2d(
            self.in_channels, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=1,  kernel_size=kernel_size, padding=padding)
        self.conv_concat = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)

        ''' flattening '''

        self.cln1 = nn.Linear(4 * 28 * 28, 16)
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout2d(0.5)
        self.cln2 = nn.Linear(16, 8)

        ''' ddays + bperps in linear layer '''

        self.ln1 = nn.Linear(135, 64)
        self.ln2 = nn.Linear(64, 4)

        ''' concat layer '''

        self.ln_concat = nn.Linear(16, 2)

    def forward(self, filt_ifg_phase, coh, ddays, bperps):  # forward propagation
        # def forward(self, ddays, bperps):  # forward propagation
        ''' filt_ifg_phase and coh model design '''

        [B, N, H, W] = filt_ifg_phase.shape

        filt_ifg_phase = F.relu(self.conv1(filt_ifg_phase))
        filt_ifg_phase = F.relu(self.bn1(self.conv2(filt_ifg_phase)))
        filt_ifg_phase = F.relu(self.bn2(self.conv3(filt_ifg_phase)))
        filt_ifg_phase = F.relu(self.bn3(self.conv4(filt_ifg_phase)))
        filt_ifg_phase = F.relu(self.bn4(self.conv5(filt_ifg_phase)))
        filt_ifg_phase = F.relu(self.bn5(self.conv6(filt_ifg_phase)))
        filt_ifg_phase = F.relu(self.bn6(self.conv7(filt_ifg_phase)))
        filt_ifg_phase = self.conv8(filt_ifg_phase)

        coh = F.relu(self.conv1(coh))
        coh = F.relu(self.bn1(self.conv2(coh)))
        coh = F.relu(self.bn2(self.conv3(coh)))
        coh = F.relu(self.bn3(self.conv4(coh)))
        coh = F.relu(self.bn4(self.conv5(coh)))
        coh = F.relu(self.bn5(self.conv6(coh)))
        coh = F.relu(self.bn6(self.conv7(coh)))
        coh = self.conv8(coh)

        concat_phase_coh = F.relu(torch.cat((filt_ifg_phase, coh), dim=1))

        concat_phase_coh_flatten = concat_phase_coh.reshape(
            concat_phase_coh.shape[0], -1)

        flattenL1 = F.relu(self.cln1(concat_phase_coh_flatten))
        flattenL2 = F.relu(self.cln2(flattenL1))

        print('flattenL2 shape: {}' .format(flattenL2.shape))

        ''' ------------ ddays + bperps ------------- '''

        ddays = F.relu(self.ln1(ddays))
        ddays = F.relu(self.ln2(ddays))

        bperps = F.relu(self.ln1(bperps))
        bperps = F.relu(self. ln2(bperps))

        concat_ddays_bperps = F.relu(torch.cat((ddays, bperps), dim=1))

        all_concat = F.relu(torch.cat((flattenL2, concat_ddays_bperps), dim=1))

        last_layer_ddays_bperp_all = F.relu(self.ln_concat(all_concat))

        r = torch.reshape(last_layer_ddays_bperp_all, [B, 2, 1, 1])

        return r

    def training_step(self, batch, batch_idx):
        # demo purpose, and our input batch are noisy signals
        # [B, 2, W, H] for real and imag two channels
        filt_ifg_phase = batch['phase']  # [B, N, H, W]
        coh = batch['coh']  # [B, N, H, W]
        ddays = batch['ddays']  # [B, N]
        bperps = batch['bperps']  # [B, N]
        mr = batch['mr']  # [B, 1, H, W]
        he = batch['he']  # [B, 1, H, W]
        conv1 = batch['conv1']  # [B]
        conv2 = batch['conv2']  # [B]
        ref_out = torch.cat([mr, he], 1)  # [B, 2, H, W]

        out = self.forward(filt_ifg_phase, coh, ddays, bperps)  # [B, 2, 1, 1] prediction

        # there are two types of baseline losses but they may not work

        if (self.loss_type == 'mse'):
            # 1# simple RSME with ground truth
            loss = F.mse_loss(out, ref_out)  # simple RMSE loss

        elif (self.loss_type == 'kld'):
            loss = F.kl_div(out, ref_out, reduction='batchmean')

        elif (self.loss_type == 'ri-mse'):
            # 2# RI-MSE between input and recon phase
            [B, N] = ddays.shape
            ddays = torch.reshape(ddays, [B, N, 1, 1])
            bperps = torch.reshape(bperps, [B, N, 1, 1])
            conv1 = torch.reshape(conv1, [B, 1, 1, 1])
            conv2 = torch.reshape(conv1, [B, 1, 1, 1])
            out_mr = out[:, 0, :, :].unsqueeze(1)  # [B, 1, H, W]
            out_he = out[:, 1, :, :].unsqueeze(1)  # [B, 1, H, W]
            recon_phase = ddays * out_mr * conv1 + bperps * out_he * conv2
            loss = torch.square(
                torch.cos(recon_phase) -
                torch.cos(filt_ifg_phase)) + torch.square(
                    torch.cos(recon_phase) - torch.cos(filt_ifg_phase))
            loss = loss.mean()

        self.log('step_loss', loss, prog_bar=True, logger=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            filt_ifg_phase = batch['phase']  # [B, N, H, W]
            coh = batch['coh']  # [B, N, H, W]
            ddays = batch['ddays']  # [B, N]
            bperps = batch['bperps']
            mr = batch['mr'].cpu().data.numpy()  # [B, 1, H, W]
            he = batch['he'].cpu().data.numpy()  # [B, 1, H, W]
            ref_mr_path = f'{self.trainer.log_dir}/ref_mr.png'
            ref_he_path = f'{self.trainer.log_dir}/ref_he.png'
            out = self.forward(filt_ifg_phase, coh, ddays,
                               bperps)  # [B, 2, H, W]

            out = out.cpu().data.numpy()
            out_mr = out[:, 0, :, :]
            out_he = out[:, 1, :, :]
            mr_rmse = np.sqrt(np.square(mr - out_mr)).mean()
            he_rmse = np.sqrt(np.square(he - out_he)).mean()
            est_mr_example_path = f'{self.trainer.log_dir}/{self.trainer.current_epoch}:{self.global_step}:{batch_idx}:rmse-{mr_rmse:.3f}.mr.png'
            est_he_example_path = f'{self.trainer.log_dir}/{self.trainer.current_epoch}:{self.global_step}:{batch_idx}:rmse-{he_rmse:.3f}.he.png'
            plt.imsave(ref_mr_path,
                       mr.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            plt.imsave(ref_he_path,
                       he.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            plt.imsave(est_mr_example_path,
                       out_mr.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            plt.imsave(est_he_example_path,
                       out_he.squeeze(),
                       cmap='rainbow',
                       vmin=-5,
                       vmax=5)
            logger.info(f'save inference mr example to {est_mr_example_path}')
            logger.info(f'save inference he example to {est_he_example_path}')
            self.log('mr_rmse', mr_rmse.mean(), logger=True)
            self.log('he_rmse', he_rmse.mean(), logger=True)
        self.train()
