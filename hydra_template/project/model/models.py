import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from mrc_insar_common.util.pt.init import weight_init
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
# import torch.utils.tensorboard as tensorboard
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

        self.log('step_loss', loss, prog_bar=True,
                 logger=True, on_step=True, log_graph=True)
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
            in_channels=64, out_channels=2,  kernel_size=kernel_size, padding=padding)

        self.cnn = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)

        # call ultimate weigth init
        self.apply(weight_init)

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

        cnn_concat_phase_coh = F.relu(self.cnn(concat_phase_coh))

        ddays = torch.reshape(ddays, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()
        bperps = torch.reshape(bperps, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()

        ddays = F.relu(self.conv1(ddays))
        ddays = F.relu(self.bn1(self.conv2(ddays)))
        ddays = F.relu(self.bn2(self.conv3(ddays)))
        ddays = F.relu(self.bn3(self.conv4(ddays)))
        ddays = F.relu(self.bn4(self.conv5(ddays)))
        ddays = F.relu(self.bn5(self.conv6(ddays)))
        ddays = F.relu(self.bn6(self.conv7(ddays)))
        ddays = self.conv8(ddays)

        bperps = F.relu(self.conv1(bperps))
        bperps = F.relu(self.bn1(self.conv2(bperps)))
        bperps = F.relu(self.bn2(self.conv3(bperps)))
        bperps = F.relu(self.bn3(self.conv4(bperps)))
        bperps = F.relu(self.bn4(self.conv5(bperps)))
        bperps = F.relu(self.bn5(self.conv6(bperps)))
        bperps = F.relu(self.bn6(self.conv7(bperps)))
        bperps = self.conv8(bperps)

        concat_ddays_bperps = F.relu(torch.cat((ddays, bperps), dim=1))

        cnn_concat_ddays_bperps = F.relu(self.cnn(concat_ddays_bperps))

        concat_all = F.relu(
            torch.cat((cnn_concat_phase_coh, cnn_concat_ddays_bperps), dim=1))

        return concat_all

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
        # ref_out = torch.cat([mr, he], 1)  # [B, 2, H, W]
        ref_out = torch.cat([mr, he], 1)  # [B, 2, H, W]

        out = self.forward(filt_ifg_phase, coh, ddays,
                           bperps)  # [B, 2, H, W] prediction

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


class NandanFantasy(NewCnn):

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
            self.in_channels*4, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            in_channels=1080, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(
            in_channels=1080, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(
            in_channels=1080, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(
            in_channels=1080, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv6 = nn.Conv2d(
            in_channels=1080, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv7 = nn.Conv2d(
            in_channels=1080, out_channels=1080, kernel_size=kernel_size, padding=padding)
        self.conv8 = nn.Conv2d(
            in_channels=1080, out_channels=270,  kernel_size=kernel_size, padding=padding)

        self.conv9 = nn.Conv2d(
            in_channels=270, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv10 = nn.Conv2d(
            in_channels=540, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv11 = nn.Conv2d(
            in_channels=540, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv12 = nn.Conv2d(
            in_channels=540, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv13 = nn.Conv2d(
            in_channels=540, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv14 = nn.Conv2d(
            in_channels=540, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv15 = nn.Conv2d(
            in_channels=540, out_channels=540, kernel_size=kernel_size, padding=padding)
        self.conv16 = nn.Conv2d(
            in_channels=540, out_channels=135,  kernel_size=kernel_size, padding=padding)

        self.conv17 = nn.Conv2d(
            in_channels=135, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv18 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv19 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv20 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv21 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv22 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv23 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv24 = nn.Conv2d(
            in_channels=64, out_channels=2,  kernel_size=kernel_size, padding=padding)

        self.bn2 = nn.BatchNorm2d(1080)
        self.bn3 = nn.BatchNorm2d(1080)
        self.bn4 = nn.BatchNorm2d(1080)
        self.bn5 = nn.BatchNorm2d(1080)
        self.bn6 = nn.BatchNorm2d(1080)
        self.bn7 = nn.BatchNorm2d(1080)
        self.bn8 = nn.BatchNorm2d(270)

        self.bn9 = nn.BatchNorm2d(540)

        self.bn10 = nn.BatchNorm2d(540)
        self.bn11 = nn.BatchNorm2d(540)
        self.bn12 = nn.BatchNorm2d(540)
        self.bn13 = nn.BatchNorm2d(540)
        self.bn14 = nn.BatchNorm2d(540)
        self.bn15 = nn.BatchNorm2d(540)
        self.bn16 = nn.BatchNorm2d(135)
        self.bn17 = nn.BatchNorm2d(64)
        self.bn18 = nn.BatchNorm2d(64)
        self.bn19 = nn.BatchNorm2d(64)
        self.bn20 = nn.BatchNorm2d(64)
        self.bn21 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)
        self.bn23 = nn.BatchNorm2d(64)

        # call ultimate weigth init
        self.apply(weight_init)

    def forward(self, filt_ifg_phase, coh, ddays, bperps):  # forward propagation
        # def forward(self, ddays, bperps):  # forward propagation
        ''' filt_ifg_phase and coh model design '''

        [B, N, H, W] = filt_ifg_phase.shape

        ddays = torch.reshape(ddays, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()
        bperps = torch.reshape(bperps, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()

        concat_all = F.relu(
            torch.cat((filt_ifg_phase, coh, ddays, bperps), dim=1))

        concat_all = F.relu(self.conv1(concat_all))
        concat_all = F.relu(self.bn2(self.conv2(concat_all)))
        concat_all = F.relu(self.bn3(self.conv3(concat_all)))
        concat_all = F.relu(self.bn4(self.conv4(concat_all)))
        concat_all = F.relu(self.bn5(self.conv5(concat_all)))
        concat_all = F.relu(self.bn6(self.conv6(concat_all)))
        concat_all = F.relu(self.bn7(self.conv7(concat_all)))

        concat_all = F.relu(self.bn8(self.conv8(concat_all)))
        concat_all = F.relu(self.bn9(self.conv9(concat_all)))
        concat_all = F.relu(self.bn10(self.conv10(concat_all)))
        concat_all = F.relu(self.bn11(self.conv11(concat_all)))
        concat_all = F.relu(self.bn12(self.conv12(concat_all)))
        concat_all = F.relu(self.bn13(self.conv13(concat_all)))

        concat_all = F.relu(self.bn14(self.conv14(concat_all)))
        concat_all = F.relu(self.bn15(self.conv15(concat_all)))
        concat_all = F.relu(self.bn16(self.conv16(concat_all)))
        concat_all = F.relu(self.bn17(self.conv17(concat_all)))
        concat_all = F.relu(self.bn18(self.conv18(concat_all)))
        concat_all = F.relu(self.bn19(self.conv19(concat_all)))

        concat_all = F.relu(self.bn20(self.conv20(concat_all)))
        concat_all = F.relu(self.bn21(self.conv21(concat_all)))
        concat_all = F.relu(self.bn22(self.conv22(concat_all)))
        concat_all = F.relu(self.bn23(self.conv23(concat_all)))
        concat_all = self.conv24(concat_all)

        return concat_all


class Exp_3(NewCnn):
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
            self.in_channels*4, out_channels=256, kernel_size=kernel_size, padding=padding)  # [B,256,64,64]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # [B,256,32,32]
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=kernel_size, padding=padding)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # [B,64,8,8]

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=128, kernel_size=2, padding=0, stride=2)
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=256, kernel_size=2, padding=0, stride=2)
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=self.in_channels*4, kernel_size=2, padding=0, stride=2)

        # call ultimate weigth init
        self.apply(weight_init)

    def forward(self, filt_ifg_phase, coh, ddays, bperps):  # forward propagation
        # def forward(self, ddays, bperps):  # forward propagation
        ''' filt_ifg_phase and coh model design '''

        [B, N, H, W] = filt_ifg_phase.shape

        ddays = torch.reshape(ddays, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()
        bperps = torch.reshape(bperps, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()

        concat_all = F.relu(
            torch.cat((filt_ifg_phase, coh, ddays, bperps), dim=1))

        concat_all = F.relu(self.conv1(concat_all))
        concat_all = F.relu(self.pool1(concat_all))
        concat_all = F.relu(self.conv2(concat_all))
        concat_all = F.relu(self.pool2(concat_all))
        concat_all = F.relu(self.conv3(concat_all))
        concat_all = F.relu(self.pool3(concat_all))

        concat_all = F.relu(self.upsample1(concat_all))
        concat_all = F.relu(self.upsample2(concat_all))
        concat_all = F.relu(self.upsample3(concat_all))

        return concat_all


class RBDN(NewCnn):
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

        self.conv_input = nn.Conv2d(
            self.in_channels*4, out_channels=64, kernel_size=9, padding=4)

        self.conv_middle = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv_concat = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.pooling_Layer = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=True)

        self.unpool_layer = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv_layer = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.deconv_output = nn.ConvTranspose2d(
            in_channels=64, out_channels=2, kernel_size=9, padding=4)

        self.bn_layer = nn.BatchNorm2d(64)

        # call ultimate weigth init
        self.apply(weight_init)

    def forward(self, filt_ifg_phase, coh, ddays, bperps):  # forward propagation
        # def forward(self, ddays, bperps):  # forward propagation
        ''' filt_ifg_phase and coh model design '''

        [B, N, H, W] = filt_ifg_phase.shape

        ddays = torch.reshape(ddays, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()
        bperps = torch.reshape(bperps, [B, N, 1, 1]).cuda(
        ) * torch.ones([B, N, H, W]).cuda()

        concat_all = F.relu(
            torch.cat((filt_ifg_phase, coh, ddays, bperps), dim=1))

        # Model data pass start

        conv1 = F.relu(self.bn_layer(self.conv_input(concat_all)))

        pool1, indice1 = self.pooling_Layer(conv1)
        pool1 = F.relu(pool1)

        convB11 = F.relu(self.bn_layer(self.conv_middle(pool1)))

        poolB1, indiceB1 = self.pooling_Layer(convB11)
        poolB1 = F.relu(poolB1)

        convB21 = F.relu(self.bn_layer(self.conv_middle(poolB1)))

        poolB2, indiceB2 = self.pooling_Layer(convB21)
        poolB2 = F.relu(poolB2)

        convB31 = F.relu(self.bn_layer(self.conv_middle(poolB2)))

        poolB3, indiceB3 = self.pooling_Layer(convB31)
        poolB3 = F.relu(poolB3)

        convB32 = F.relu(self.bn_layer(self.conv_middle(poolB3)))

        unpoolB3 = F.relu(self.unpool_layer(convB32, indiceB3))

        deconvB31 = F.relu(self.bn_layer(self.deconv_layer(unpoolB3)))

        concat_poolB2_deconv31 = torch.cat((poolB2, deconvB31), dim=1)

        convB22 = F.relu(self.bn_layer(
            self.conv_concat(concat_poolB2_deconv31)))

        unpoolB2 = F.relu(self.unpool_layer(convB22, indiceB2))

        deconvB21 = F.relu(self.bn_layer(self.deconv_layer(unpoolB2)))

        concat_poolB1_deconvb21 = torch.cat(
            (poolB1, deconvB21), dim=1)

        convB12 = F.relu(self.bn_layer(
            self.conv_concat(concat_poolB1_deconvb21)))

        unpoolB1 = F.relu(self.unpool_layer(convB12, indiceB1))

        deconvB11 = F.relu(self.bn_layer(self.deconv_layer(unpoolB1)))

        concat_pool1_deconvb11 = torch.cat((pool1, deconvB11), dim=1)

        conv21 = F.relu(self.bn_layer(
            self.conv_concat(concat_pool1_deconvb11)))

        conv22 = F.relu(self.bn_layer(self.conv_middle(conv21)))
        conv23 = F.relu(self.bn_layer(self.conv_middle(conv22)))
        conv24 = F.relu(self.bn_layer(self.conv_middle(conv23)))
        conv25 = F.relu(self.bn_layer(self.conv_middle(conv24)))
        conv26 = F.relu(self.bn_layer(self.conv_middle(conv25)))
        conv27 = F.relu(self.bn_layer(self.conv_middle(conv26)))
        conv28 = F.relu(self.bn_layer(self.conv_middle(conv27)))
        conv29 = F.relu(self.bn_layer(self.conv_middle(conv28)))

        unpool1 = F.relu(self.unpool_layer(conv29, indice1))

        deconv1 = F.relu(self.deconv_output(unpool1))

        return deconv1
