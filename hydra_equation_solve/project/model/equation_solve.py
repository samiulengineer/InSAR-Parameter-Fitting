import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

import logging


logger = logging.getLogger(__name__)


'''
this Eqmdel inherit the the LightningModule.
the constructor takes the argumets are input_channels, learning rate and lose_type
At first we created the model in the constructor by using nn.Linear function. 
To know about the LightningModule see the documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
'''


class EqModel(pl.LightningModule):

    def __init__(self, channels=3, lr=3e-4, loss_type='ri-mse', *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.loss_type = loss_type
        self.channels = channels

        self.L1 = nn.Linear(self.channels, 64)
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 16)
        self.L4 = nn.Linear(16, 8)
        self.L5 = nn.Linear(8, 4)
        self.L6 = nn.Linear(4, 2)

    def forward(self, y_input, a, b):
        y_input = torch.cat([y_input, a, b], dim=2)
        out = self.L1(y_input)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        out = self.L5(out)
        out = self.L6(out)

        return out

    def training_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        a = batch["a"]
        b = batch["b"]
        [B, N, X] = y_input.shape

        if(self.current_epoch == 1):
            first = torch.rand((1, 100, 1))
            second = torch.rand((1, 100, 1))
            third = torch.rand((1, 100, 1))

            self.logger.experiment.add_graph(EqModel(), [first, second, third])

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input, a, b)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        recon_y = (a * out_x1) + (b * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        elif (self.loss_type == 'ri-mse'):
            loss = torch.square(
                torch.sin(y_input) -
                torch.sin(recon_y)) + torch.square(
                    torch.cos(y_input) - torch.cos(recon_y))
            loss = loss.mean()

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # self.eval()
        # with torch.no_grad():
        y_input = batch["y_input"]  # [B, N, 1]
        x1 = batch["x1"]  # [B, N, 1]
        x2 = batch["x2"]  # [B, N, 1]
        a = batch["a"]
        b = batch["b"]
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input, a, b)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        # calculate y by using predicted x1 and x2
        recon_y = (a * out_x1) + (b * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        elif (self.loss_type == 'ri-mse'):
            loss = torch.square(
                torch.sin(y_input) -
                torch.sin(recon_y)) + torch.square(
                torch.cos(y_input) - torch.cos(recon_y))
            loss = loss.mean()

        # self.train()

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        a = batch["a"]
        b = batch["b"]
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input, a, b)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        # calculate y by using predicted x1 and x2
        recon_y = (a * out_x1) + (b * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        elif (self.loss_type == 'ri-mse'):
            loss = torch.square(
                torch.sin(y_input) -
                torch.sin(recon_y)) + torch.square(
                torch.cos(y_input) - torch.cos(recon_y))
            loss = loss.mean()

        self.log(name='test_loss', value=loss, prog_bar=True)
        return loss


class EqModel12(pl.LightningModule):

    def __init__(self, channels=1, lr=3e-4, loss_type='ri-mse', *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.loss_type = loss_type
        self.channels = channels

        self.L1 = nn.Linear(self.channels, 64)
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 16)
        self.L4 = nn.Linear(16, 8)
        self.L5 = nn.Linear(8, 4)
        self.L6 = nn.Linear(4, 2)

    # forward function returns the prediction. and also passed the data in the model
    def forward(self, y_input):
        # y_input = torch.cat([y_input, a, b], dim=2)
        out = self.L1(y_input)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        out = self.L5(out)
        out = self.L6(out)

        return out

    def training_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        # a = batch["a"]
        # b = batch["b"]
        a, b = 1, 2
        [B, N, X] = y_input.shape

        if(self.current_epoch == 1):
            first = torch.rand((1, 100, 1))
            second = torch.rand((1, 100, 1))
            third = torch.rand((1, 100, 1))

            self.logger.experiment.add_graph(EqModel12(), [first])

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)
        # a = torch.reshape(a, [B, N, X])
        # b = torch.reshape(b, [B, N, X])
        # calculate y by using predicted x1 and x2
        recon_y = (a * out_x1) + (b * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        elif (self.loss_type == 'ri-mse'):
            loss = torch.square(
                torch.sin(y_input) -
                torch.sin(recon_y)) + torch.square(
                    torch.cos(y_input) - torch.cos(recon_y))
            loss = loss.mean()

        self.log('train_loss', loss, prog_bar=True)
        # self.logger.experiment.add_scalar(
        #     "Loss/Train", loss, self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # self.eval()
        # with torch.no_grad():
        y_input = batch["y_input"]  # [B, N, 1]
        x1 = batch["x1"]  # [B, N, 1]
        x2 = batch["x2"]  # [B, N, 1]
        # a = batch["a"]
        # b = batch["b"]
        a, b = 1, 2
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        # calculate y by using predicted x1 and x2
        recon_y = (a * out_x1) + (b * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        elif (self.loss_type == 'ri-mse'):
            loss = torch.square(
                torch.sin(y_input) -
                torch.sin(recon_y)) + torch.square(
                torch.cos(y_input) - torch.cos(recon_y))
            loss = loss.mean()

        # self.train()

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        # a = batch["a"]
        # b = batch["b"]
        a, b = 1, 2
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        # calculate y by using predicted x1 and x2
        recon_y = (a * out_x1) + (b * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        elif (self.loss_type == 'ri-mse'):
            loss = torch.square(
                torch.sin(y_input) -
                torch.sin(recon_y)) + torch.square(
                torch.cos(y_input) - torch.cos(recon_y))
            loss = loss.mean()

        self.log(name='test_loss', value=loss, prog_bar=True)
        return loss
