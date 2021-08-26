import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error

import logging


logger = logging.getLogger(__name__)
torch.set_printoptions(edgeitems=25)


'''
this Eqmdel inherit the the LightningModule.
the constructor takes the argumets are input_channels, learning rate and lose_type
At first we created the model in the constructor by using nn.Linear function.
To know about the LightningModule see the documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
'''


class EqModel(pl.LightningModule):

    def __init__(self, channels=3, lr=3e-4, *args, **kwargs):
        super().__init__()
        self.lr = lr
        # self.loss_type = loss_type
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

        if(self.current_epoch == 9):

            x1_pred = out_x1.flatten()
            x1_gt = x1.flatten()
            x1_diff = x1_gt-x1_pred
            self.print(x1_pred, "x1_pred")
            self.print(x1_gt, "x1_gt")
            self.print(x1_diff, "x1_diff")

        # recon_y = (a * out_x1) + (b * out_x2)
        recon_y = (out_x1-a)**2 + b*(out_x2-out_x1**2)**2

        recon_y = torch.reshape(recon_y, [B, N, X])

        y_loss = F.mse_loss(recon_y, y_input)
        x_loss = F.mse_loss(out, ref_out)
        x1_loss = F.mse_loss(out_x1, x1)
        x2_loss = F.mse_loss(out_x2, x2)
        ri_mse = torch.square(torch.sin(y_input) - torch.sin(recon_y)) + \
            torch.square(torch.cos(y_input) - torch.cos(recon_y))
        ri_mse = ri_mse.mean()
        avg_x1_error = torch.mean(torch.abs(x1-out_x1))
        cosineSimilrity = torchmetrics.CosineSimilarity()
        cosineSimilrity = cosineSimilrity(out, ref_out)
        explained_variance = torchmetrics.ExplainedVariance()
        explained_variance = explained_variance(out, ref_out)
        # mean_absolute_error = torchmetrics.MeanAbsoluteError()
        mean_absolute_error1 = mean_absolute_error(out, ref_out)
        # mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
        mean_abs_percentage_error = mean_absolute_percentage_error(
            out, ref_out)
        # mean_squared_error = torchmetrics.MeanSquaredError()
        mean_squared_error1 = mean_squared_error(out, ref_out)
        # mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        # mean_squared_log_error1 = mean_squared_log_error(out, ref_out)
        # pearson = torchmetrics.PearsonCorrcoef()
        # pearson = pearson(out, ref_out)
        # r2score = torchmetrics.R2Score()
        # r2score = r2score(out, ref_out)
        # spearman = torchmetrics.SpearmanCorrcoef()
        # spearman = spearman(out, ref_out)
        # smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        # smape = smape(out, ref_out)

        # if(self.loss_type == 'y_loss'):
        #     loss = F.mse_loss(recon_y, y_input)
        # elif (self.loss_type == 'x_loss'):
        #     loss = F.mse_loss(out, ref_out)
        # elif (self.loss_type == 'ri-mse'):
        #     loss = torch.square(
        #         torch.sin(y_input) -
        #         torch.sin(recon_y)) + torch.square(
        #             torch.cos(y_input) - torch.cos(recon_y))
        #     loss = loss.mean()
        # elif(self.loss_type == "avg_x1_error"):
        #     dif = torch.abs(x1-out_x1)
        #     loss = torch.mean(dif)
        # elif(self.loss_type == "cosineSimilrity"):
        #     dif1 = torchmetrics.CosineSimilarity()
        #     loss = dif1(out, ref_out)

        self.log('y_loss', y_loss, prog_bar=True)
        self.log('x_loss', x_loss, prog_bar=True)
        self.log('x1_loss', x1_loss, prog_bar=True)
        self.log('x2_loss', x2_loss, prog_bar=True)
        self.log('ri_mse', ri_mse, prog_bar=True)
        self.log('avg_x1_error', avg_x1_error, prog_bar=True)
        self.log('cosineSimilrity', cosineSimilrity, prog_bar=True)
        self.log('explained_variance', explained_variance, prog_bar=True)
        self.log('mean_absolute_error', mean_absolute_error1, prog_bar=True)
        self.log('mean_abs_percentage_error',
                 mean_abs_percentage_error, prog_bar=True)
        self.log('mean_squared_error', mean_squared_error1, prog_bar=True)
        # self.log('mean_squared_log_error',
        #          mean_squared_log_error1, prog_bar=True)

        # self.print(x1_pred, "x1_pred")
        # self.print(x1_gt, "x1_gt")
        # self.print(x1_diff, "x1_diff")

        # self.log('pearson', pearson, prog_bar=True)
        # self.log('r2score', r2score, prog_bar=True)
        # self.log('spearman', spearman, prog_bar=True)
        # self.log('smape', smape, prog_bar=True)
        # self.log('ri_mse', ri_mse, prog_bar=True)

        return {
            "loss": y_loss,
            "x_loss": x_loss,
            "x1_loss": x1_loss,
            "x2_loss": x2_loss,
            "ri_mse": ri_mse,
            "avg_x1_error": avg_x1_error,
            "cosineSimilrity": cosineSimilrity,
            "explained_variance": explained_variance,
            "mean_absolute_error": mean_absolute_error1,
            "mean_abs_percentage_error": mean_abs_percentage_error,
            "mean_squared_error": mean_squared_error1,
            # "mean_squared_log_error": mean_squared_log_error1,

            # "pearson": pearson,
            # "r2score": r2score,
            # "spearman": spearman,
            # "smape": smape
        }

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
        # recon_y = (a * out_x1) + (b * out_x2)
        recon_y = (out_x1-a)**2 + b*(out_x2-out_x1**2)**2

        recon_y = torch.reshape(recon_y, [B, N, X])

        y_loss_val = F.mse_loss(recon_y, y_input)
        x_loss_val = F.mse_loss(out, ref_out)
        ri_mse = torch.square(torch.sin(y_input) - torch.sin(recon_y)) + \
            torch.square(torch.cos(y_input) - torch.cos(recon_y))
        ri_mse_val = ri_mse.mean()
        avg_x1_error_val = torch.mean(torch.abs(x1-out_x1))
        cosineSimilrity = torchmetrics.CosineSimilarity()
        cosineSimilrity_val = cosineSimilrity(out, ref_out)
        explained_variance = torchmetrics.ExplainedVariance()
        explained_variance_val = explained_variance(out, ref_out)
        # mean_absolute_error = torchmetrics.MeanAbsoluteError()
        mean_absolute_error_val = mean_absolute_error(out, ref_out)
        # mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
        mean_abs_percentage_error_val = mean_absolute_percentage_error(
            out, ref_out)
        # mean_squared_error = torchmetrics.MeanSquaredError()
        mean_squared_error_val = mean_squared_error(out, ref_out)
        # mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        # mean_squared_log_error_val = mean_squared_log_error(out, ref_out)
        # pearson = torchmetrics.PearsonCorrcoef()
        # pearson_val = pearson(out, ref_out)
        # r2score = torchmetrics.R2Score()
        # r2score_val = r2score(out, ref_out)
        # spearman = torchmetrics.SpearmanCorrcoef()
        # spearman_val = spearman(out, ref_out)
        # smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        # smape_val = smape(out, ref_out)

        # if(self.loss_type == 'y_loss'):
        #     loss = F.mse_loss(recon_y, y_input)
        # elif (self.loss_type == 'x_loss'):
        #     loss = F.mse_loss(out, ref_out)
        # elif (self.loss_type == 'ri-mse'):
        #     loss = torch.square(
        #         torch.sin(y_input) -
        #         torch.sin(recon_y)) + torch.square(
        #         torch.cos(y_input) - torch.cos(recon_y))
        #     loss = loss.mean()
        # elif(self.loss_type == "avg_x1_error"):
        #     dif = torch.abs(x1-out_x1)
        #     loss = torch.mean(dif)
        # elif(self.loss_type == "cosineSimilrity"):
        #     dif1 = torchmetrics.CosineSimilarity()
        #     loss = dif1(out, ref_out)

        # # self.train()

        # self.log('val_loss', loss, prog_bar=True)

        self.log('y_loss_val', y_loss_val, prog_bar=True)
        self.log('x_loss_val', x_loss_val, prog_bar=True)
        self.log('ri_mse_val', ri_mse_val, prog_bar=True)
        self.log('avg_x1_error_val', avg_x1_error_val, prog_bar=True)
        self.log('cosineSimilrity_val', cosineSimilrity_val, prog_bar=True)
        self.log('explained_variance_val',
                 explained_variance_val, prog_bar=True)
        self.log('mean_absolute_error_val',
                 mean_absolute_error_val, prog_bar=True)
        self.log('mean_abs_percentage_error_val',
                 mean_abs_percentage_error_val, prog_bar=True)
        self.log('mean_squared_error_val',
                 mean_squared_error_val, prog_bar=True)
        # self.log('mean_squared_log_error_val',
        #          mean_squared_log_error_val, prog_bar=True)
        # self.log('pearson_val', pearson_val, prog_bar=True)
        # self.log('r2score_val', r2score_val, prog_bar=True)
        # self.log('spearman_val', spearman_val, prog_bar=True)
        # self.log('smape_val', smape_val, prog_bar=True)

        # self.log('ri_mse', ri_mse, prog_bar=True)

        return {
            "loss": y_loss_val, "x_loss_val": x_loss_val,
            "ri_mse_val": ri_mse_val, "avg_x1_error_val": avg_x1_error_val,
            "cosineSimilrity_val": cosineSimilrity_val, "explained_variance_val": explained_variance_val,
            "mean_absolute_error_val": mean_absolute_error_val,
            "mean_abs_percentage_error_val": mean_abs_percentage_error_val,
            "mean_squared_error_val": mean_squared_error_val,
            # "mean_squared_log_error_val": mean_squared_log_error_val,
            # "pearson_val": pearson_val,
            # "r2score_val": r2score_val,
            # "spearman_val": spearman_val,
            # "smape_val": smape_val

        }

        # return loss

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
        # recon_y = (a * out_x1) + (b * out_x2)
        recon_y = (out_x1-a)**2 + b*(out_x2-out_x1**2)**2

        recon_y = torch.reshape(recon_y, [B, N, X])

        y_loss_test = F.mse_loss(recon_y, y_input)
        x_loss_test = F.mse_loss(out, ref_out)
        ri_mse = torch.square(torch.sin(y_input) - torch.sin(recon_y)) + \
            torch.square(torch.cos(y_input) - torch.cos(recon_y))
        ri_mse_test = ri_mse.mean()
        avg_x1_error_test = torch.mean(torch.abs(x1-out_x1))
        cosineSimilrity = torchmetrics.CosineSimilarity()
        cosineSimilrity_test = cosineSimilrity(out, ref_out)
        explained_variance = torchmetrics.ExplainedVariance()
        explained_variance_test = explained_variance(out, ref_out)
        # mean_absolute_error = torchmetrics.MeanAbsoluteError()
        mean_absolute_error_test = mean_absolute_error(out, ref_out)
        # mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
        mean_abs_percentage_error_test = mean_absolute_percentage_error(
            out, ref_out)
        # mean_squared_error = torchmetrics.MeanSquaredError()
        mean_squared_error_test = mean_squared_error(out, ref_out)
        # mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        # mean_squared_log_error_test = mean_squared_log_error(out, ref_out)
        # pearson = torchmetrics.PearsonCorrcoef()
        # pearson_test = pearson(out, ref_out)
        # r2score = torchmetrics.R2Score()
        # r2score_test = r2score(out, ref_out)
        # spearman = torchmetrics.SpearmanCorrcoef()
        # spearman_test = spearman(out, ref_out)
        # smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        # smape_test = smape(out, ref_out)

        # if(self.loss_type == 'y_loss'):
        #     loss = F.mse_loss(recon_y, y_input)
        # elif (self.loss_type == 'x_loss'):
        #     loss = F.mse_loss(out, ref_out)
        # elif (self.loss_type == 'ri-mse'):
        #     loss = torch.square(
        #         torch.sin(y_input) -
        #         torch.sin(recon_y)) + torch.square(
        #         torch.cos(y_input) - torch.cos(recon_y))
        #     loss = loss.mean()
        # elif(self.loss_type == "avg_x1_error"):
        #     dif = torch.abs(x1-out_x1)
        #     loss = torch.mean(dif)
        # elif(self.loss_type == "cosineSimilrity"):
        #     dif1 = torchmetrics.CosineSimilarity()
        #     loss = dif1(out, ref_out)

        # self.log(name='test_loss', value=loss, prog_bar=True)

        self.log('y_loss_test', y_loss_test, prog_bar=True)
        self.log('x_loss_test', x_loss_test, prog_bar=True)
        self.log('ri_mse_test', ri_mse_test, prog_bar=True)
        self.log('avg_x1_error_test', avg_x1_error_test, prog_bar=True)
        self.log('cosineSimilrity_test', cosineSimilrity_test, prog_bar=True)
        self.log('explained_variance_test',
                 explained_variance_test, prog_bar=True)
        self.log('mean_absolute_error_test',
                 mean_absolute_error_test, prog_bar=True)
        self.log('mean_abs_percentage_error_test',
                 mean_abs_percentage_error_test, prog_bar=True)
        self.log('mean_squared_error_test',
                 mean_squared_error_test, prog_bar=True)
        # self.log('mean_squared_log_error_test',
        #          mean_squared_log_error_test, prog_bar=True)
        # self.log('pearson_test', pearson_test, prog_bar=True)
        # self.log('r2score_test', r2score_test, prog_bar=True)
        # self.log('spearman_test', spearman_test, prog_bar=True)
        # self.log('smape_test', smape_test, prog_bar=True)
        # self.log('ri_mse', ri_mse, prog_bar=True)

        return {
            "loss": y_loss_test, "x_loss_test": x_loss_test,
            "ri_mse_test": ri_mse_test, "avg_x1_error_test": avg_x1_error_test,
            "cosineSimilrity_test": cosineSimilrity_test, "explained_variance_test": explained_variance_test,
            "mean_absolute_error_test": mean_absolute_error_test,
            "mean_abs_percentage_error_test": mean_abs_percentage_error_test,
            "mean_squared_error_test": mean_squared_error_test,
            # "mean_squared_log_error_test": mean_squared_log_error_test,
            # "pearson_test": pearson_test,
            # "r2score_test": r2score_test,
            # "spearman_test": spearman_test,
            # 'smape_test': smape_test

        }
        # return loss

    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
    #     return self(4)


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

        # a, b = 1, 2 (for experiment 1 & 2)
        a, b = 1, 100  # (for experiment rosenbrock)

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

        # if(self.current_epoch == 49999):
        #     x1_pred = out_x1.flatten()
        #     x1_gt = x1.flatten()
        #     x1_diff = x1_gt-x1_pred
        #     self.print(x1_pred, "x1_pred")
        #     self.print(x1_gt, "x1_gt")
        #     self.print(x1_diff, "x1_diff")

        # a = torch.reshape(a, [B, N, X])
        # b = torch.reshape(b, [B, N, X])
        # calculate y by using predicted x1 and x2
        # recon_y = (a * out_x1) + (b * out_x2)
        recon_y = (out_x1-a)**2 + b*(out_x2-out_x1**2)**2

        recon_y = torch.reshape(recon_y, [B, N, X])

        y_loss = F.mse_loss(recon_y, y_input)
        x_loss = F.mse_loss(out, ref_out)
        x1_loss = F.mse_loss(out_x1, x1)
        x2_loss = F.mse_loss(out_x2, x2)
        ri_mse = torch.square(torch.sin(y_input) - torch.sin(recon_y)) + \
            torch.square(torch.cos(y_input) - torch.cos(recon_y))
        ri_mse = ri_mse.mean()
        avg_x1_error = torch.mean(torch.abs(x1-out_x1))
        cosineSimilrity = torchmetrics.CosineSimilarity()
        # cosineSimilrity = cosineSimilrity(out, ref_out)
        cosineSimilrity = cosineSimilrity(recon_y, y_input)

        explained_variance = torchmetrics.ExplainedVariance()
        explained_variance = explained_variance(out, ref_out)
        # mean_absolute_error = torchmetrics.MeanAbsoluteError()
        mean_absolute_error1 = mean_absolute_error(out, ref_out)
        # mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
        mean_abs_percentage_error = mean_absolute_percentage_error(
            out, ref_out)
        # mean_squared_error = torchmetrics.MeanSquaredError()
        mean_squared_error1 = mean_squared_error(out, ref_out)
        # mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        # mean_squared_log_error1 = mean_squared_log_error(out, ref_out)
        # pearson = torchmetrics.PearsonCorrcoef()
        # pearson = pearson(out, ref_out)
        # r2score = torchmetrics.R2Score()
        # r2score = r2score(out, ref_out)
        # spearman = torchmetrics.SpearmanCorrcoef()
        # spearman = spearman(out, ref_out)
        # smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        # smape = smape(out, ref_out)

        # if(self.loss_type == 'y_loss'):
        #     loss = F.mse_loss(recon_y, y_input)
        # elif (self.loss_type == 'x_loss'):
        #     loss = F.mse_loss(out, ref_out)
        # elif (self.loss_type == 'ri-mse'):
        #     loss = torch.square(
        #         torch.sin(y_input) -
        #         torch.sin(recon_y)) + torch.square(
        #             torch.cos(y_input) - torch.cos(recon_y))
        #     loss = loss.mean()
        # elif(self.loss_type == "avg_x1_error"):
        #     dif = torch.abs(x1-out_x1)
        #     loss = torch.mean(dif)
        # elif(self.loss_type == "cosineSimilrity"):
        #     dif1 = torchmetrics.CosineSimilarity()
        #     loss = dif1(out, ref_out)

        self.log('y_loss', y_loss, prog_bar=True)
        self.log('x_loss', x_loss, prog_bar=True)
        self.log('x1_loss', x1_loss, prog_bar=True)
        self.log('x2_loss', x2_loss, prog_bar=True)
        self.log('ri_mse', ri_mse, prog_bar=True)
        self.log('avg_x1_error', avg_x1_error, prog_bar=True)
        self.log('cosineSimilrity', cosineSimilrity, prog_bar=True)
        self.log('explained_variance', explained_variance, prog_bar=True)
        self.log('mean_absolute_error', mean_absolute_error1, prog_bar=True)
        self.log('mean_abs_percentage_error',
                 mean_abs_percentage_error, prog_bar=True)
        self.log('mean_squared_error', mean_squared_error1, prog_bar=True)
        # self.log('mean_squared_log_error',
        #          mean_squared_log_error1, prog_bar=True)
        # self.log('out_x1',
        #          out_x1, prog_bar=True)
        # self.log('pearson', pearson, prog_bar=True)
        # self.log('r2score', r2score, prog_bar=True)
        # self.log('spearman', spearman, prog_bar=True)
        # self.log('smape', smape, prog_bar=True)
        # self.log('ri_mse', ri_mse, prog_bar=True)

        return {
            "loss": y_loss,
            "x_loss": x_loss,
            "x1_loss": x1_loss,
            "x2_loss": x2_loss,
            "ri_mse": ri_mse,
            "avg_x1_error": avg_x1_error,
            "cosineSimilrity": cosineSimilrity,
            "explained_variance": explained_variance,
            "mean_absolute_error": mean_absolute_error1,
            "mean_abs_percentage_error": mean_abs_percentage_error,
            "mean_squared_error": mean_squared_error1,
            # "mean_squared_log_error": mean_squared_log_error1,
            # "out_X1": out_x1,
            # "pearson": pearson,
            # "r2score": r2score,
            # "spearman": spearman,
            # "smape": smape
        }

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
        # a, b = 1, 2
        a, b = 1, 100

        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        # calculate y by using predicted x1 and x2
        # recon_y = (a * out_x1) + (b * out_x2)
        recon_y = (out_x1-a)**2 + b*(out_x2-out_x1**2)**2

        recon_y = torch.reshape(recon_y, [B, N, X])

        y_loss_val = F.mse_loss(recon_y, y_input)
        x_loss_val = F.mse_loss(out, ref_out)
        ri_mse = torch.square(torch.sin(y_input) - torch.sin(recon_y)) + \
            torch.square(torch.cos(y_input) - torch.cos(recon_y))
        ri_mse_val = ri_mse.mean()
        avg_x1_error_val = torch.mean(torch.abs(x1-out_x1))
        cosineSimilrity = torchmetrics.CosineSimilarity()
        cosineSimilrity_val = cosineSimilrity(out, ref_out)
        explained_variance = torchmetrics.ExplainedVariance()
        explained_variance_val = explained_variance(out, ref_out)
        # mean_absolute_error = torchmetrics.MeanAbsoluteError()
        mean_absolute_error_val = mean_absolute_error(out, ref_out)
        # mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
        mean_abs_percentage_error_val = mean_absolute_percentage_error(
            out, ref_out)
        # mean_squared_error = torchmetrics.MeanSquaredError()
        mean_squared_error_val = mean_squared_error(out, ref_out)
        # mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        # mean_squared_log_error_val = mean_squared_log_error(out, ref_out)
        # pearson = torchmetrics.PearsonCorrcoef()
        # pearson_val = pearson(out, ref_out)
        # r2score = torchmetrics.R2Score()
        # r2score_val = r2score(out, ref_out)
        # spearman = torchmetrics.SpearmanCorrcoef()
        # spearman_val = spearman(out, ref_out)
        # smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        # smape_val = smape(out, ref_out)

        # if(self.loss_type == 'y_loss'):
        #     loss = F.mse_loss(recon_y, y_input)
        # elif (self.loss_type == 'x_loss'):
        #     loss = F.mse_loss(out, ref_out)
        # elif (self.loss_type == 'ri-mse'):
        #     loss = torch.square(
        #         torch.sin(y_input) -
        #         torch.sin(recon_y)) + torch.square(
        #         torch.cos(y_input) - torch.cos(recon_y))
        #     loss = loss.mean()
        # elif(self.loss_type == "avg_x1_error"):
        #     dif = torch.abs(x1-out_x1)
        #     loss = torch.mean(dif)
        # elif(self.loss_type == "cosineSimilrity"):
        #     dif1 = torchmetrics.CosineSimilarity()
        #     loss = dif1(out, ref_out)

        # # self.train()

        # self.log('val_loss', loss, prog_bar=True)

        self.log('y_loss_val', y_loss_val, prog_bar=True)
        self.log('x_loss_val', x_loss_val, prog_bar=True)
        self.log('ri_mse_val', ri_mse_val, prog_bar=True)
        self.log('avg_x1_error_val', avg_x1_error_val, prog_bar=True)
        self.log('cosineSimilrity_val', cosineSimilrity_val, prog_bar=True)
        self.log('explained_variance_val',
                 explained_variance_val, prog_bar=True)
        self.log('mean_absolute_error_val',
                 mean_absolute_error_val, prog_bar=True)
        self.log('mean_abs_percentage_error_val',
                 mean_abs_percentage_error_val, prog_bar=True)
        self.log('mean_squared_error_val',
                 mean_squared_error_val, prog_bar=True)
        # self.log('mean_squared_log_error_val',
        #          mean_squared_log_error_val, prog_bar=True)
        # self.log('pearson_val', pearson_val, prog_bar=True)
        # self.log('r2score_val', r2score_val, prog_bar=True)
        # self.log('spearman_val', spearman_val, prog_bar=True)
        # self.log('smape_val', smape_val, prog_bar=True)

        # self.log('ri_mse', ri_mse, prog_bar=True)

        return {
            "loss": y_loss_val, "x_loss_val": x_loss_val,
            "ri_mse_val": ri_mse_val, "avg_x1_error_val": avg_x1_error_val,
            "cosineSimilrity_val": cosineSimilrity_val, "explained_variance_val": explained_variance_val,
            "mean_absolute_error_val": mean_absolute_error_val,
            "mean_abs_percentage_error_val": mean_abs_percentage_error_val,
            "mean_squared_error_val": mean_squared_error_val,
            # "mean_squared_log_error_val": mean_squared_log_error_val,
            # "pearson_val": pearson_val,
            # "r2score_val": r2score_val,
            # "spearman_val": spearman_val,
            # "smape_val": smape_val

        }

        # return loss

    def test_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        # a = batch["a"]
        # b = batch["b"]
        # a, b = 1, 2
        a, b = 1, 100

        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0].unsqueeze(2)
        out_x2 = out[:, :, 1].unsqueeze(2)

        x1_pred = out_x1.flatten()
        x1_gt = x1.flatten()
        x1_diff = x1_gt-x1_pred
        self.print(x1_pred, "x1_pred")
        self.print(x1_gt, "x1_gt")
        self.print(x1_diff, "x1_diff")

        # calculate y by using predicted x1 and x2
        # recon_y = (a * out_x1) + (b * out_x2)
        recon_y = (out_x1-a)**2 + b*(out_x2-out_x1**2)**2

        recon_y = torch.reshape(recon_y, [B, N, X])

        y_loss_test = F.mse_loss(recon_y, y_input)
        x_loss_test = F.mse_loss(out, ref_out)
        ri_mse = torch.square(torch.sin(y_input) - torch.sin(recon_y)) + \
            torch.square(torch.cos(y_input) - torch.cos(recon_y))
        ri_mse_test = ri_mse.mean()
        avg_x1_error_test = torch.mean(torch.abs(x1-out_x1))
        cosineSimilrity = torchmetrics.CosineSimilarity()
        cosineSimilrity_test = cosineSimilrity(out, ref_out)
        explained_variance = torchmetrics.ExplainedVariance()
        explained_variance_test = explained_variance(out, ref_out)
        # mean_absolute_error = torchmetrics.MeanAbsoluteError()
        mean_absolute_error_test = mean_absolute_error(out, ref_out)
        # mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError()
        mean_abs_percentage_error_test = mean_absolute_percentage_error(
            out, ref_out)
        # mean_squared_error = torchmetrics.MeanSquaredError()
        mean_squared_error_test = mean_squared_error(out, ref_out)
        # mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        # mean_squared_log_error_test = mean_squared_log_error(out, ref_out)
        # pearson = torchmetrics.PearsonCorrcoef()
        # pearson_test = pearson(out, ref_out)
        # r2score = torchmetrics.R2Score()
        # r2score_test = r2score(out, ref_out)
        # spearman = torchmetrics.SpearmanCorrcoef()
        # spearman_test = spearman(out, ref_out)
        # smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        # smape_test = smape(out, ref_out)

        # if(self.loss_type == 'y_loss'):
        #     loss = F.mse_loss(recon_y, y_input)
        # elif (self.loss_type == 'x_loss'):
        #     loss = F.mse_loss(out, ref_out)
        # elif (self.loss_type == 'ri-mse'):
        #     loss = torch.square(
        #         torch.sin(y_input) -
        #         torch.sin(recon_y)) + torch.square(
        #         torch.cos(y_input) - torch.cos(recon_y))
        #     loss = loss.mean()
        # elif(self.loss_type == "avg_x1_error"):
        #     dif = torch.abs(x1-out_x1)
        #     loss = torch.mean(dif)
        # elif(self.loss_type == "cosineSimilrity"):
        #     dif1 = torchmetrics.CosineSimilarity()
        #     loss = dif1(out, ref_out)

        # self.log(name='test_loss', value=loss, prog_bar=True)

        self.log('y_loss_test', y_loss_test, prog_bar=True)
        self.log('x_loss_test', x_loss_test, prog_bar=True)
        self.log('ri_mse_test', ri_mse_test, prog_bar=True)
        self.log('avg_x1_error_test', avg_x1_error_test, prog_bar=True)
        self.log('cosineSimilrity_test', cosineSimilrity_test, prog_bar=True)
        self.log('explained_variance_test',
                 explained_variance_test, prog_bar=True)
        self.log('mean_absolute_error_test',
                 mean_absolute_error_test, prog_bar=True)
        self.log('mean_abs_percentage_error_test',
                 mean_abs_percentage_error_test, prog_bar=True)
        self.log('mean_squared_error_test',
                 mean_squared_error_test, prog_bar=True)
        # self.log('mean_squared_log_error_test',
        #          mean_squared_log_error_test, prog_bar=True)
        # self.log('pearson_test', pearson_test, prog_bar=True)
        # self.log('r2score_test', r2score_test, prog_bar=True)
        # self.log('spearman_test', spearman_test, prog_bar=True)
        # self.log('smape_test', smape_test, prog_bar=True)
        # self.log('ri_mse', ri_mse, prog_bar=True)

        return {
            "loss": y_loss_test, "x_loss_test": x_loss_test,
            "ri_mse_test": ri_mse_test, "avg_x1_error_test": avg_x1_error_test,
            "cosineSimilrity_test": cosineSimilrity_test, "explained_variance_test": explained_variance_test,
            "mean_absolute_error_test": mean_absolute_error_test,
            "mean_abs_percentage_error_test": mean_abs_percentage_error_test,
            "mean_squared_error_test": mean_squared_error_test,
            # "mean_squared_log_error_test": mean_squared_log_error_test,
            # "pearson_test": pearson_test,
            # "r2score_test": r2score_test,
            # "spearman_test": spearman_test,
            # 'smape_test': smape_test

        }
        # return loss
