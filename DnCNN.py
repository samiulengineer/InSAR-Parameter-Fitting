import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from DnCNN_Dataloader import NoisyDataset



''' Hyperparameter '''

# learning_rate = 1e-3
# hidden_dim = 128
# batch_size = 32


class DnCNN(pl.LightningModule):

    def __init__(self):
        super(DnCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3,  out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 64, out_channels = 3,  kernel_size = 3, padding = 1)

        self.bn1 = nn.BatchNorm2d(64, 64)
        self.bn2 = nn.BatchNorm2d(64, 64)
        self.bn3 = nn.BatchNorm2d(64, 64)
        self.bn4 = nn.BatchNorm2d(64, 64)
        self.bn5 = nn.BatchNorm2d(64, 64)
        self.bn6 = nn.BatchNorm2d(64, 64)

        self.dataset_dir = "/home/mdsamiul/InSAR-Parameter-Fitting/data/BSDS300/images"
        # self.train_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()


    def forward(self, x):  # forward propagation
        in_data = F.relu(self.conv1(x))
        in_data = F.relu(self.bn1(self.conv2(in_data)))
        in_data = F.relu(self.bn2(self.conv3(in_data)))
        in_data = F.relu(self.bn3(self.conv4(in_data)))
        in_data = F.relu(self.bn4(self.conv5(in_data)))
        in_data = F.relu(self.bn5(self.conv6(in_data)))
        in_data = F.relu(self.bn6(self.conv7(in_data)))
        residual = self.conv8(in_data)

        y = residual + x

        return y


    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        tensorboard_logs = {'train_loss': loss}
        # self.train_acc(out, y)
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return {'loss': loss, 'log': tensorboard_logs}


    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log('valid_loss', loss)


    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



    ''' dataset preparation '''

    # def setup(self, stage):

    #     dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    #     self.mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    #     self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(NoisyDataset(self.dataset_dir), batch_size=20)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size)

    def test_dataloader(self):
        return DataLoader(NoisyDataset("/home/mdsamiul/InSAR-Parameter-Fitting/data/BSDS300/images", mode='test', img_size=(320, 320)))


def main():

    # ------------
    # model
    # ------------

    model = DnCNN()

    logger = TensorBoardLogger( "lightning_logs", 
                                name = "DnCNN"
                                )



    # ------------
    # training
    # ------------

    trainer = pl.Trainer( max_epochs = 10,
                          fast_dev_run = True,
                          gpus = 2, 
                          logger = logger
                          )
    trainer.fit(model)



    # ------------
    # testing
    # ------------

    trainer.test(model)




if __name__ == '__main__':
    main()