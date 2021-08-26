import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split


import logging

logger = logging.getLogger(__name__)


"""
input_y function takes argument stack_size. 
return the 3 tensor y, x1, x2
this function creates random values from the x1 and x2
here, the eqn is y = a*x1 + b*x2
a and b are the two constants which value are 2 and 3 respectively
x1 and x2 also created randomly with the tensor size (stack_size,1)
and the shape of the y is also (stack_size)
N.B.: this function is used to create training random data
"""


def input_y(stack_size, low_limit, high_limit, experiment):
    x1 = x2 = torch.from_numpy(
        np.random.uniform(low_limit, high_limit, (stack_size, 1)).astype(np.float32))
    y = np.zeros([stack_size, 1])
    if (experiment == 1):
        a, b = 1, 2
        for i in range(stack_size):
            y[i] = (a * x1[i]) + (b * x2[i])
        y = torch.Tensor(y)
        return y, x1, x2

    elif (experiment == 2):
        a, b = 1, 2
        for i in range(stack_size):
            y[i] = (a * x1[i]) + (b * x2[i])
        y = torch.Tensor(y)
        y = np.angle(np.exp(1j * y))
        return y, x1, x2

    elif (experiment == 3):
        a = b = torch.from_numpy(
            np.random.uniform(low_limit, high_limit, (stack_size, 1)).astype(np.float32))
        for i in range(stack_size):
            y[i] = (a[i] * x1[i]) + (b[i] * x2[i])
        y = torch.Tensor(y)
        y = np.angle(np.exp(1j * y))
        return y, x1, x2, a, b
    elif (experiment == "rosenbrock"):
        a, b = 1, 100
        for i in range(stack_size):
            y[i] = (x1[i]-a)**2 + b*(x2[i]-x1[i]**2)**2
        y = torch.Tensor(y)
        return y, x1, x2


"""
input_y_test function takes argument stack_size. 
return the 3 tensor y, x1, x2
this function creates random values from the x1 and x2. To avoid the linearity problem we multiply 2 and another random torch with same shape.
here, the eqn is y = a*x1 + b*x2
a and b are the two constants which values are 4 and 5 respectively
x1 and x2 are also created randomly with the tensor size (stack_size,1)
and the shape of the y is also (stack_size)
N.B.: this function is used to create test random data 
"""


def input_y_test(stack_size, low_limit, high_limit, experiment):

    x1 = x2 = torch.from_numpy(np.random.uniform(
        low_limit, high_limit, (stack_size, 1)).astype(np.float32))
    # * \
    #     torch.from_numpy(
    #         np.random.uniform(low_limit, high_limit, (stack_size, 1)).astype(np.float32)) * 2

    y = np.zeros([stack_size, 1])
    if (experiment == 1):
        a, b = 1, 2
        for i in range(stack_size):
            y[i] = (a * x1[i]) + (b * x2[i])
            y = torch.Tensor(y)
        return y, x1, x2

    elif (experiment == 2):
        a, b = 1, 2
        for i in range(stack_size):
            y[i] = (a * x1[i]) + (b * x2[i])
            y = torch.Tensor(y)
            y = np.angle(np.exp(1j * y))
        return y, x1, x2

    elif (experiment == 3):
        a = b = torch.from_numpy(np.random.uniform(low_limit, high_limit, (stack_size, 1)).astype(
            np.float32))
        # * torch.from_numpy(np.random.uniform(low_limit, high_limit, (stack_size, 1)).astype(np.float32)) * 2

        for i in range(stack_size):
            y[i] = (a[i] * x1[i]) + (b[i] * x2[i])
        y = torch.Tensor(y)
        y = np.angle(np.exp(1j * y))
        return y, x1, x2, a, b
    elif (experiment == "rosenbrock"):
        a, b = 1, 100
        for i in range(stack_size):
            y[i] = (x1[i]-a)**2 + b*(x2[i]-x1[i]**2)**2
        y = torch.Tensor(y)
        return y, x1, x2


# def wrap_y(y):
#     wrapY = np.angle(np.exp(1j * y))
#     return wrapY


'''
The EqnPrepare class which inherited the Dataset abstract class to create the sample data.
The len function returns the length of the dataset and the __getitem__ function returns the
supporting to fetch a data sample for a given key.
In the __init__ function we use the input_y function to create the randomm data for training.
And the __getitem__ function returns the dictionary of x1,x2 and y. 
'''


class EqnPrepare(Dataset):
    def __init__(self, stack_size=500, low_limit=-100, high_limit=100, experiment=3):

        self.stack_size = stack_size
        # self.wrap = wrap
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.experiment = experiment

        # if(self.wrap):
        #     self.y_input, self.x1, self.x2, self.a, self.b = input_y(
        #         self.stack_size, self.low_limit, self.high_limit)
        #     self.y_input = wrap_y(self.y_input)
        # else:
        #     self.y_input, self.x1, self.x2, self.a, self.b = input_y(
        #         self.stack_size, self.low_limit, self.high_limit)

    def __len__(self):
        return self.stack_size

    def __getitem__(self, idx):

        if(self.experiment == 1):
            self.y_input, self.x1, self.x2 = input_y(
                self.stack_size, self.low_limit, self.high_limit, 1)
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input}

        elif (self.experiment == 2):
            self.y_input, self.x1, self.x2 = input_y(
                self.stack_size, self.low_limit, self.high_limit, 2)
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input}

        elif (self.experiment == 3):
            self.y_input, self.x1, self.x2, self.a, self.b = input_y(
                self.stack_size, self.low_limit, self.high_limit, 3)
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input,
                "a": self.a,
                "b": self.b
            }
        elif (self.experiment == "rosenbrock"):
            self.y_input, self.x1, self.x2 = input_y(
                self.stack_size, self.low_limit, self.high_limit, "rosenbrock")
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input}


'''
The EqnTestPrepare class which inherited the Dataset abstract class to create the sample data.
The len function returns the length of the dataset and the __getitem__ function returns the
supporting to fetch a data sample for a given key.
In the __init__ function we use the input_y_test function to create the randomm data for test. here we use the default stack_size 10000.
And the __getitem__ function returns the dictionary of x1,x2 and y. 
'''


class EqnTestPrepare(Dataset):
    def __init__(self, stack_size=500, low_limit=-100, high_limit=100, experiment=3):
        # self.wrap = wrap
        self.stack_size = stack_size
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.experiment = experiment

        # if(self.wrap):
        #     self.y_input, self.x1, self.x2, self.a, self.b = input_y_test(
        #         self.stack_size, self.low_limit, self.high_limit)
        #     self.y_input = wrap_y(self.y_input)
        # else:
        #     self.y_input, self.x1, self.x2, self.a, self.b = input_y_test(
        #         self.stack_size, self.low_limit, self.high_limit)

    def __len__(self):
        return self.stack_size

    def __getitem__(self, idx):

        if(self.experiment == 1):
            self.y_input, self.x1, self.x2 = input_y_test(
                self.stack_size, self.low_limit, self.high_limit, 1)
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input, }

        elif (self.experiment == 2):
            self.y_input, self.x1, self.x2 = input_y_test(
                self.stack_size, self.low_limit, self.high_limit, 2)
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input, }

        elif (self.experiment == 3):
            self.y_input, self.x1, self.x2, self.a, self.b = input_y_test(
                self.stack_size, self.low_limit, self.high_limit, 3)
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input,
                "a": self.a,
                "b": self.b
            }
        elif (self.experiment == "rosenbrock"):
            self.y_input, self.x1, self.x2 = input_y_test(
                self.stack_size, self.low_limit, self.high_limit, "rosenbrock")
            return {
                "x1": self.x1,
                "x2": self.x2,
                "y_input": self.y_input}


'''
The EqnDataLoader class which is inherited the LightningDataModule.
A DataModule standardizes the training, val, test splits, data preparation and transforms.
The main advantage is consistent data splits, data preparation and transforms across models.
In __init__ function we split the datset for the training and test. i.e. 80% for training and 20% for the vlidation.
 
Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset. And it tranforms the dataset in tensor.
To know more about the LightningDataModule see the documention : https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
'''


class EqnDataLoader(pl.LightningDataModule):
    def __init__(self, train=True,
                 experiment=3,
                 train_batch_size=4,
                 val_batch_size=4,
                 test_batch_size=4,
                 train_num_workers=4,
                 val_num_workers=4,
                 test_num_workers=4,
                 low_limit=-100,
                 high_limit=100,
                 stack_size=500,
                 * args,  **kwargs, ):
        super().__init__(*args, **kwargs)
        self.experiment = experiment
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.stack_size = stack_size

        if train:
            self.dataset = EqnPrepare(
                stack_size=self.stack_size, experiment=self.experiment, low_limit=self.low_limit, high_limit=self.high_limit)
            train_size = int(0.8*len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size])
        else:
            self.test_dataset = EqnTestPrepare(
                stack_size=self.stack_size, experiment=self.experiment, low_limit=self.low_limit, high_limit=self.high_limit)

    def train_dataloader(self):

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.train_batch_size,
                                      shuffle=False,
                                      num_workers=self.train_num_workers,
                                      )

        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=self.val_batch_size,
                                    shuffle=False,
                                    num_workers=self.val_num_workers)
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.test_batch_size,
                                     shuffle=False,
                                     num_workers=self.test_num_workers)
        return test_dataloader


if __name__ == "__main__":
    train_dataset = EqnPrepare(
        stack_size=500, experiment=1, low_limit=-100, high_limit=100)

    for batch_idx, batch in enumerate(train_dataset):

        ''' if we want to return not in dictionary we can check in this way '''

        # input_filt, coh, ddays, bperps, mr, he = batch
        # [B, N] = batch['x1'].shape

        # print(input_filt.shape)

        ''' if we want to return in dictionary we can check in this way '''

        print('Batch Index \t = {}'.format(batch_idx))
        # print('Input Type \t = {}'.format(batch['input'].dtype))
        # print('Input Shape \t = {}'.format(batch['input'].shape))
        # print('x1 Shape \t = {}'.format(batch['x1']))
        print('x2 Shape \t = {}'.format(batch['x2']))
        # print(batch['mr']) # to check the output of mr
        # print('a Shape \t = {}'.format(batch['a'].shape))
        # print('b Shape \t = {}'.format(batch['b'].shape))
        # print('y_input Shape \t = {}'.format(batch['y_input']))
        # print('conv1 shape \t = {}'.format(batch['conv1'].shape))
        # print('Wrap recon phase shape = {}'.format(
        #     batch['wrap_recon_phase'].shape))

        # print(np.angle(1*np.exp(batch['input'][0][0][0]) - (batch['wrap_recon_phase'][0][0][0])))

        break
