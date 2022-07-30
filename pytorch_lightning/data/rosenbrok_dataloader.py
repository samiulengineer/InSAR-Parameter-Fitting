import numpy as np
from scipy.optimize import minimize
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
import pandas as pd


import logging

logger = logging.getLogger(__name__)


"""
input_y function takes argument stack_size, low_limit, high_limit and the experiments.
low_limit and high_limit used as the interval. for each experiment it needs different return so we handle the experiment here.
return the 5 tensors y, x1, x2, a and b according to the experiment.
this function creates random values from the x1 and x2 for all experiments and creates random constant value a and b for the experiment 3
here, the rosenbrock eqn is y = (x1-a)**2 + b*(x2-x1**2)**2
a and b are the two constants which value are 1 and 2 (for experiment 1 and experiment 2) respectively
x1 and x2 also created randomly with the tensor size (stack_size,1)
and the shape of the y is also (stack_size,1)
N.B.: this function is used to create training random data. for experiment 1 and 2 the constants value of a & b are 1 and 10
N.B.: to avoid the calculation error, the low limit and the high limit was used -1 and 1. which is handled by hydra
"""
# add link of rosenbrock equation


# def rosen(a, b, x):
#     """The Rosenbrock function"""
#     return (sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (b-x[:-1])**2.0))**2
#
#
# def input_y(stack_size, low_limit, high_limit):
#
#     x0 = np.array([10, 10])
#     data = np.zeros([stack_size, 4])
#     baseline_a = np.zeros([stack_size, 1]).astype(np.float32)
#     baseline_b = np.zeros([stack_size, 1]).astype(np.float32)
#     x1_orginal = np.zeros([stack_size, 1]).astype(np.float32)
#     x2_orginal = np.zeros([stack_size, 1]).astype(np.float32)
#
#     count = 0
#     while (count < stack_size):
#         a = np.random.uniform(low_limit, high_limit)
#         b = np.random.uniform(low_limit, high_limit)  # description of uniform
#
#         def f(x):
#             return rosen(a, b, x)
#         res = minimize(f, x0, method='nelder-mead',
#                        options={'xatol': 1e-8, 'disp': False})  # description
#         x1, x2 = res.x
#         data[count] = [a, b, x1, x2]  # [stack_size, 4]
#         baseline_a[count] = [a]  # [stack_size, 1]
#         baseline_b[count] = [b]  # [stack_size, 1]
#         x1_orginal[count] = [x1]  # [stack_size, 1]
#         x2_orginal[count] = [x2]  # [stack_size, 1]
#         count += 1
#
#     return baseline_a, baseline_b, x1_orginal, x2_orginal
#
#
# """
# input_y_test function takes argument stack_size, low_limit, high_limit and the experiments.
# low_limit and high_limit used as the interval. for each experiment it needs different return so we handle the experiment here.
# return the 5 tensors y, x1, x2, a and b according to the experiment.
# this function creates random values from the x1 and x2 for all experiments and creates random constant value a and b for the experiment 3
# here, the rosenbrock eqn is y = (x1-a)**2 + b*(x2-x1**2)**2
# a and b are the two constants which value are 1 and 2 (for experiment 1 and experiment 2) respectively
# x1 and x2 also created randomly with the tensor size (stack_size,1)
# and the shape of the y is also (stack_size,1)
# N.B.: this function is used to create training random data. for experiment 1 and 2 the constants value of a & b are 1 and 10
# N.B.: to avoid the calculation error, the low limit and the high limit was used -1 and 1. which is handled by hydra
# """
#
#
# def input_y_test(stack_size, low_limit, high_limit):
#
#     x0 = np.array([10, 10])
#     data = np.zeros([stack_size, 4])
#     baseline_a = np.zeros([stack_size, 1]).astype(np.float32)
#     baseline_b = np.zeros([stack_size, 1]).astype(np.float32)
#     x1_orginal = np.zeros([stack_size, 1]).astype(np.float32)
#     x2_orginal = np.zeros([stack_size, 1]).astype(np.float32)
#
#     count = 0
#     while (count < stack_size):
#         a = np.random.uniform(low_limit, high_limit)
#         b = np.random.uniform(low_limit, high_limit)  # description of uniform
#
#         def f(x):
#             return rosen(a, b, x)
#         res = minimize(f, x0, method='nelder-mead',
#                        options={'xatol': 1e-8, 'disp': False})  # description
#         x1, x2 = res.x
#         data[count] = [a, b, x1, x2]  # [stack_size, 4]
#         baseline_a[count] = [a]  # [stack_size, 1]
#         baseline_b[count] = [b]  # [stack_size, 1]
#         x1_orginal[count] = [x1]  # [stack_size, 1]
#         x2_orginal[count] = [x2]  # [stack_size, 1]
#         count += 1
#
#     return baseline_a, baseline_b, x1_orginal, x2_orginal


'''
The EqnPrepareRosbrock class which inherited the Dataset abstract class to create the sample data.
The len function returns the length of the dataset and the __getitem__ function returns the
supporting to fetch a data sample for a given key.
In the __init__ function we use the input_y function to create the randomm data for training.
And the __getitem__ function returns the dictionary of x1,x2, y, a and b according to the experiments which was given by the user. 
'''


class EqnPrepareRosbrock(Dataset):
    def __init__(self):
        self.data_frame = pd.read_csv(
            "/home/mdsamiul/InSAR-Coding/parameter_fitting_learning_based/hydra_equation_solve/project/data/train_data_10000.csv")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        self.a = torch.tensor(
            self.data_frame.iloc[:, 0].values, dtype=torch.float32)
        self.b = torch.tensor(
            self.data_frame.iloc[:, 1].values, dtype=torch.float32)
        self.x1 = torch.tensor(
            self.data_frame.iloc[:, 2].values, dtype=torch.float32)
        self.x2 = torch.tensor(
            self.data_frame.iloc[:, 3].values, dtype=torch.float32)

        return {
            "x1": self.x1.reshape((10000, 1)),
            "x2": self.x2.reshape((10000, 1)),
            "a": self.a.reshape((10000, 1)),
            "b": self.b.reshape((10000, 1))
        }


'''
The EqnTestPrepare class which inherited the Dataset abstract class to create the sample data.
The len function returns the length of the dataset and the __getitem__ function returns the
supporting to fetch a data sample for a given key.
In the __init__ function we use the input_y_test function to create the randomm data for test. here we use the default stack_size 10000.
And the __getitem__ function returns the dictionary of x1,x2, y, a and b according to the experiments which was given by the user. 
'''


# class EqnTestPrepareRosebrock(Dataset):
#     def __init__(self):
#
#
#
#     def __len__(self):
#         return self.stack_size
#
#     def __getitem__(self, idx):
#
#         return {
#             "x1": self.x1,
#             "x2": self.x2,
#             "a": self.a,
#             "b": self.b
#         }


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
                 train_batch_size=4,
                 val_batch_size=4,
                 test_batch_size=4,
                 train_num_workers=4,
                 val_num_workers=4,
                 test_num_workers=4,
                 * args,  **kwargs):
        super().__init__(*args, **kwargs)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        # self.low_limit = low_limit
        # self.high_limit = high_limit
        # self.stack_size = stack_size

        if train:
            self.dataset = EqnPrepareRosbrock()
            train_size = int(0.8*len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size])
        # else:
        #     self.test_dataset = EqnTestPrepareRosebrock()

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

    # def test_dataloader(self):
    #     test_dataloader = DataLoader(self.test_dataset,
    #                                  batch_size=self.test_batch_size,
    #                                  shuffle=False,
    #                                  num_workers=self.test_num_workers)
    #     return test_dataloader


if __name__ == "__main__":
    train_dataset = EqnPrepareRosbrock()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=4,
                                  )

    for batch_idx, batch in enumerate(train_dataloader):

        ''' if we want to return not in dictionary we can check in this way '''

        # input_filt, coh, ddays, bperps, mr, he = batch
        # [B, N] = batch['x1'].shape

        # print(input_filt.shape)

        ''' if we want to return in dictionary we can check in this way '''

        # print('Batch Index \t = {}'.format(batch_idx))
        # print('Input Type \t = {}'.format(batch['input'].dtype))
        # print('Input Shape \t = {}'.format(batch['input'].shape))
        # print('x1 Shape \t = {}'.format(batch['x1']))
        # {batch_size, stack, 1}
        print('x2 Shape \t = {}'.format(batch['a'].shape))

        break
