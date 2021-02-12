import numpy as np
import argparse
from ..utils.utils import readFloat
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/mnt/hdd1/alvinsun/3vG-Parameter-Fitting-Data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/')
parser.add_argument('--he', type=str, default='hgt_fit_m')
parser.add_argument('--mr', type=str, default='def_fit_cmpy')
parser.add_argument('--out_dir', type=str)

args = parser.parse_args()

def demo():
    mr_path = '{}/{}'.format(args.data_dir, args.mr)
    he_path = '{}/{}'.format(args.data_dir, args.he)

    mr = readFloat(mr_path, 1500)
    he = readFloat(he_path, 1500)

    mr_arc_h = mr[:-1, :] - mr[1:, :]
    mr_arc_v = mr[:, :-1] - mr[:, 1:]

    he_arc_h = he[:-1, :] - he[1:, :]
    he_arc_v = he[:, :-1] - he[:, 1:]

    fig, ax = plt.subplots(1,4, figsize=(14,3))
    ax[0].imshow(mr, cmap='rainbow', vmin=-1, vmax=1)
    ax[1].imshow(mr, cmap='rainbow', vmin=-5, vmax=5)
    ax[2].imshow(mr, cmap='rainbow', vmin=-10, vmax=10)
    ax[3].imshow(mr, cmap='rainbow', vmin=-15, vmax=15)
    fig.show()

    fig, ax = plt.subplots(1,4, figsize=(14,3))
    ax[0].imshow(he, cmap='rainbow', vmin=-1, vmax=1)
    ax[1].imshow(he, cmap='rainbow', vmin=-5, vmax=5)
    ax[2].imshow(he, cmap='rainbow', vmin=-10, vmax=10)
    ax[3].imshow(he, cmap='rainbow', vmin=-15, vmax=15)
    fig.show()

    fig, ax = plt.subplots(1,4, figsize=(14,3))
    ax[0].imshow(mr_arc_h, cmap='rainbow', vmin=-1, vmax=1)
    ax[1].imshow(mr_arc_v, cmap='rainbow', vmin=-5, vmax=5)
    ax[2].imshow(mr_arc_v, cmap='rainbow', vmin=-10, vmax=10)
    ax[3].imshow(mr_arc_v, cmap='rainbow', vmin=-15, vmax=15)
    fig.show()

    fig, ax = plt.subplots(1,4, figsize=(14,3))
    ax[0].imshow(he_arc_h, cmap='rainbow', vmin=-1, vmax=1)
    ax[1].imshow(he_arc_v, cmap='rainbow', vmin=-5, vmax=5)
    ax[2].imshow(he_arc_v, cmap='rainbow', vmin=-10, vmax=10)
    ax[3].imshow(he_arc_v, cmap='rainbow', vmin=-15, vmax=15)
    fig.show()
    input()

if __name__ == "__main__":
    