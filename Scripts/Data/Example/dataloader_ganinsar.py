import numpy as np
import os
import torch
import glob
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def readShortComplex(fileName, width=1):
    return np.fromfile(fileName, '>i2').astype(np.float).view(np.complex).reshape(-1, width)


def readFloatComplex(fileName, width=1):
    return np.fromfile(fileName, '>c8').astype(np.complex).reshape(-1, width)


def readFloat(fileName, width=1):
    return np.fromfile(fileName, '>f4').astype(np.float).reshape(-1, width)


def writeShortComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.copy().view(np.float).astype('>i2').tofile(out_file)
    out_file.close()


def writeFloatComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>c8').tofile(out_file)
    out_file.close()


def writeFloat(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>f4').tofile(out_file)
    out_file.close()


# Patch-wise reading
def readFloatComplexRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 8 / width
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(8 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(8 * patch_size), dtype=">c8").astype(np.complex))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height


def readShortFloatComplexRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            size_of_file = os.path.getsize(fileName)
            # print(size_of_file)
            height = size_of_file / 4 / width
            # print(height)
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">i2").astype(np.float).view(np.complex))
            patches.append(np.reshape(img, [patch_size, patch_size]))

    return patches, rows, cols, height


def readFloatRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 4 / width
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">f4").astype(np.float))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        self.preload()
        return input 


# Single channel reading
class ParameterDataset(Dataset):

    def __init__(self, data_paths, patch_size, n_patch_per_sample, data_width):
        self.data_paths = data_paths
        self.patch_size = patch_size
        self.n_patch_per_sample = n_patch_per_sample
        self.length = len(data_paths) * self.n_patch_per_sample
        self.data_width = data_width

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_idx = int(idx / self.n_patch_per_sample)
        file_path = self.data_paths[file_idx]
        [ps, _, _, _] = readFloatRandomPathces(file_path, width = self.data_width, num_sample=1, patch_size=self.patch_size, height=self.data_width)
        ps = ps[0]
        # ps = (ps-ps.min())/(ps.max()-ps.min())
        return np.expand_dims(ps, 0)

# Double channel reading
class ParameterDatasetCombineMandH(Dataset):

    def __init__(self, data_paths, patch_size, n_patch_per_sample, data_width):
        self.data_paths = data_paths
        self.patch_size = patch_size
        self.n_patch_per_sample = n_patch_per_sample
        self.length = len(data_paths) * self.n_patch_per_sample
        self.data_width = data_width

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_idx = int(idx / self.n_patch_per_sample)
        file_path_mr = self.data_paths[file_idx]
        file_path_he = file_path_mr.replace('/def_fit_cmpy', '/hgt_fit_m')
        [ms, rs, cs, _] = readFloatRandomPathces(file_path_mr, width = self.data_width, num_sample=1, patch_size=self.patch_size, height=self.data_width)
        [hs, _, _, _] = readFloatRandomPathces(file_path_he, width = self.data_width, num_sample=1, patch_size=self.patch_size, height=self.data_width, rows=rs, cols=cs)
        ps = np.concatenate([np.expand_dims(ms[0],0), np.expand_dims(hs[0],0)])
        # ps = (ps-ps.min())/(ps.max()-ps.min())
        return ps

class ParameterDatasetWrap(Dataset):

    def __init__(self, data_paths, patch_size, n_patch_per_sample, data_width):
        self.data_paths = data_paths
        self.patch_size = patch_size
        self.n_patch_per_sample = n_patch_per_sample
        self.length = len(data_paths) * self.n_patch_per_sample
        self.data_width = data_width

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_idx = int(idx / self.n_patch_per_sample)
        file_path = self.data_paths[file_idx]
        [ps, _, _, _] = readFloatRandomPathces(file_path, width = self.data_width, num_sample=1, patch_size=self.patch_size, height=self.data_width)
        ps = ps[0]
        # ps = (ps-ps.min())/(ps.max()-ps.min())
        t = (abs(ps)/(ps+1e-10))*(abs(ps)%10)/10
        t = np.expand_dims(t, 0)
        t2 = (abs(ps)/(ps+1e-10))*(abs(ps)//10)
        t2 = np.expand_dims(t2, 0)
        return np.concatenate([t, t2], 0)


if __name__ == "__main__":

    all_paths = glob.glob('/mnt/hdd1/alvinsun/3vG-Parameter-Fitting-Data/*/fit_hr/def_fit_cmpy')

    # d = ParameterDataset(all_paths, 256, 100, 1500)
    d = ParameterDatasetCombineMandH(all_paths, 256, 100, 1500)

    dataloader = DataLoader(d, batch_size=32,
                        shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)

