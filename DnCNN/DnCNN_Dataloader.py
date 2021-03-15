import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision as tv
from PIL import Image


# Download the Dataset from terminal

# !wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
# !tar xvzf BSDS300-images.tgz
# !rm BSDS300-images.tgz


class NoisyDataset(Dataset):
  
  def __init__(self, in_path, mode='train', img_size=(180, 180), sigma=30):
    super(NoisyDataset, self).__init__()

    self.mode = mode #train or test
    self.in_path = in_path # ./BSDS300/images
    self.img_size = img_size # (180, 180)


    self.img_dir = os.path.join(in_path, mode)
    self.imgs = os.listdir(self.img_dir)
    self.sigma = sigma

  def __len__(self):
      return len(self.imgs)
  
  def __repr__(self):
      return "Dataset Parameters: mode={}, img_size={}, sigma={}".format(self.mode, self.img_size, self.sigma)
    
  def __getitem__(self, idx):

      img_path = os.path.join(self.img_dir, self.imgs[idx])
      clean_img = Image.open(img_path).convert('RGB')
      left = np.random.randint(clean_img.size[0] - self.img_size[0])
      top = np.random.randint(clean_img.size[1] - self.img_size[1])
      # .crop(left, upper, right, lower)
      cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
      transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
      
      ground_truth = transform(cropped_clean)

      noisy = ground_truth + 2 / 255 * self.sigma * torch.randn(ground_truth.shape)
      
      return noisy, ground_truth 
