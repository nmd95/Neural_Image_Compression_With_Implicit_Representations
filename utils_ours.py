import numpy as np
import torch
from torch._C import dtype
from typing import Dict
import imageio
import torchvision
from torchvision import transforms
import torch.nn.utils.prune as prune


def load_img_to_tensor(img_path: str):

  dtype = torch.float32
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

  img = imageio.imread(img_path)
  img = transforms.ToTensor()(img).float().to(device, dtype)

  return img

