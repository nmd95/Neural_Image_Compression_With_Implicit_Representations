import argparse
import getpass
import imageio
import json
import os
import random
import torch

import utils_ours
import pruning_utils
import training_utils
import siren_network

from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.utils.prune as prune




if __name__ == "__main__":


  random_seed = 52
  img_path = "/content/drive/MyDrive/working_with_COIN/our_repo_pruning/kodak-dataset/kodim01.png"
  save_path = "/content/drive/MyDrive/working_with_COIN/our_repo_pruning/exec_logs"
  prunning_rates_list = [0.0, 0.3, 0.2/0.7, 0.4, 0.1/0.3] # corresponds to 0% 30 %, 50 %, 70 % and 90 % of the original model (respectively).
  epochs_per_prune_list = [50, 10, 5, 3, 1]
  model_arch = "5_20"
  arch_params = [5, 20]
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)

  img_tensor = utils_ours.load_img_to_tensor(img_path=img_path)

  
  pruning_utils.single_img_experiment(save_path=save_path, img=img_tensor, model_arch=model_arch, arch_params=arch_params, prunning_rates=prunning_rates_list, epochs_per_prune=epochs_per_prune_list)




