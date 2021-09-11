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




if __name__ == '__main__':
    random_seed = 52

    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--pruning-rates-list', type=str, required=True)
    parser.add_argument('--epochs-per-prune-list', type=str, required=True)
    parser.add_argument('--model-arch', type=str, required=True)
    parser.add_argument('--arch-params', type=str, required=True)

    original_args = vars(parser.parse_args())
    args = {**original_args}

    list_params = ['pruning_rates_list', 'epochs_per_prune_list', 'arch_params']
    for list_param in list_params:
        values = [eval(value) for value in original_args[list_param].split(',')]
        args[list_param] = values

    # Do something with the arguments

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    img_tensor = utils_ours.load_img_to_tensor(img_path=args['img_path'])

    pruning_utils.single_img_experiment(
      save_path = args['save_path'], 
      img = img_tensor, 
      model_arch = args['model_arch'], 
      arch_params = args['arch_params'], 
      prunning_rates = args['pruning_rates_list'], 
      epochs_per_prune = args['epochs_per_prune_list']
    )




