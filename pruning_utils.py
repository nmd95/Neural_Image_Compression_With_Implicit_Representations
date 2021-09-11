import numpy as np
import torch
from torch._C import dtype
from typing import Dict
import imageio
import torchvision
from torchvision import transforms
import torch.nn.utils.prune as prune
import siren_network
import utils_ours
import training_utils
from torchvision.utils import save_image


dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


def get_prune_params(func_rep, layers:int):

  prune_params = []

  for i in range(0, layers):
    prune_params.append(((func_rep.net[i].linear, 'weight')))
    prune_params.append((func_rep.net[i].linear, 'bias'))
  
  parameters_to_prune = tuple(prune_params)

  return parameters_to_prune


def single_img_experiment(img, model_arch:str, save_path:str, arch_params:list, epochs_per_prune:list, prunning_rates:list):

  # prunning_rates = [0.0, 0.3, 0.2/0.7, 0.4, 0.1/0.3] # corresponds to 0% 30 %, 50 %, 70 % and 90 % of the original model (respectively).
  # epochs_per_prune = [50000, 10000, 5000, 2500, 1250]

  func_rep = siren_network.Siren(
        dim_in=2,
        dim_hidden=arch_params[1],
        dim_out=3,
        num_layers=arch_params[0],
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0
    ).to(device)

  coordinates = ((torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()) / (img.shape[1] - 1) - 0.5) * 2
  features = img.reshape(img.shape[0], -1).T

  # coordinates, features = utils_ours.to_coordinates_and_features(img)
  # coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
  
  
  for i in range(len(prunning_rates)):

    results = {}

    trainer = training_utils.Trainer(func_rep, lr=2e-4)
    trainer.train(coordinates, features, num_iters=epochs_per_prune[i])

    # Log full precision results

    results['fp_psnr']= trainer.best_vals['psnr']

    # Save best model
    torch.save(trainer.best_model, save_path + "/" + model_arch + f'_best_model_fp_p{prunning_rates[i]}.pt')

    func_rep.load_state_dict(trainer.best_model)



    with torch.no_grad():
      img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
      save_image(torch.clamp(img_recon, 0, 1).to('cpu'), save_path + "/" + model_arch + f'_best_model_fp_p{prunning_rates[i]}.png')
    
    
    torch.save(results, save_path + "/" + model_arch + f'_results{prunning_rates[i]}.pkl')


    func_rep.load_state_dict(trainer.best_model)

    # pruning procedure 

    parameters_to_prune = get_prune_params(func_rep, arch_params[0])

    if i < len(epochs_per_prune) - 1:
        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prunning_rates[i+1],
    )
