import sys
sys.path.insert(1, '/media/bruno/Matosak/repos/SenForFlood')

import torch
from torch import nn
from SenForFlood import SenForFlood
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import UNet_t, AttUNet_t

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def prepare_data_for_plot(data):
    data_div = data[0]/data[1]
    p = np.nanpercentile(data_div, q=[5,95])
    data_div = (data_div-p[0])/(p[1]-p[0])
    data_div = np.clip(data_div, 0, 1)
    data_ = np.clip(data, 0, 1)
    return np.moveaxis(np.concatenate([data_, data_div[None,:,:]], axis=0), 0, -1)

def plot_pairs(data_before, data_after, data_after_est, folder_save):

    for i in range(data_before.shape[0]):
    
        f, ax = plt.subplots(2, 3, figsize=(3*3,3*2))

        ax[0,0].imshow(prepare_data_for_plot(data_before[i].numpy()))
        ax[0,0].axis('off')
        ax[0,0].title.set_text('$X_0$')

        ax[0,1].imshow(prepare_data_for_plot(data_after[i].numpy()))
        ax[0,1].title.set_text('$X_1$')
        ax[0,1].axis('off')

        ax[0,2].imshow(prepare_data_for_plot(data_after_est[-1][i]))
        ax[0,2].title.set_text('$X_1\'$')
        ax[0,2].axis('off')


        diff = np.linalg.norm(data_after[i]-data_before[i], axis=0)
        percentiles = np.percentile(diff, q=[2,98])
        ax[1,0].imshow(diff, vmin=percentiles[0], vmax=percentiles[1])
        ax[1,0].title.set_text('$|X_1-X_0|$')
        ax[1,0].axis('off')

        diff = np.linalg.norm(data_after_est[-1][i]-data_before[i], axis=0)
        percentiles = np.percentile(diff, q=[2,98])
        ax[1,1].imshow(diff, vmin=percentiles[0], vmax=percentiles[1])
        ax[1,1].title.set_text('$|X_1\'-X_0|$')
        ax[1,1].axis('off')

        diff = np.linalg.norm(data_after_est[-1][i]-data_after[i], axis=0)
        percentiles = np.percentile(diff, q=[2,98])
        ax[1,2].imshow(diff, vmin=percentiles[0], vmax=percentiles[1])
        ax[1,2].title.set_text('$|X_1\'-X_1|$')
        ax[1,2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{folder_save}/{i:02d}.png')
        plt.close()

if __name__=='__main__':
    model_id = 'test_bangladesh'
    model_name = 'model-e0020'

    data_model = torch.load(f'models_FM/{model_id}/Checkpoints/{model_name}.pt', weights_only=True)

    os.makedirs(f'models_FM/{model_id}/plots', exist_ok=True)
    test_dataset = SenForFlood(
        dataset_folder='/media/bruno/Matosak/SenForFlood',
        chip_size=512,
        events=['DFO_4459_Bangladesh'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain'] if data_model['use_terrain'] else ['s1_before_flood', 's1_during_flood'],
        use_data_augmentation=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, drop_last=False)

    for ind, data in enumerate(test_loader):
        x1 = data[0][:,:2]
        x0 = data[1][:,:2]
        slope = torch.Tensor([0])
        if data_model['use_terrain']:
            slope = data[2][:,1][:,None,:,:]
        break
    
    # create the model
    model = AttUNet_t(
        in_channels = 3 if data_model['use_terrain'] else 2, 
        out_channels=2, 
        base=data_model['model_base'], 
        use_terrain = data_model['use_terrain']
    ).to(DEVICE)
    model.load_state_dict(data_model['model_state_dict'])

    model.eval()
    with torch.no_grad():
        xts = [x0]
        t_span = torch.linspace(0, 1, 100)
        for s,t in zip(t_span[:-1], t_span[1:]):
            xt = xts[-1]
            t_expanded = torch.Tensor([t]).repeat(xt.shape[0])
            xts.append((model(xt.to(DEVICE), t_expanded.to(DEVICE), slope.to(DEVICE)).detach().cpu() * (t - s) + xt).detach().cpu())
        plot_pairs(x0, x1, xts, f'models_FM/{model_id}/plots')