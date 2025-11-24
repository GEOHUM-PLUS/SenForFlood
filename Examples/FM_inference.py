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

def prepare_data_for_plot(x0, x1, x1_):
    data1 = x0.copy()
    data2 = x1.copy()
    data3 = x1_.copy()
    
    div1 = np.positive(data1[0])/np.positive(data1[1])
    div2 = np.positive(data2[0])/np.positive(data2[1])
    div3 = np.positive(data3[0])/np.positive(data3[1])

    data1 = np.moveaxis(np.concatenate([data1, div1[None,:,:]], axis=0), 0, -1)
    data2 = np.moveaxis(np.concatenate([data2, div2[None,:,:]], axis=0), 0, -1)
    data3 = np.moveaxis(np.concatenate([data3, div3[None,:,:]], axis=0), 0, -1)

    for i in range(3):
        merged = np.concatenate([data1[:,:,i], data2[:,:,i], data3[:,:,i]], axis=0)
        p = np.percentile(merged[np.isfinite(merged)], q=[2,98])
        data1[:,:,i] = (data1[:,:,i]-p[0])/(p[1]-p[0])
        data2[:,:,i] = (data2[:,:,i]-p[0])/(p[1]-p[0])
        data3[:,:,i] = (data3[:,:,i]-p[0])/(p[1]-p[0])
    
    data1[~np.isfinite(data1)] = 0
    data2[~np.isfinite(data2)] = 0
    data3[~np.isfinite(data3)] = 0

    data1 = np.clip(data1, 0, 1)
    data2 = np.clip(data2, 0, 1)
    data3 = np.clip(data3, 0, 1)

    return data1, data2, data3

def plot_pairs(data_before, data_after, data_after_est, folder_save):

    for i in range(data_before.shape[0]):
    
        f, ax = plt.subplots(2, 3, figsize=(3*3,3*2))

        db, da, da_ = prepare_data_for_plot(data_before[i].numpy(), data_after[i].numpy(), data_after_est[-1][i].numpy())

        ax[0,0].imshow(db)
        ax[0,0].axis('off')
        ax[0,0].title.set_text('$X_0$')

        ax[0,1].imshow(da)
        ax[0,1].title.set_text('$X_1$')
        ax[0,1].axis('off')

        ax[0,2].imshow(da_)
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
    model_id = 'test_times'
    model_name = 'model-e0100'

    data_model = torch.load(f'models_FM/{model_id}/Checkpoints/{model_name}.pt', weights_only=True)

    os.makedirs(f'models_FM/{model_id}/plots', exist_ok=True)
    test_dataset = SenForFlood(
        dataset_folder='/media/bruno/Matosak/SenForFlood',
        chip_size=512,
        events=['DFO_4459_Bangladesh'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain'] if data_model['use_terrain'] else ['s1_before_flood', 's1_during_flood'],
        use_data_augmentation=False,
        normalize=True,
        # scale_0_1=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, drop_last=False)

    for ind, data in enumerate(test_loader):
        x1 = data[0][:,:2]
        x0 = data[1][:,:2]
        slope = torch.Tensor([0])
        if data_model['use_terrain']:
            slope = data[2][:,1][:,None,:,:]
        break
    
    # create the model
    model = AttUNet_t(
        in_channels=2,
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