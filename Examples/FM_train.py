# Bruno Menini Matosak
# bruno.menini-matosak(at)plus.ac.at
import sys
sys.path.insert(1, '/media/bruno/Matosak/repos/SenForFlood')

from SenForFlood import SenForFlood

import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy
import argparse
import os
import time
import datetime
from models import AttUNet_t, UNet_t
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Auxiliary functions
def smooth_series(series, window_size = 10):
    series_np = np.asarray(series)
    series_smooth = np.copy(series_np)

    overlap = int(window_size/2)

    for i in range(len(series_smooth)):
        series_smooth[i] = np.median(series_np[max(-overlap+i,0):min(overlap+i, len(series_smooth))])

    return series_smooth

def plot_loss(losses, save_path=None, smooth_window=100):
    f, ax = plt.subplots(1,2, figsize=(10,4), gridspec_kw={'width_ratios': [2, 1]})

    f.suptitle('Loss', fontweight='bold')

    ax[0].scatter(range(len(losses)), losses, facecolors=None, alpha=0.1, label='raw loss')
    ax[0].plot(smooth_series(losses, smooth_window), label='smoothed loss', color='tab:orange')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid()
    
    ax[1].scatter(range(len(losses)), losses, facecolors=None, alpha=0.1, label='raw loss')
    ax[1].plot(smooth_series(losses, int(len(losses)/20)), label='smoothed loss', color='tab:orange')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlim([len(losses)/2, len(losses)])
    ax[1].set_ylim([min(losses[int(len(losses)/2):]), max(losses[int(len(losses)/2):])])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def trajectory(self, x, t_span):
    xts = [x]
    for s,t in zip(t_span[:-1], t_span[1:]):
        xt = xts[-1]
        t_expanded = (torch.zeros([x.shape[0],1,x.shape[2],x.shape[3]])+t)
        xts.append(self.forward(xt, t_expanded.to(DEVICE)) * (t - s) + xt)
    return torch.stack(xts, dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--model-identifier', type=str, required=True, help='The model identifier. The name to a folder used to store the model files.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help="The initial learning rate. Must be greater than 0. Default to 0.001.")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="The maximum amount of epochs. Must be greater than 0. Default to 50.")
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help="The batch size. Must be greater than 0. Default to 256.")
    parser.add_argument('-esm', '--epochs-save-model', type=int, default=10, help="The interval (in epochs) to save the model during training. Must be greater than 0. Default to 5.")
    parser.add_argument('-s', '--sigma', type=float, default=0.0, help="The sigma value. Must be greater than 0. Default to 0.01.")
    parser.add_argument('-b', '--base', type=int, default=64)
    parser.add_argument('-cs', '--chip-size', type=int, default=128)
    parser.add_argument('-d', '--dropout', type=float, default=0.1)
    parser.add_argument('-t', '--terrain', action='store_true')
    args = parser.parse_args()

    if args.batch_size<=0:
        raise ValueError('Invalid batch size (-bs). Must be an integer greater than 0.')
    if args.epochs<=0:
        raise ValueError('Invalid amount of epochs (-e). Must be an integer greater than 0.')
    if args.learning_rate<=0:
        raise ValueError('Invalid learning rate (-lr). Must be a float greater than 0.')
    if args.epochs_save_model<=0:
        raise ValueError('Invalid interval in epochs to save model (-es). Must be an integer greater than 0.')
    if args.sigma<0:
        raise ValueError('Invalid sigma (-s). Must be a float greater than or equal to 0.')

    print('Device:', DEVICE)

    # creates folder to store the files generated during training
    # this folder has the same name as given for model_id
    os.makedirs(f'models_FM/{args.model_identifier}/Checkpoints', exist_ok=True)
    with open(f'models_FM/{args.model_identifier}/hyperparam.txt', 'w') as f:
        args_dict = vars(args)
        for key in args_dict.keys():
            f.write(f'{key}: {args_dict[key]}\n')

    # create the model
    model = AttUNet_t(in_channels=2, out_channels=2, base=args.base, dropout=args.dropout, use_terrain=args.terrain).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # get dataloaders
    train_dataset = SenForFlood(
        dataset_folder='/media/bruno/Matosak/SenForFlood',
        chip_size=args.chip_size,
        # events=['DFO_4459_Bangladesh'],
        countries=['Bangladesh'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain'] if args.terrain else ['s1_before_flood', 's1_during_flood'],
        use_data_augmentation=True
    )
    print(f'{len(train_dataset):,d} training chips ({args.chip_size}x{args.chip_size})')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4)

    #####################################
    # training the model

    times = []
    steps = len(train_loader)
    starting_epoch = 0
    losses = []

    if glob.glob(f'models_FM/{args.model_identifier}/Checkpoints/model-e*.pt'):
        files = glob.glob(f'models_FM/{args.model_identifier}/Checkpoints/model-e*.pt')
        files.sort()
        data = torch.load(files[-1], weights_only=True)

        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        losses = data['loss']
        starting_epoch = data['epoch']+1
    
    initial_time = time.time()

    # train loop
    for i in range(starting_epoch, args.epochs, 1):
        for ind, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            x0 = data[1][:,:2].to(DEVICE)
            x1 = data[0][:,:2].to(DEVICE)

            slope = None
            if args.terrain:
                slope = data[2][:,1][:,None,:,:].to(DEVICE)

            t = torch.rand(x0.shape[0]).to(DEVICE)

            # xt = (t*x1)+(1-t)*x0
            xt = torch.cos(torch.pi*t[:,None,None,None]/2)*x0 + torch.sin(torch.pi*t[:,None,None,None]/2)*x1
            # ut = x1 - x0
            ut = (torch.pi/2) * (torch.cos(torch.pi*t[:,None,None,None]/2)*x1 - torch.sin(torch.pi*t[:,None,None,None]/2)*x0)

            if args.sigma>0:
                ut = ut+(args.sigma*torch.randn_like(ut))
            
            vt = model(xt, t, slope)
            loss = ((vt-ut)**2).mean()
            
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
            times.append(time.time()-initial_time)
            initial_time = time.time()

            print(f'epoch: {i+1}  steps: {ind+1}/{steps}  {f"loss_step: {losses[-1]:0.4f}" if ind!=(steps-1) else f"mean_loss: {np.mean(losses[-steps:]):.4f}"}  {np.mean(times[max(-10, -len(times)):]):.2f}s/i{f"  rt: {str(datetime.timedelta(seconds=int(np.mean(times[max(-10, -len(times))])*(steps-1-ind))))}        " if ind!=(steps-1) else f"  total_time: {str(datetime.timedelta(seconds=int(np.sum(times[-steps:]))))}"}', end='\n' if ind==(steps-1) else '\r')

        # save intermediary model, if necessary
        if (i+1)%args.epochs_save_model==0 or (i+1)==args.epochs:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                'model_base': model.base,
                'use_terrain': model.use_terrain
            }, f'models_FM/{args.model_identifier}/Checkpoints/model-e{(i+1):04d}.pt')
    
        # plotting the losses
        plot_loss(losses, f'models_FM/{args.model_identifier}/losses.png', smooth_window=steps)