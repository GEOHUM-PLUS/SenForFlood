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

def plot_loss(losses, save_path, smooth_window=100):
    plt.figure(figsize=(8,4))
    plt.plot(losses, alpha=0.5, label='raw loss')
    plt.plot(smooth_series(losses, smooth_window), label='smoothed loss')
    plt.title('Loss', fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    model = AttUNet_t(in_channels=2, out_channels=2, base=args.base).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # get dataloaders
    train_dataset = SenForFlood(
        dataset_folder='/media/bruno/Matosak/SenForFlood',
        chip_size=args.chip_size,
        # events=['DFO_4459_Bangladesh'],
        data_to_include=['s1_before_flood', 's1_during_flood'],
        use_data_augmentation=True
    )
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
        for ind, (x1, x0) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            x0 = x0[:,:2].to(DEVICE)
            x1 = x1[:,:2].to(DEVICE)

            t = (torch.zeros([x0.shape[0], 1, x0.shape[2], x0.shape[3]])+torch.rand(x0.shape[0])[:,None,None,None]).to(DEVICE)

            # xt = (t*x1)+(1-t)*x0
            xt = torch.cos(torch.pi*t/2)*x0 + torch.sin(torch.pi*t/2)*x1
            # ut = x1 - x0
            ut = (torch.pi/2) * (torch.cos(torch.pi*t/2)*x1 - torch.sin(torch.pi*t/2)*x0)

            if args.sigma>0:
                ut = ut+(args.sigma*torch.randn_like(ut))
            
            vt = model(xt, t)
            loss = ((vt-ut)**2).mean()
            
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
            times.append(time.time()-initial_time)
            initial_time = time.time()

            print(f'epoch: {i+1}  steps: {ind+1}/{steps}  {f"loss_step: {losses[-1]:0.4f}" if ind!=(steps-1) else f"mean_loss: {np.mean(losses[-steps:]):.4f}"}  {np.mean(times[max(-10, -len(times)):]):.2f}s/i{f"  rt: {str(datetime.timedelta(seconds=int(np.mean(times[max(-10, -len(times))])*(steps-1-ind))))}" if ind!=(steps-1) else f"  total_time: {str(datetime.timedelta(seconds=int(np.sum(times[-steps:]))))}"}', end='\n' if ind==(steps-1) else '\r')

        # save intermediary model, if necessary
        if (i+1)%args.epochs_save_model==0 or (i+1)==args.epochs:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                'model_base': model.base
            }, f'models_FM/{args.model_identifier}/Checkpoints/model-e{(i+1):04d}.pt')
    
        # plotting the losses
        plot_loss(losses, f'models_FM/{args.model_identifier}/losses.png', smooth_window=steps)