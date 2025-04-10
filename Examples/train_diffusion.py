import sys
sys.path.insert(1, 'denoising-diffusion-pytorch')
sys.path.insert(2, 'FloodChangeDataset')

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import argparse
from Sen2Flood import Sen2FloodLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('Device:', DEVICE)


def train():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 256,
        timesteps = 1000  # number of steps
    )

    diffusion = diffusion.to(DEVICE)

    # dataset
    dataset = Sen2FloodLoader(dataset_folder='/media/bruno/Matosak/Sen2Flood', chip_size=256,
                              data_to_include=['s1_during_flood'],
                              percentile_scale_bttm=5, percentile_scale_top=95,
                              countries=['Bangladesh', 'India', 'Pakistan', 'Sri Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                              use_data_augmentation=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, drop_last=True)

    for ind, (s1bf) in enumerate(dataloader):
        print(s1bf[0].shape)
        loss = diffusion(s1bf[0][:,:3].to(DEVICE))
        loss.backward()
        break

def sample():
    print('Sample called.')
    # sampled_images = diffusion.sample(batch_size = 4)
    # sampled_images.shape # (4, 3, 128, 128)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--sample', action='store_true')

    args = parser.parse_args()

    if args.train:
        train()
    
    if args.sample:
        sample()