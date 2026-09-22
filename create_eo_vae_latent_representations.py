import sys
sys.path.insert(1, '../SenForFlood')
sys.path.insert(2, '../eo-vae')

from SenForFlood import SenForFlood
from eo_vae.models.new_autoencoder import EOFluxVAE


from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def get_save_file_path(dataset, ind, i, batch_size, save_folder_path, data_type):
    original_file = dataset.samples_ids[ind*batch_size+i][0]
    internal_tile = dataset.samples_ids[ind*batch_size+i][1]

    folder_structure = original_file.split('SenForFlood')[-1].replace('.tif', f'_{internal_tile}.pt').replace('flood_mask_v1.1', data_type)[1:]
    final_path = os.path.join(save_folder_path, folder_structure)
    return final_path

def prepare_data_for_plotting(data_input):
    data = np.zeros([256,256,3], dtype=np.float32)+0.5
    data[:,:,0] = (data_input[0]+2.5)/5
    data[:,:,1] = (data_input[1]+2.5)/5
    return np.clip(data, 0, 1)

if __name__=='__main__':
    batch_size = 32
    device = torch.device('cuda')

    save_folder_path = r"D:\Bruno\datasets\SenForFlood_Latents"

    senforflood_dfo = SenForFlood(
        '../../datasets/SenForFlood',
        source='DFO',
        shuffle_seed=0,
        chip_size=256,
        data_to_include=['s1_before_flood', 's1_during_flood'],
        use_data_augmentation=False,
        normalize=True,
        normalize_s1_to_match_terramesh=True
    )

    senforflood_cems = SenForFlood(
        '../../datasets/SenForFlood',
        source='CEMS',
        shuffle_seed=0,
        chip_size=256,
        data_to_include=['s1_before_flood', 's1_during_flood'],
        use_data_augmentation=False,
        normalize=True,
        normalize_s1_to_match_terramesh=True
    )

    vae = EOFluxVAE.from_pretrained(
        repo_id="nilsleh/eo-vae", 
        ckpt_filename="eo-vae.ckpt",
        config_filename="model_config.yaml",
        device=device,
    )
    vae.eval()

    wvs = torch.tensor([5.4, 5.6], dtype=torch.float32).to(device)

    with torch.no_grad():
        for dataset in [senforflood_cems, senforflood_dfo]:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
            for ind, (s1b, s1d) in tqdm(enumerate(dataloader), ncols=100, total=len(dataloader)):
                z_s1b = vae.encode_spatial_normalized(s1b[:,:2].to(device), wvs)
                z_s1d = vae.encode_spatial_normalized(s1d[:,:2].to(device), wvs)

                # saving the files for later usage
                for i in range(z_s1b.shape[0]):
                    save_file_path = get_save_file_path(dataset, ind, i, batch_size, save_folder_path, 's1_before_flood')
                    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                    torch.save(
                        z_s1b[i].detach().cpu(),
                        save_file_path
                    )

                    save_file_path = get_save_file_path(dataset, ind, i, batch_size, save_folder_path, 's1_during_flood')
                    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                    torch.save(
                        z_s1d[i].detach().cpu(),
                        save_file_path
                    )
                
                # checking something real quick...
                # import matplotlib.pyplot as plt
                # f, ax = plt.subplots(2,8, figsize=(16,4))
                # for i in range(8):
                #     s1_i = s1d[i].detach().cpu().numpy() # vae.decode_spatial_normalized(z_s1b).detach().cpu().numpy()[i]
                #     s1_o = vae.decode_spatial_normalized(z_s1d, wvs).detach().cpu().numpy()[i]
                #     ax[0,i].imshow(prepare_data_for_plotting(s1_i))
                #     ax[1,i].imshow(prepare_data_for_plotting(s1_o))
                #     ax[0,i].axis('off')
                #     ax[1,i].axis('off')
                # plt.tight_layout()
                # plt.show()

                # plt.hist(s1_i[:2].ravel(), alpha=0.5)
                # # plt.hist(s1_o.ravel(), alpha=0.5)
                # plt.show()
                # exit()