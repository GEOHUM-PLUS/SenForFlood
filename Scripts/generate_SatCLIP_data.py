import sys
sys.path.insert(1, '/media/bruno/Matosak/repos/satclip/satclip')

from huggingface_hub import hf_hub_download
from load import get_satclip
import torch
import rasterio as r
import glob
from tqdm import tqdm
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_sample_centroid(sample_path):
    sample = r.open(sample_path)

    center = r.warp.transform(
        sample.crs, 
        'EPSG:4326', 
        [(sample.bounds.right+sample.bounds.left)/2],
        [(sample.bounds.top+sample.bounds.bottom)/2])

    return center

def get_samples_paths(dataset_path):
    return (glob.glob(dataset_path+'/DFO/*/*/flood_mask_v1.1/*_flood_mask_v1.1.tif')
     + glob.glob(dataset_path+'/CEMS/*/flood_mask_v1.1/*_flood_mask_v1.1.tif'))

def get_embeddings(coordinates, model):
    c = torch.Tensor(coordinates).squeeze()
    model.eval()
    with torch.no_grad():
        emb = model(c.double().to(DEVICE)).detach().cpu()
    
    return emb

def store_embedding(embedding, sample_path):
    emb_path = sample_path.replace('flood_mask_v1.1', 'SatCLIP_embedding').replace('.tif', '.npy')
    os.makedirs(emb_path.rsplit('/',1)[0], exist_ok=True)
    np.save(emb_path, embedding)

if __name__=='__main__':
    batch_size = 256

    samples_paths = get_samples_paths('/media/bruno/Matosak/SenForFlood')
    print('Total Samples:', len(samples_paths))

    model = get_satclip(
        hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
        device=DEVICE,
    )  # Only loads location encoder by default

    for i in tqdm(range(0, len(samples_paths), batch_size), ncols=150):
        batch_paths = samples_paths[i:i+batch_size]
        centroids = []
        for sample_path in batch_paths:
            centroids.append(get_sample_centroid(sample_path))
        embeddings = get_embeddings(centroids, model)

        for j in range(len(batch_paths)):
            store_embedding(embeddings[j].detach().cpu().numpy(), batch_paths[j])