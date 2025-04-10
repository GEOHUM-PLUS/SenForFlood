from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent))

from Sen2Flood import Sen2FloodLoader
from plots import plot_curves
from IPython.display import clear_output
from unet import UNet
import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('Device:', DEVICE)

def train():
    model = UNet(in_channels=5, out_channels=2, dropout_val=0.2).to(DEVICE)
    dataset = Sen2FloodLoader(dataset_folder='/media/bruno/Matosak/Sen2Flood', chip_size=256,
                              data_to_include=['s1_during_flood', 'terrain', 'flood_mask'],
                              percentile_scale_bttm=5, percentile_scale_top=95,
                              countries=['Bangladesh', 'India', 'Pakistan', 'Sri Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                              use_data_augmentation=True)

    print(f'Total Samples: {len(dataset)}')

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    training_curves = {'Loss': {'train': [], 'eval': []},
                   'Accuracy': {'train': [], 'eval': []},
                   }
    
    save_model_path = '/home/bruno/dataset_Sen2Flood/models/UNet_SouthAsia.pt'

    def model_train(input_train, output_train):
        model.train()
        output_ = model(input_train)
        loss_train = criterion(output_.type(torch.float32), output_train)
        model.zero_grad()
        loss_train.backward()
        opt.step()

        acc_train = (output_.argmax(1) == output_train.argmax(1)).type(torch.float).sum().item()/(output_train.shape[0]*output_train.shape[2]*output_train.shape[3])

        training_curves['Loss']['train'].append(loss_train.item())
        training_curves['Accuracy']['train'].append(acc_train)

    def model_val(input_eval, output_eval):
        model.eval()
        with torch.no_grad():
            output_ = model(input_eval)
            acc_eval = (output_.argmax(1) == output_eval.argmax(1)).type(torch.float).sum().item()/(output_eval.shape[0]*output_eval.shape[2]*output_eval.shape[3])
            loss_eval = criterion(output_.type(torch.float32), output_eval)
            training_curves['Loss']['eval'].append(loss_eval.item())
            training_curves['Accuracy']['eval'].append(acc_eval)

            return acc_eval


    max_epoch = 50
    train_size = 64
    batch_size = 64+32

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

    step = 0
    best_acc = 0
    for epoch in range(max_epoch):
        epoch_acc = []
        for ind, (s1b, te, fm) in enumerate(dataloader):
            # train
            input_train = torch.cat([s1b[:train_size,:3,:,:], te[:train_size,:,:,:]], dim=1).type(torch.float32).to(DEVICE)
            output_train = torch.nn.functional.one_hot((fm[:train_size,:,94:-94,94:-94]*2).to(torch.int64), 3).squeeze().moveaxis(-1,1)[:,:2].type(torch.float32).to(DEVICE)
            
            model_train(input_train, output_train)

            # validate
            input_eval = torch.cat([s1b[train_size:,:3,:,:], te[train_size:,:,:,:]], dim=1).type(torch.float32).to(DEVICE)
            output_eval = torch.nn.functional.one_hot((fm[train_size:,:,94:-94,94:-94]*2).to(torch.int64), 3).squeeze().moveaxis(-1,1)[:,:2].type(torch.float32).to(DEVICE)

            val_acc = model_val(input_eval, output_eval)
            epoch_acc.append(val_acc)
            
            step += 1

            if step%10==0:
                clear_output()
                plot_curves(training_curves, [i for i in range(step)], 
                            smoothed=True,
                            save_path='/home/bruno/dataset_Sen2Flood/models/curves_UNet_SouthAsia.png',
                            show=False,
                            title=f'Epochs: {epoch}')
            
        if numpy.mean(epoch_acc)>best_acc:
            model.eval()
            best_acc = numpy.mean(epoch_acc)
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model ever stored (acc. {best_acc:.4f}).")

def predict():
    model_path = '/home/bruno/dataset_Sen2Flood/models/UNet_SouthAsia.pt'
    model = UNet(in_channels=5, out_channels=2, dropout_val=0.2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)

    model.eval()

    batch_size = 64+32
    train_size = 64

    dataset = Sen2FloodLoader(dataset_folder='/media/bruno/Matosak/Sen2Flood', chip_size=256,
                              data_to_include=['s1_during_flood', 'terrain', 'flood_mask'],
                              percentile_scale_bttm=5, percentile_scale_top=95,
                              countries=['Bangladesh', 'India', 'Pakistan', 'Sri Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                              use_data_augmentation=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)
    
    for ind, (s1b, te, fm) in enumerate(dataloader):
        input_eval = torch.cat([s1b[train_size:,:3,:,:], te[train_size:,:,:,:]], dim=1).type(torch.float32).to(DEVICE)
        output_eval = torch.nn.functional.one_hot((fm[train_size:,:,94:-94,94:-94]*2).to(torch.int64), 3).squeeze().moveaxis(-1,1)[:,:2].type(torch.float32).to(DEVICE)
        with torch.no_grad():
            output_ = model(input_eval)
        
        out_ =    output_.detach().cpu().numpy()
        inp =  input_eval.detach().cpu().numpy()
        out = output_eval.detach().cpu().numpy()

        for i in range(16):
            f, ax = plt.subplots(1,4)
            ax[0].imshow(numpy.moveaxis(inp[i,:3,94:-94,94:-94], 0, -1))
            ax[1].imshow(numpy.moveaxis(inp[i,4,94:-94,94:-94], 0, -1))
            ax[2].imshow(numpy.argmax(out[i,:,:,:], axis=0), vmin=0, vmax=1)
            ax[3].imshow(numpy.argmax(out_[i,:,:,:], axis=0), vmin=0, vmax=1)
            for k in range(4):
                ax[k].axis('off')
            plt.tight_layout()
            plt.savefig(f'/home/bruno/dataset_Sen2Flood/models/{ind:03d}-{i:02d}.png')
            plt.close()

if __name__=='__main__':
    train()