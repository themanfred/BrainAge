import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from model import Encoder, Decoder
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

import nibabel as nib

import numpy as np
import os
import argparse
import time

# Author: @simonamador

# The following code performs the training of the AE model. The training can be performed for different
# views, model types, loss functions, epochs, and batch sizes. 

# Dataset generator class. It inputs the dataset path and view, outputs the image given an index.
# performs image extraction according to the view, normalization and convertion to tensor.

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))

class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(MS_SSIM_Loss, self).forward(img1, img2))
    
class Mixed(SSIM):
    def forward(self, img1, img2):
        # L1 = nn.L1Loss()

        # sz1 = img2.size(dim=2)
        # sz2 = img2.size(dim=3)

        # g1 = torch.arange(-(sz1/2), sz1/2)
        # g1 = torch.exp(-1.*g1**2/(2*0.5**2))
        # g2 = torch.arange(-(sz2/2), sz2/2)
        # g2 = torch.exp(-1.*g2**2/(2*0.5**2))
        # Gl = torch.outer(g1, g2)
        # Gl /= torch.sum(Gl)

        return 100 * (0.84 * ( 1 - super(Mixed, self).forward(img1, img2)) + (1-0.84) * torch.mean(((img1-img2)**2)*self.win))

class img_dataset(Dataset):
    def __init__(self, root_dir, view):
        self.root_dir = root_dir
        self.view = view

    def __len__(self):
        if self.view == 'L':
            size = 110
        elif self.view == 'A':
            size = 158
        else:
            size = 126
        return size
    
    def __getitem__(self, idx):
        raw = nib.load(self.root_dir).get_fdata()
        if self.view == 'L':
            n_img = raw[idx,:158,:]    
        elif self.view == 'A':
            n_img = raw[:110,idx,:]
        else:
            n_img = raw[:110,:158,idx]

        num = n_img-np.min(n_img)
        den = np.max(n_img)-np.min(n_img)
        out = np.zeros((n_img.shape[0], n_img.shape[1]))
    
        n_img = np.divide(num, den, out=out, where=den!=0)

        n_img = np.expand_dims(n_img,axis=0)
        n_img = torch.from_numpy(n_img).type(torch.float)

        return n_img

# Validation function. Acts as the testing portion of training. Inputs the testing dataloader, encoder and
# decoder models, and the loss function. Outputs the loss of the model on the testing data.
def validation(ds,encoder,decoder,loss):
    encoder.eval()
    decoder.eval()

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    ae_loss = 0.0
    metric1 = 0.0
    metric2 = 0.0
    metric3 = 0.0 

    with torch.no_grad():
        for data in ds:
            img = data.to(device)

            z = encoder(img)
            x_recon = decoder(z)

            ed_loss = loss(x_recon, img)

            ae_loss += ed_loss
            metric1 += ssim(x_recon, img, data_range=1.0, win_size = 11)
            metric2 += mse(x_recon, img)
            metric3 += mae(x_recon, img)
        ae_loss /= len(ds)
        metric1 /= len(ds)
        metric2 /= len(ds)
        metric3 /= len(ds)        
    
    metrics = (ae_loss, metric1, metric2, metric3)

    return metrics

# Training function. Inputs training dataloader, validation dataloader, h and w values (shape of image),
# size of the z_vector (512), model type, epochs of training and loss function. Trains the model, saves 
# training and testing loss for each epoch, saves the parameters for the best model and the last model.

def train(train_ds,val_ds,h,w,z_dim,mtype,epochs,loss):
    # Creates encoder & decoder models.

    encoder = Encoder(h,w,z_dim=z_dim,model=mtype)
    decoder = Decoder(h,w,z_dim=int(z_dim/2),model=mtype)

    encoder = nn.DataParallel(encoder).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    # Sets up the optimizer
    optimizer = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)

    # Initialize the logger
    writer = open(tensor_path,'w')
    writer.write('Epoch, Train_loss, Val_loss, SSIM, MSE, MAE'+'\n')

    step = 0
    best_loss = 10000

    # Trains for all epochs
    for epoch in range(epochs):
        print('-'*15)
        print(f'epoch {epoch+1}/{epochs}')
        encoder.train()
        decoder.train()

        ae_loss_epoch = 0.0

        for data in train_ds:
            img = data.to(device)

            z = encoder(img)
            x_recon = decoder(z)

            ed_loss = loss(x_recon,img)

            optimizer.zero_grad()
            ed_loss.backward()
            optimizer.step()

            ae_loss_epoch += ed_loss.item()
            
            step +=1

        tr_loss = ae_loss_epoch / len(train_ds)
        metrics = validation(val_ds, encoder, decoder, loss)
        val_loss = metrics[0].item()

        print('train_loss: {:.4f}'.format(tr_loss))
        print('val_loss: {:.4f}'.format(val_loss))

        writer.write(str(epoch+1) + ', ' + str(tr_loss)+ ', ' + str(val_loss)+ ', ' + 
                     str(metrics[1].item())+ ', ' + str(metrics[2].item())+ ', ' + 
                     str(metrics[3].item()) + '\n')

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, model_path + f'/encoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, model_path + f'/decoder_{epoch + 1}.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, model_path + f'/encoder_best.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, model_path + f'/decoder_best.pth')
            print(f'saved best model in epoch: {epoch+1}')
    
    writer.close()


# Main code
if __name__ == '__main__':

# The code first parses through input arguments --model_type, --model_view, --gpu, --epochs, --loss, --batch.
# Model type: default or residual (for now). Which model is it going to train.
# Model view: which view is the model getting train to (L=saggital,A=frontal,S=axial)
# GPU: Defines which GPU to use
# Epochs: How many epochs to train the model for
# Loss: Which loss function to implement
# Batch: Batch size for training

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type',
        dest='type',
        choices=['default', 'residual', 'self-attention','full'],
        required=True,
        help='''
        Type of model to train. Available options:
        "defalut" Default VAE using convolution blocks
        "residual: VAE which adds residual blocks between convolutions''')
    
    parser.add_argument('--model_view',
        dest='view',
        choices=['L', 'A', 'S'],
        required=True,
        help='''
        The view of the image input for the model. Options:
        "L" Left view
        "A" Axial view
        "S" Sagittal view''')
    
    parser.add_argument('--gpu',
        dest='gpu',
        choices=['0', '1', '2'],
        required=True,
        help='''
        The GPU that will be used for training. Terminals have the following options:
        Hanyang: 0, 1
        Busan: 0, 1, 2
        Sejong 0, 1, 2
        Songpa 0, 1
        Gangnam 0, 1
        ''')
    
    parser.add_argument('--epochs',
        dest='epochs',
        type=int,
        default=50,
        choices=range(1, 15000),
        required=False,
        help='''
        Number of epochs for training.
        ''')
    
    parser.add_argument('--loss',
        dest='loss',
        default='SSIM',
        choices=['L2', 'SSIM', 'MS_SSIM', 'Mixed'],
        required=False,
        help='''
        Loss function:
        L2 = Mean square error.
        SSIM = Structural similarity index.
        ''')

    parser.add_argument('--batch',
        dest='batch',
        type=int,
        default=1,
        choices=range(1, 512),
        required=False,
        help='''
        Number of batch size.
        ''')

    args = parser.parse_args()

    print(args)
    print('-'*25)


    model = args.type
    view = args.view
    gpu = args.gpu
    epochs = args.epochs
    batch_size = args.batch
    loss_type = args.loss
    z_dim = 512                 # Dimension of parameters for latent vector (latent vector size = z_dim/2)

# Connect to GPU

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print('GPU was correctly assigned.')
    print('-'*25)

# Define paths for obtaining dataset and saving models and results.
    path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

    source_path = path + 'healthy_dataset/'

    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = path + 'Results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    folder_name = "/Relu_{0}_{1}_AE_{2}_b{3}_{4}".format(view,model,loss_type,batch_size,date)
    tensor_path = results_path + folder_name + '/history.txt'
    model_path = results_path + folder_name + '/Saved_models/'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(model_path)
    
    print('Directories and paths are correctly initialized.')
    print('-'*25)

    print('Initializing loss function.')
    print('-'*25)

# Defining the loss function. Either L2 or SSIM (for now).

    if loss_type == 'L2':
        loss = nn.MSELoss()
    elif loss_type == 'SSIM':
        loss = SSIM_Loss(data_range=1.0, win_size = 11, size_average=True, channel=1)
    elif loss_type == 'MS_SSIM':
        loss = MS_SSIM_Loss(data_range=1.0, win_size = 5, size_average=True, channel=1)
    elif loss_type == 'Mixed':
        loss = Mixed(data_range=1.0, win_size = 11, size_average=True, channel=1)

# Define h and w (shape of the images), change depending on the view.
    if view == 'L':
        h = 158
        w = 126
        ids = np.arange(start=40,stop=70)
    elif view == 'A':
        h = 110
        w = 126
        ids = np.arange(start=64,stop=94)
    else:
        h = 110
        w = 158
        ids = np.arange(start=48,stop=78)

    print(f"h={h}, w={w}")
    print()

    print('Loading data.')
    print('-'*25)

# Begin the initialization of the datasets. Creates dataset iterativey for each subject and
# concatenates them together for both training and testing datasets (implements img_dataset class).

    train_id = os.listdir(source_path+'train/')
    test_id = os.listdir(source_path+'test/')

    train_set = img_dataset(source_path+'train/'+train_id[0], view)
    train_set = Subset(train_set,ids)
    test_set = img_dataset(source_path+'test/'+test_id[0],view)
    test_set = Subset(test_set,ids)

    for idx,image in enumerate(train_id):
        if idx != 0:
            train_path = source_path + 'train/' + image
            tr_set = img_dataset(train_path,view)
            tr_set = Subset(tr_set,ids)
            train_set = torch.utils.data.ConcatDataset([train_set, tr_set])

    for idx,image in enumerate(test_id):
        if idx != 0:
            test_path = source_path + 'test/' + image
            ts_set = img_dataset(test_path,view)
            ts_set = Subset(ts_set,ids)
            test_set = torch.utils.data.ConcatDataset([test_set, ts_set])

# Dataloaders generated from datasets 
    train_final = DataLoader(train_set, shuffle=True, batch_size=batch_size,num_workers=12)
    val_final = DataLoader(test_set, shuffle=True, batch_size=batch_size,num_workers=12)

    print('Data has been properly loaded.')
    print('-'*25)


    print('Beginning training.')
    print('.'*50)

# Conducts training
    train(train_final,val_final,h,w,z_dim,model,epochs,loss)
