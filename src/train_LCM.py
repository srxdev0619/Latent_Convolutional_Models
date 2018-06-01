import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from data_loader import *
from models import *
from custom_losses import *
from helper_funcs import *
import itertools
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


IMG_SIZE = 128
S_SIZE = 40
BATCH_SIZE = 10 #was 10
LATENT_NET_NAME = 'Latent4LSND'
RESTRICT = True
L2_W = 0.0
RESTRICT_VAL = 0.01
fname = "train_LCM"
NUM_TRAIN_SAMPLES = 150000

LOG_DIR = '../runs/LCM_CelebA/'

MODEL_NAME = 'CelebA150k_LCM'

LATENT_CHECK_DIR = '../' + MODEL_NAME + '_latentDIR/'

if not os.path.exists(LATENT_CHECK_DIR):
            os.makedirs(LATENT_CHECK_DIR)

PREV_EPOCH = None
NUM_EPOCHS = 150000
SAVE_EVERY = 15000

#The Generator Network
G = EncDecCelebA(in_channels=64)
G = G.to(device)

#Uniform noise
s =  get_noise(input_depth=2, method='noise', spatial_size=[S_SIZE, S_SIZE], batch_size=1)

np.save(MODEL_NAME + '_commoninput', s.data.numpy())


writer = SummaryWriter(log_dir=LOG_DIR)

dataset_folder = '../celebA_split/train/'

data_reader = DataReader_Disk(dataset_folder=dataset_folder,
                              device=device,
                              to_shuffle=True,
                              img_size=IMG_SIZE,
                              model_name=MODEL_NAME)


if PREV_EPOCH:
    s_np = np.load(MODEL_NAME + "_commoninput.npy")
    s.data = torch.from_numpy(s_np)
    START_EPOCH = PREV_EPOCH + 1
    END_EPOCH = START_EPOCH + NUM_EPOCHS
    data_reader.load(latent_net_name=LATENT_NET_NAME,
                     num_epoch=PREV_EPOCH,
                     saved_model_name=MODEL_NAME,
                     num_load=NUM_TRAIN_SAMPLES,
                     latent_dir=LATENT_CHECK_DIR)
    G.load_state_dict(torch.load('../models/CelebA/' + MODEL_NAME + str(PREV_EPOCH)))
else:
    START_EPOCH = 0
    END_EPOCH = START_EPOCH + NUM_EPOCHS
    data_reader.load(latent_net_name=LATENT_NET_NAME,
                     num_load=NUM_TRAIN_SAMPLES)
s = s.to(device)
G_optimizer = optim.SGD(G.parameters(), lr=1.0)



def trainZG(epoch, data_in, net_in, num_epochs=100):
    """
     Jointly trains the latent networks and the generator network.
     """
    G.train()
    for p in G.parameters():
        p.requires_grad=True
    BATCH_SIZE = len(net_in)
    nets_params = []
    for i in range(BATCH_SIZE):
        for p in net_in[i].parameters():
            p.requires_grad=True
        nets_params += list(net_in[i].parameters())
    optim_nets = optim.SGD(nets_params, lr=1.0, weight_decay=L2_W)
    for ep in range(num_epochs):
        G_optimizer.zero_grad()
        optim_nets.zero_grad()
        map_out_lst = []
        for i in range(BATCH_SIZE):
            m_out = net_in[i](s)
            map_out_lst.append(m_out)
        map_out = torch.cat(map_out_lst, 0)
        g_out = G(map_out)
        lap_loss = laploss(g_out, data_in)
        mse_loss = F.mse_loss(g_out, data_in)
        loss = mse_loss + lap_loss
        loss.backward()
        optim_nets.step()
        G_optimizer.step()
        if RESTRICT:
            val = RESTRICT_VAL
            for i in range(BATCH_SIZE):
                net_in[i].restrict(-1.0*val, val)
    optim_nets.zero_grad()
    G_optimizer.zero_grad()
    if epoch%10 == 0:
        G.eval()
        map_out_lst = []
        for i in range(BATCH_SIZE):
            m_out = net_in[i](s)
            map_out_lst.append(m_out)
        map_out = torch.cat(map_out_lst, 0)
        g_out = G(map_out)
        writer.add_scalar('Z_loss', loss.data.item(), epoch)
        writer.add_scalar('Z_loss_lap', lap_loss.data.item(), epoch)
        writer.add_scalar('Z_loss_MSE', mse_loss.data.item(), epoch)
        if epoch%100 == 0:
            writer.add_image('Real_Images', data_in[:5].data.cpu(), epoch)
            writer.add_image('Generated_Images_Z', g_out[:5].data.cpu(), epoch)
            writer.add_image('latent_Z', map_out[:10,:3,:,:].data.cpu(), epoch)

    return net_in



for epoch in range(START_EPOCH, END_EPOCH + 1):
    #Get a batch of images, their latent networks and corresponding network ids
    data_in, latent_nets, latent_net_ids = data_reader.get_batch(batch_size=BATCH_SIZE)

    #train the latent networks and generator
    latent_nets = trainZG(epoch, data_in, latent_nets, num_epochs=50)

    #update the latent networks
    data_reader.update_state(latent_nets, latent_net_ids)
    print(fname + " Epoch: ", epoch)
    if epoch%SAVE_EVERY == 0:
        if epoch > 0:
            data_reader.save_latent_net(name=MODEL_NAME + "_latentnet_" +str(epoch) + "_", latent_dir=LATENT_CHECK_DIR)
            torch.save(G.state_dict(), '../models/CelebA/' + MODEL_NAME + str(epoch))
writer.close()
