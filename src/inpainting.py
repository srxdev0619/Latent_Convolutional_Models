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
import skimage.io
os.environ["CUDA_VISIBLE_DEVICES"]="0"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



RESTRICT = True
RESTRICT_VAL = 0.01
L2_W = 0.0
LATENT_NET_NAME = 'Latent4LSND'
LOG_DIR = '../runs/CelebA50k_Inpaint_Center'
IMG_DIR = "../" + LOG_DIR.split("/")[-1] + "/"
SEED = 42

MODEL_NAME = 'CelebA150k_LCM'

MODEL_NUM = 50000
LATENT_DIR = '../' + MODEL_NAME + '_latentDIR/'

#Defining and loading the model
G = EncDecCelebA(in_channels=64)
G = G.to(device)

G.load_state_dict(torch.load('../models/CelebA/' + MODEL_NAME + str(MODEL_NUM)))
s_np = np.load(MODEL_NAME + "_commoninput.npy")
s = torch.from_numpy(s_np)
s = s.to(device)


img_size = 128
hole_size = 50

dataset_folder='../celebA_split/test/'


data_reader = DataReader_Disk(dataset_folder=dataset_folder,
                              device=device,
                              to_shuffle=True,
                              img_size=img_size,
                              model_name=MODEL_NAME)
data_reader.load(latent_net_name=LATENT_NET_NAME, num_load=50, same_seed=SEED)
writer = SummaryWriter(log_dir = LOG_DIR)

#Generate the mask
def make_mask(shape, hole_size, extreme=False, seed=42):
    mask = torch.ones(shape)
    w_mask = torch.ones(shape)
    batch_size = shape[0]
    x_max = shape[2]
    y_max = shape[3]
    np.random.seed(seed)
    for i in range(batch_size):
        start_x = int((x_max/2) - (hole_size/2.0))
        start_y = int((y_max/2.0) - (hole_size/2.0))
        w_start_x = max(0, start_x - 5)
        w_end_x = min(x_max, start_x+hole_size+5)
        w_start_y = max(0, start_y - 5)
        w_end_y = min(y_max, start_y+hole_size+5)
        if extreme:
            mask[:,:,start_x:start_x+hole_size, start_y:start_y+hole_size] = 0.0
            w_mask[:,:,w_start_x:w_end_x, w_start_y:w_end_y] = 5.0
        else:
            mask[i,:,start_x:start_x+hole_size, start_y:start_y+hole_size] = 0.0
            w_mask[i,:,w_start_x:w_end_x, w_start_y:w_end_y]=5.0
    return mask, w_mask


def inpaint(s, num_epochs=100):
    G.eval()
    for p in G.parameters():
        p.requires_grad=False
    count = 0
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    batch_size = 50
    for _ in range(1):
        data_in,net_in,_  = data_reader.get_batch_from(100, batch_size=batch_size)
        writer.add_image('Org_image', data_in, count)
        
        for img_num in range(batch_size):
            skimage.io.imsave(IMG_DIR + 'Org_image' + str(img_num)+".png", data_in[img_num].numpy().transpose(1,2,0))

        gen_mask, w_mask = make_mask(data_in.shape, hole_size, extreme=False, seed=SEED)
        gen_mask = gen_mask.to(device)
        w_mask = w_mask.to(device)
        BATCH_SIZE = batch_size
        writer.add_image('Hole_Image', gen_mask*data_in, 0)
        nets_params = []
        for i in range(BATCH_SIZE):
            for p in net_in[i].parameters():
                p.requires_grad=True
            nets_params += list(net_in[i].parameters())
        optim_nets = optim.SGD(nets_params, lr=1.0, weight_decay=L2_W) #1.0, 0.5
        for ep in range(num_epochs):
            optim_nets.zero_grad()
            map_out_lst = []
            for i in range(BATCH_SIZE):
                m_out = net_in[i](s)
                map_out_lst.append(m_out)
            map_out = torch.cat(map_out_lst, 0)
            g_out = G(map_out)
            lap_loss = laploss(gen_mask*g_out, gen_mask*data_in)
            mse_loss = F.mse_loss(w_mask*gen_mask*g_out, w_mask*gen_mask*data_in)
            loss = mse_loss + lap_loss
            loss.backward()
            optim_nets.step()
            if RESTRICT:
                val = RESTRICT_VAL
                for i in range(BATCH_SIZE):
                    net_in[i].restrict(-val, val)
            writer.add_scalar('ep_z_loss', loss.data, ep)
            writer.add_scalar('ep_z_loss_Lap', lap_loss.data.item(), ep)
            writer.add_scalar('ep_z_loss_MSE', mse_loss.data.item(), ep)
            if ep%100 == 0:
                writer.add_image('Inpainted_Image_ep', g_out, ep)
                writer.add_image('latent_Z', map_out[:10,:3,:,:].data.cpu(), ep)

        map_out_lst = []
        for i in range(BATCH_SIZE):
            m_out = net_in[i](s)
            map_out_lst.append(m_out)
        map_out = torch.cat(map_out_lst, 0)
        g_pred = G(map_out)
        writer.add_scalar('z_loss', loss.data.item(), count)
        writer.add_image('Inpainted_Image', g_pred, count)
        for img_num in range(batch_size):
            skimage.io.imsave(IMG_DIR + 'Hole_Image' + str(img_num) +".png", (gen_mask*data_in)[img_num].data.cpu().numpy().transpose(1,2,0))
            skimage.io.imsave(IMG_DIR + 'Inpainted_Image' + str(img_num) + ".png", g_pred[img_num].data.cpu().numpy().transpose(1,2,0))




inpaint(s, 3001)
writer.close()
