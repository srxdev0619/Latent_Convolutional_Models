import numpy as np
from tensorboardX import SummaryWriter
import scipy.misc
from os import listdir
from os.path import isfile, join
import random
import torch
import latent_models as lm
from skimage.io import imread, imsave
from skimage.transform import resize
import skimage.transform
import time
import os




class DataReader_Disk():

    """
      This is the class loads a dataset of images and associates with each image a corresponding
      latent ConvNet.
     """

    def __init__(self, dataset_folder='./patched_data_512/',
                 device=torch.device("cuda:0"),
                 img_size=64,
                 to_shuffle=False,
                 model_name=None):
        """
          Parameters:
                    dataset_folder (String): Path to the directory containing images to be trained on.
                    device (torch.device): Device onto which images and the latent ConvNets are to be loaded
                    img_size (int, optional): Size to which images are to be resized to. All images would be resized to (img_size, img_size). (Default 64)
                    to_shuffle (bool, optional): Set to True to shuffle the dataset. (Default: False)
                    model_name: The model name. This name would be used to create a temporary directory where intermediate latent ConvNets are stored
          """

        self.end = 0
        self.start = 0
        self.model_name = model_name
        self.device=device
        self.temp_latent_dir = './latent_temp_dir/' + self.model_name + '_latent_temp/'
        if not os.path.exists(self.temp_latent_dir):
            os.makedirs(self.temp_latent_dir)
        self.source_dir = dataset_folder
        self.all_imgs = [join(self.source_dir, f) for f in listdir(self.source_dir) if isfile(join(self.source_dir, f))]
        self.all_imgs.sort()
        if to_shuffle:
            random.seed(42)
            random.shuffle(self.all_imgs)
            random.seed()
        self.img_size = img_size

    def load(self, latent_net_name,
            num_epoch=None,
            saved_model_name=None,
            num_load=None,
            same_seed=42,
            latent_dir='./latent_nets_2/'):

        """
          This method loads with each image name a corresponding network_id. This network_id is used to load and save the appropriate networks.

          Parameters:
                    latent_net_name (String): Name of the module to be used as the latent ConvNet. The module must be defined in the
                                              latent_models.py file.
                    num_epoch (int, optional): Upload the latent nets of a model saved at `num_epoch`. Usually used to continue training (Default: None)
                    saved_model_name (String, optional): Name to the model whoose latent nets need to be uploaded. (Default: None)
                    num_load (int, optional): Number of images from `dataset_folder` to be used for training. If None all images are used. (Default: None)
                    same_seed (int, optional): Seed with which the nets are to be initialized. (Default: 42; Becasue atleast one Deep AI thought this was the answer to
                                               Ultimate Question of Life, the Universe and Everything.)
                    latent_dir (String, optional): Directory from which to load previously saved latent nets. This must be specified if `num_epoch` is not None. (Default: None)
          """

        self.data_lst = []
        self.data_idx = []
        self.num_samples = 0
        self.num_epoch = num_epoch
        self.latent_dir = latent_dir
        self.latent_net_name = latent_net_name
        if num_load is None:
            all_imgs_lst = self.all_imgs
        else:
            all_imgs_lst = self.all_imgs[:num_load]
        for img_name in all_imgs_lst:
            if same_seed:
                torch.manual_seed(same_seed)
            latent_net = getattr(lm, self.latent_net_name)()
            latent_net = latent_net.to(self.device)
            if num_epoch:
                print("Num prev Loaded: ", self.num_samples)
                latent_net.load_state_dict(torch.load(latent_dir + saved_model_name + "_latentnet_" + str(num_epoch) + '_' + str(self.num_samples)))
            torch.save(latent_net.state_dict(), self.temp_latent_dir + "temp_latent_net_" + str(self.num_samples))
            self.data_lst.append([img_name, self.num_samples])
            self.data_idx.append(np.random.randint(0, self.img_size - 20, size=(5,2)))
            self.num_samples += 1
            if num_epoch is None:
                print("Networks loaded: ", self.num_samples)
        print("Number of samples loaded: ", len(self.data_lst))

    def get_nets(self, net_ids):
        latent_nets = []
        for i in net_ids:
            latent_net = getattr(lm, self.latent_net_name)()
            latent_net = latent_net.to(self.device)
            latent_net.load_state_dict(torch.load(self.temp_latent_dir + "temp_latent_net_" + str(i)))
            latent_nets.append(latent_net)
        return latent_nets

    def save_nets(self, nets, net_ids):
        num_nets = len(net_ids)
        for i in range(num_nets):
            torch.save(nets[i].state_dict(), self.temp_latent_dir + "temp_latent_net_" + str(net_ids[i]))



    def get_imgs(self, img_list):
        data_out = None
        for img_name in img_list:
            inputs = imread(img_name)
            inputs = resize(inputs, (self.img_size, self.img_size))
            if len(inputs.shape) != 3:
                inputs = np.expand_dims(inputs, -1)
                inputs = np.repeat(inputs, 3, -1)
                if inputs.shape[-1] != 3:
                    raise ValueError("Input must have last dimension as 3")
            inputs = np.transpose(inputs, [2,0,1])
            inputs = np.expand_dims(inputs, 0)
            inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
            inputs = torch.from_numpy(inputs).float()
            if data_out is None:
                data_out = inputs
            else:
                data_out = torch.cat([data_out, inputs])
        return data_out

    def get_batch(self, batch_size=10):
        """
          This method gets the next batch of size specified by `batch_size`. If the counter has reached the end of dataset the batch size
          returned would be less than or equal to the specified `batch_size`.
          Parameters:
                    batch_size (int, optional): Batch size to be returned.
          """
        self.start = self.end
        start = self.start
        end = min(start + batch_size, self.num_samples)
        eff_batch = end - start
        if end >= self.num_samples:
            end = 0
        self.end = end
        data_out_lst = []
        latent_net_ids = []
        for i in range(eff_batch):
            data_out_lst.append(self.data_lst[start + i][0])
            latent_net_ids.append(self.data_lst[start + i][1])

        data_out = self.get_imgs(data_out_lst)
        latent_nets = self.get_nets(latent_net_ids)
        data_out = data_out.to(self.device)
        return data_out, latent_nets, latent_net_ids


    def get_batch_from(self, start, batch_size=10):
        """
          This method gets the next batch of size specified by `batch_size` starting from `start`. If the counter has reached the end of dataset the batch size
          returned would be less than or equal to the specified `batch_size`.
          Parameters:
                    start (int): The index from which to start the batch.
                    batch_size (int, optional): Batch size to be returned.
          """
        start = start
        end = min(start + batch_size, self.num_samples)
        eff_batch = end - start
        if end >= self.num_samples:
            end = 0
        data_out = None
        latent_nets_ids = []
        for i in range(eff_batch):
            if data_out is None:
                data_out = self.data_lst[start + i][0]
            else:
                data_out = torch.cat([data_out, self.data_lst[start + i][0]])

            latent_net_ids.append(self.data_lst[start + i][1])
        latent_nets = self.get_nets(latent_net_ids)
        return data_out, latent_nets, latent_nets_ids

    def update_state(self, latent_nets, latent_nets_ids):
        """
          Updates the latent networks during training.
          Parameters:
                    latent_nets (List): List of latent networks (nn.Modules) to be saved.
                    latent_nets_ids (List): List of ids of the latent networks to be saved.
          """
        self.save_nets(latent_nets, latent_nets_ids)

    def save_latent_net(self, latent_dir ,name):
        """
          Saves the latent networks in `latent_dir` to create a model checkpoint.
          Parameters:
                    latent_dir (String): Name of directory where the latent networks are saved.
                    name (String): Name with which the latent networks must be saved.
          """
        if not os.path.exists(latent_dir):
            os.makedirs(latent_dir)
        for i in range(self.num_samples):
            latent_net = self.get_nets([i])[0]
            torch.save(latent_net.state_dict(), latent_dir + name + str(i))

    def get_vec(self, idx):
        return self.data_lst[idx][1]
