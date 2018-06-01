# Latent Convolutional Models


![Img1](Sample_Restorations.png)
*Sample resotrations using a Latent Convolutional Model.*


Latent Convolutional Models work by parametrizing the latent space of a generator using convolutional neural networks. A schematic can be found below


![Img2](NormNet_Paper.png)
*The Schematic of a Latent Convolutional Model. The smaller ConvNet **f** (red) is unique to each image is parametrize the latent space of the generator **g_theta** (magenta) which is common to all images. The input **s** is fixed to random noise and is not updated during the training process.*


## Installation Dependencies
	1. numpy 1.14.3
	2. pytorch 0.4.0
	3. [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)



