#Latent Convolutional Models

<p>
<a href="http://shahrukhathar.github.io/" target="_blank">ShahRukh Athar</a>, 
<a href="http://sites.skoltech.ru/compvision/members/vilem/" target="_blank">Victor Lempitsky</a> and
<a href="https://faculty.skoltech.ru/people/evgenyburnaev" target="_blank">Evgeny Burnaev</a>
</p>


![](/docs/images/Sample_Restorations.png)
*Sample resotrations using a Latent Convolutional Model.*


Deep Convolutional Networks have made a lasting impact on the entire field of computer vision with great results on segmentation, recognition and generative modelling of images. Of late, there has been an increasing amount of work suggesting that the structure of the Convolutional Neural Network itself plays a rather important role in building unsupervised models of images and our work provides further evidence to support this claim. We present a new latent model of natural images that can be learned on large-scale datasets. The learning process provides a latent embedding for every image in the training dataset, as well as a deep convolutional network that maps the latent space to the image space. This latent space our model uses is relatively high-dimensional and is parametrized by convolutional neural networks; we call it the *convolutional manifold*.


## How does it work?

The Latent Convolutional Model (LCM) works by jointly training a generator and convolutional neural networks unique to each image. Given a dataset of images $$\mathcal{D} = \{x_{1}, x_{2}, ..., x_{M}\}$$ from some distribution $$\mathcal{X}$$, we pair with each image $$x_{i}$$ a latent CNN $$f_{\phi_{i}}$$ with parameters $$\phi_{i}$$. We now jointly optimize the $$\phi_{i}$$'s and the parameters $$\theta$$ of a generator $$G_{\theta}$$ for a minibatch of size $$N$$ as follows

$$
    \underset{\theta}{\text{min }} \frac{1}{N}\sum\left[\underset{\phi_{i}}{\text{min }} \mathcal{L}(x_{i},G_{\theta}(f_{\phi_{i}}(s)))\right]
$$

with additional constraints that $$\phi_{i} \in [-0.01, 0.01]^{N_{\phi}}$$. The loss function we use is the Laplacian-L1: $$\mathcal{L}(x_{1},x_{2})_\text{Lap-L1} = \sum_{j}2^{-2j}|L^{j}(x_{1} - x_{2})|_{1}$$ where $$L^{j}$$ is the $$j$$th level of the Laplacian image pyramid. To speed up convergence during training we also add the MSE loss to the Lap-L1 term.
We carry out the optimization above using vanilla stochastic gradient descent. As the model is learnt, each image, $$x_{i}$$, gets a representation $$z_{i} = f_{\phi_{i}}(s)$$ on the convolutional manifold.

![Img2](/docs/images/NormNet_Paper.png)
*The Schematic of a Latent Convolutional Model. The smaller ConvNet $$f_{\phi}$$ (red) is unique to each image is parametrize the latent space of the generator $$g_{\theta}$$ (magenta) which is common to all images. The input $$s$$ is fixed to random noise and is not updated during the training process.*

## How well does it work?

Given below are the results of inpainting and super-resolution using LCM and a few similar methods
![](/docs/images/results_celeba.png)
*Inpainting and Super-Resolution on CelebA at $$128\times{}128$$ resolution. The compared models are Generative Latent Space Optimization ([GLO](https://arxiv.org/abs/1707.05776)), Deep Image Prior ([DIP](https://dmitryulyanov.github.io/deep_image_prior)), Wasserstein GAN with Gradient Penalty ([WGAN](https://arxiv.org/abs/1704.00028)) and Autoencoders ([AE](http://www.deeplearningbook.org/contents/autoencoders.html)).*
