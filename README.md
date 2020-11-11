
<div align='center'>

![logo](imgs/logo.jpg)
</div>


A deep learning framework for image processing with single pair of training images

---

In biological studies, there is a huge demand to recover high-quality images from low quality images. Basically, two typical steps are implemented to denoise and then deconvolute. Traditional methods have been developed to deal with different senarios separately. For us, it’s a reverse process either decreasing or increasing information of images. Here, we have proposed DeepS to fulfill the reverse functions with the same deep learning framework.

![img](imgs/img.jpg)

This 

## Input and Output
input: wide-field image

output: optically sectioned image

### Neural Network Architecture
Using a generative adversarial network.

G: unet

D: cnn

### Loss Functions
MSE loss

Perceptive loss (VGG16)

Adversarial loss

**Latest updates:** 

💜 we havce

:purple_heart: DeepLabCut supports multi-animal pose estimation (BETA release, plese give us your feedback! `pip install deeplabcut==2.2b8`).

:purple_heart: We have a real-time package available! http://DLClive.deeplabcut.org