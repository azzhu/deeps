# Deep learning optical-sectioning method

### Input and Output
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