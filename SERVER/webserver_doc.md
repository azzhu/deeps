
<div align='center'>

![logo](../imgs/logo2.jpg)
</div>


A deep learning framework of image processing with single pair of training images

---

This tutorial is mainly to introduce some key features and usages of **DeepS**, including: 

- How to inference your data;
- How to train new model;
- How to use new trained model to inference data;
- How to download pre-trained and new trained model;
- How to download software program to perform inferring on local computers and so on.
 
## <font color=blue> Introduction  </font>

### Home

Brief introduction to **DeepS**.

### Demo

The results of the **DeepS** demo are shown.

### Super Resolution and Optical Section

There are two function models in this page: ***'Training'*** (train your data) & ***'Inference'*** (inference your data). The ***'Training'*** module is shown in the figure below: 

![train](imgs/train.jpg "train")

- Do not save my data

Generally, we hope you can share your data to enhance our model performance. But if you don't want to share, we must respect your data privacy and delete it in our server timely. And please remember to check this option to tell us don't save your data. 

- Choose my data file and choose my label file

Choose one paired images which can be trained. In order to achieve better training effect, it is recommended that you upload images with higher resolution. We support resolutions from '512x512' to '10000x10000'. It should be noted that the training data must be strictly aligned. If the training data is not aligned, then the training will be meaningless.

- Add more training data and remove last training data

By default, only a pair of training data can be selected, but if you want better experimental results and you have more data, it is recommended that you manage (***Add more training data or Remove last training data***) more training data with this option.

- Example

If you don't have training data, but still want to try the training process, you can check this option and use the sample training data we provided. Note that using the sample training data to train the model  can't increase model performance, because the data was included in the training set of the current pre-training model.

- Run

Click the button to start training. In order to solve the training problem of the model in the small sample case, we use the *transfer learning* method. Your data will start training on the basis of the model we have pre-trained previously.

The schematic diagram of the ***'Inference module'*** is shown below.

![inference](imgs/inference.jpg)

- Do not save my data

If you do not want us to keep your data, please check this option. 

- Just use unet

We used two model architecture, 'deeps' and 'unet', and these two architecture were trained to have their own characteristics. We recommend that you use '***deeps***' (don't check this option). But if '***deeps***' can't reach your expectations, you can also try '***unet***' model. Good Luck.

- Use personal trained set

If you have trained the new model with your own data in the previous step, you can check this option here to see the effect of the model you have trained. If this option is not checked, the default model (that is, the model we pre-trained) is used for inferring.

- Choose my image

Select your own data. If you check the following example, the sample data will be used. 

- Run

Start to execute the inferring. Please wait. 

## <font color=blue> What do you want to do?

### Use pre-trained model to inference your data 

1. Home page 
2. Chick on 'Super Resolution' or 'Optical Section' 
3. Choose 'Inference' 
4. Click on 'Choose my image'
5. Click on 'Run' 

### Train a new model based on your own data

1. Home page 
2. Chick on 'Super Resolution' or 'Optical Section' 
3. Choose 'Train' 
4. Choose my data file & choose my label file
5. Chick on 'Run' 

### Use new trained models to inference new data

1. Choose 'Inference' 
2. Choose 'Use personal trained set' 
3. Chick on 'Choose my image' 
5. Chick on 'Run' 

### Download model 

You can download the model by clicking on the link provided after the training. 

### Use the downloaded model to run on your local computer

Please see this for more information: https://github.com/azzhu/Deeps_Inference_Package/blob/master/README.md



## Useful Links

💜 Deeps homepage: http://deeps.cibr.ac.cn/

💜 Deeps documentation: https://github.com/azzhu/deeps/blob/master/SERVER/webserver_doc.md

💜 Deeps repository: https://github.com/azzhu/deeps

💜 Deeps inference package repository: https://github.com/azzhu/Deeps_Inference_Package

💜 CIBR homepage: http://www.cibr.ac.cn/