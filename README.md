# ------------------- IN PROGRESS ---------------------------------

# Birds Image Recognition Task - Kaggle Dataset

The aim of this project is to identify 200 different species of birds from 224x224x3 pixel images. This is an image recognition task and there is no detection or segmentation done on this data. 

I chose this project to develop several skills including: 

- Further experience with the Keras and Tensorflow machine learning libraries.

  - Utilising both in-built and custom callbacks during training 
  - Parameter tuning using keras tuner
  - Transfer learning with a variety of pre-trained models including VGG-16, Xception and Resnet architectures. 
  - Using Tensorboard to evaluate training 'on the fly'. 

- Understand and visualise the 'black box' inside neural networks. 

- Evaluate image recognition tasks

- Train models using GPUs. In this case Google Colab was used for free access to a GPU. 

# Table of Contents

<!--ts-->
* [Introduction to the Challenge](#introduction-to-the-challenge)
* [Data](#data)
* [Setup](#setup)
* [My Approach](#my-approach)
  * [Data Prep](#data-prep)
  * [Training](#training)
  * [Model Evaluation](#model-evaluation)
  * [Understanding the Model](#understanding-the-model)
* [Results](#results)
* [Final Thoughts ](#final-thoughts)
* [Author](#author)
* [License](#license)
* [Acknowledgments](#acknowledgments)
<!--te-->

# Introduction to the Challenge

# Data

224x224x3 RGB images are provided at https://www.kaggle.com/gpiosenka/100-bird-species. The data is pre-split into train, validation and test. However, I didn't use these pre-split folders due to the train to validation/test ratio being very small. Therefore I use the 'consolidated' folder. 

There is a total of 29503 images with 200 different classes or birds. I split the data into a train and test dataset split 80-20%. The number of images per class varies between 97 and 310 with a mean number of images at 148. Below is a histogram of the classes sorted by the number of images for the corresponding class. There is a folder per class so Keras' Directory Iterator can be used to batch the data. 

# Setup

I use Google Colab to train my models as this provides free access to a GPU. All the required libraries are available on Colab. Note some paths are absolute paths and may need editing. Any other notebooks are run on Jupyter Notebook with no additional setup required. 

# My Approach

I started by trying to build the best model from scratch as possible. Keras Tuner was used to test some different architectures, learning rates and batch sizes. I was confident that transfer learning was needed to get high levels of accuracy so I didn't spend too much time building my own architectures before turning to pre-trained models. 

A variety of model architectures trained on ImageNet were tested including AlexNet, VGG-16, Inception and Xception. Xception provided the best initial results on this data so this model was taken forward and different techniques of updating the weights were tested to best suit this task. Further details can be found in [the training section.](#training)

I devote time to evaluating the results and 'understanding the model'. I expand on this in sections [Model Evaluation](#model-evaluation) and [Understanding the Model](#understanding-the-model)

## Data Pre-processing

To date, minimal pre-processing is done. I use the Keras Data Generator class to batch and augment the data. Below are the steps I apply to the data before training: 

- Split the data 80%-20% using a random seed of 42 for repeatability.  
- Pixels are rescaled between 0 and 1. 
- Random horizontal flip
- Shuffled (training data only)
- *Brightness Augmentation - Not currently applied*
- *Zoom Augmentation - Not currently applied* 


## Training 

## Model Evaluation 

## Understanding the Model

# Results

# Final Thoughts

# Author

# License

# Acknowledgements





