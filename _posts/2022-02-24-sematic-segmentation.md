---
layout: post
title: Semantic Segmentation with U-Net
author: Artan Zandian
date: Feb 24, 2022
excerpt: "In a second phase of my Neural Style Transfer model, I have applied Semantic Segmentation to exclude person figures from the style transfer. In this post, I will be going over the U-Net architecture and the project pipeline used to achieve this."
---

In a second phase of my Neural Style Transfer model (see [project](https://artanzand.github.io//neural-style-transfer/) and [repo](https://github.com/artanzand/neural_style_transfer)), I have applied Semantic Segmentation to exclude person figures from the style transfer (see [repo] for reproducible code). In this post, I will be going over the U-Net architecture and the project pipeline used to achieve this. This project is inspired by a stylized image of a child riding a bicycle which I came across with in a research paper ([Kasten et al. (2021)](https://layered-neural-atlases.github.io/)) by Adobe Research team.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/Adobe_stylized.jpg?raw=True" width=600></center>
<br>

# Motivation

So far, we have been able to created a stylized version of an image using Neural style Transfer (see this [post](https://artanzand.github.io//neural-style-transfer/) for walkthrough of that project). This works perfect if the intention is to stylize the whole image and where the partial details of the original image do not matter. A simple use case of this is for stylizing a scenary image. However, as shown in the example image below when human figures appear in a "content" image, the results of the generated stylized image are not satisfactory. To overcome this issue, our neural network pipeline needs to somehow learn where the people are in an image and cut them out.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/nst_problem.JPG?raw=True"></center>
<br>

# General Guideline

<br>
Our main objective for this project is to find masks which we would help us isolate the figure image from its background and vice versa. Masks are defined as tensors of 0's and 1's in this context, and when multiplied by by an image tensor produce our desired cutouts. For our purpose we will need two masks were one is the complement of the other. For example, if we have the figure mask as tensor Y, the background mask can be calculated by 1 - Y.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/masks.JPG?raw=True" width=400></center>
<br>

The below diagrams shows the overall pipeline. We have two neural networks. Neural Style Transfer is in charge of creating stylized images which has already been modeled in my [previous project](https://artanzand.github.io//neural-style-transfer/), and Semantic Segmentation network, which we will walk through in this post, will be generating tensors of zeros and ones. The outputs of Semantic segmentation will create the stylized bachground and the figure cutout when multiplied by the stylized and content image respectively. The final output is a summation of the two isolated pieces.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/NST_segmentation_framework.JPG?raw=True"></center>
<br>

# Semantic Image Segmentation

We will be building a special type of Convolutional Neural Networks (CNN) designed for quick and precise image segmentation. The predicted result of this network will be in shape of a label for every single pixel in an image. This is referred to as pixelwise prediction in the literature. This method of image classification is called semantic image classification and plays a critical role in self-driving cars which demand a perfect understanding of the surounding environment so that they avoid other cars and people, or change lanes.

A major source of confusion is the difference between object detection and image segmentation. The similarity is that both techniques intend to answer the question: "What objects are present in this image and what are the locations of thos objects?". The main difference is that the objective of object detection techniques like [YOLO](https://arxiv.org/abs/1506.02640) is to label objects by bounding boxes whereas semantic image segmentation aims to predict a pixel precise mask for each object in the image by labeling each pixel in the image with its corresponding class.

For reproducibility purposes and to allow for faster training of our model, I have opted for a dataset that would only label person figures. The technique can, however, be generalized to image segmentation for multiple classes, and in my post I will be referring to the respective changes in the general code and concept.

Create an image with multiple layers

<br>

# U-Net Architecture

U-Net, named after its U-shape, was originally created in 2015 for biomedical image segmentation (Ronneberger et al. (2015) [paper](https://arxiv.org/abs/1505.04597)), but soon became very popular for other semantic segmentation tasks. Note that due to computational constraints I have reduced the number of filters and steps used in the original paper.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/U-net.JPG?raw=True"></center>
<caption><center>U-Net Architecture used for this project</center></caption>

U-Net is based on the Fully Convolutional Network (FCN) which is comprised of two main sections; an encoder section which downsamples the input image (similar to a typical CNN network), and a decoder section which replaces the dense layer found in the output layer of CNN with transposed convolution layer (see diagram below on how transposed convolutions work) that upsample the feature map back to the size of the original input image. A major downfall of an FCN is that the final feature layer of the FCN suffers from information loss due to excessive downsampling which are required for inference. Solving this issue is necessary because the dense layers destroy spatial information which is the most essential part of image segmentation tasks.  

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/transpose_conv.gif?raw=True"></center>
<caption><center>Transpose Convolution ([image reference](https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1))</center></caption>  
<br>

U-Net improves on the FCN by introducing skip connections shown by green arrows in the U-Net diagram which are critical for preserving the spatial information. Further to that and instead of one transposed convolution at the end of the network, U-Net uses a matching number of transposed convolutions for upsampling (decoding) to those used for downsampling (encoding). This will map the encoded image back up to the original input image size. The role of skip connections is to retain information that would otherwise become lost during encoding. Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while also reducing computation cost.

To build the U-Net model we need two blocks. Each step in the U-Net diagram is considered a block.

1. Encoder block - A series of convolution layers followed by maxpooling
2. Decoder block - A transposed convolution layer followed by convolution layers
<br>

## Encoder Block

<br>

## Decoder Block

<br>

## Putting it together

<br>

# Neural Style Transfer + Segmentation

The final ingredient
<br>

# Final Thoughts

process improvement for computation efficiency

## References

[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.: [link to paper](https://arxiv.org/abs/1505.04597)  
[2] Kasten, Yoni, et al. "Layered neural atlases for consistent video editing." ACM Transactions on Graphics (TOG) 40.6 (2021): [link to paper](https://arxiv.org/pdf/2109.11418.pdf)  
[3] [DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes  
[4] Image Segmentation with DeepLabV3Plus: [link to repo](https://github.com/nikhilroxtomar/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow)
[5] Style image [reference](https://androidphototips.com/5-best-painting-apps-that-turn-your-iphone-photos-into-paintings/)
