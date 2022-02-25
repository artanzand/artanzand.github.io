---
layout: post
title: Semantic Segmentation with U-Net
author: Artan Zandian
date: Feb 24, 2022
excerpt: "In a second phase of my Neural Style Transfer model (see [project](https://artanzand.github.io//neural-style-transfer/) and [repo](https://github.com/artanzand/neural_style_transfer)), I have applied Semantic Segmentation to exclude person figures from the style transfer. In this post, I will be going over the U-Net architecture and the project pipeline used to achieve this."
---

In a second phase of my Neural Style Transfer model (see [project](https://artanzand.github.io//neural-style-transfer/) and [repo](https://github.com/artanzand/neural_style_transfer)), I have applied Semantic Segmentation to exclude person figures from the style transfer. In this post, I will be going over the U-Net architecture and the project pipeline used to achieve this. This project is inspired by a stylized image of a child riding a bicycle which I came across with in a research paper ([Kasten et. al (2021)](https://layered-neural-atlases.github.io/)) by Adobe Research team.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/Adobe_stylized.jpg?raw=True" width=400></center>
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

The below diagrams shows the overal pipeline. We have two neural networks. Neural Style Transfer is in charge of creating stylized images which has already been created in my [previous project](https://artanzand.github.io//neural-style-transfer/), and Semantic Segmentation network will be generating tensors of zeros and ones. The outputs of Semantic segmentation will create the stylized bachground and the figure cutout when multiplied by the stylized and content image respectively. The final output is a summation of the two isolated pieces.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/NST_segmentation_framework.JPG?raw=True"></center>
<br>

# Semantic Segmentation

<br>

# U-Net Architecture

<br>
image diagram

## Encoder Block

<br>

## Decoder Block

<br>

## Putting it together

<br>

# Neural Style Transfer + Segmentation

# Final Thoughts

process improvement for computation efficiency

## References

[1] Yoni Kasten, Dolev Ofri, Oliver Wang, Tali Dekel. "Layered Neural Atlases for Consistent Video Editing" SIGGRAPH Asia (2021): [link to paper](https://arxiv.org/pdf/2109.11418.pdf)  
