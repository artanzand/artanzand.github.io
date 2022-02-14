---
layout: post
title: Neural Style Transfer
author: Artan Zandian
date: Feb 10, 2022
excerpt: "Most of the supervised deep learning algorithms optimize a cost function to get a set of parameter values (weights). With Neural Style Transfer, we optimize a cost function to get pixel values of a generated image. This algorithm is, therefore, considered an unsupervised deep learning due to absence of a labeled target."
---

In this blog I will be explaining the concept of Neural Style Transfer, along with the calculations and highlevel code structure. For Tensorflow coding implementation details please refer to the project [repository](https://github.com/artanzand/neural_style_transfer).

# Motivation

My intention for doing this project was two-fold. First, as a deep learning enthusiast I wanted to explore if I can implement a research paper ([[Gatys et al. (2015).](https://arxiv.org/abs/1508.06576)](<https://arxiv.org/abs/1508.06576>)) in code to run on a GPU-accelerated machine (an NVIDIA product), and second, I wanted to have fun implementing a project. I have always been mezmorized by applications that would take an image and create a stylized version of it. With over 15 years of experience with graphics applications, the one thing that I knew for sure was that this is different from applying multiple filters on an image as these applications also changed the patterns of the input image. I knew that Neural Networks were somehow involved, but with my limited knowledge of deep learning I couldn't figure out how it is possible to create an output when no prediction is involved. It turned out that this can be done through unsupervised deep learning rather than usuall supervised learning that neural networks are known for.

An inspiring piece of work for my project was this [video](https://www.youtube.com/watch?v=4b9PYIxmcNc&t=800s&ab_channel=CGGeek) from CG Geek who created a realistic 3D rendering of Bob Ross's famous Summer painting. The overal goal for me was whether I would be able to do the same, but rather than being realistic I wanted to create a model that would stylize video frames of a real scenary into a painting. This allows you to walk through a painting and peek around! The first step towards this goal was clearly transfering an image into a stylized painting.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/summer.jpg?raw=True"></center>
<br>

# General Framework

Most of the supervised deep learning algorithms optimize a cost function to get a set of parameter values (weights). With Neural Style Transfer, we optimize a cost function to get pixel values of a generated image. This algorithm is, therefore, considered an unsupervised deep learning due to absence of a labeled target.
<br>

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/VGG10.PNG?raw=True"></center>
<caption><center>[1] VGG19 - Supervised Neural Network: ([Image source](https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0))</center></caption>  
<br>

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/NST_diagram.JPG?raw=True"></center>
<caption><center>[2] Neural Style Transfer - Unsupervised Learning: ([Image source](https://arxiv.org/abs/1508.06576))</center></caption>  
<br>

<br>

# Pieces

<br>

# Final Thoughts

To answer my earlier question in the Motivation section, here is the final result when I stylize my Moraine Lake photo with Bob Ross's Summer painting.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/moraine_style.JPG?raw=True"></center>
<br>

The biggest limitation of this technique is being computationally expensive. My implementation of Neural Style Transfer needs to run for over 10,000 epochs to generate a reasonable stylized image which would lead to longer run times. The output below, for example, took over 10 mins with 10,000 epochs on my [Nvidia Jetson Nano](https://artanzand.github.io//Setup-Jetson-Nano/). This would be very problematic for my ultimate goal of creating a moving Bob Ross painting! The two solutions that come into mind is to, first, reduce the frames per second for the video or alternatively create a GIF output instead of a video output, and second, which would require much more research, is whether I would be able to only calculate the extra pixels that appear on a new frame (compared to the previous one) and concatenate them with the already stylized pixels from the previous frame.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/moraine_GIF.gif?raw=True"></center>
<br>

## References

[1] Gatys, Leon A., Ecker, Alexander S. and Bethge, Matthias. "A Neural Algorithm of Artistic Style.." CoRR abs/1508.06576 (2015): [link to paper](https://arxiv.org/abs/1508.06576)  

[2] Athalye A., athalye2015neuralstyle, "Neural Style" (2015): [Repository](https://github.com/anishathalye/neural-style)  

[3] [DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes

[4] Bob Ross [painting](https://blog.twoinchbrush.com/article/paint-better-mountains-upgrade-your-titanium-white/)
