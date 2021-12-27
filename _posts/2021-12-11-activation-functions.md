---
layout: post
title: Activation Functions - Which One to Use When
author: Artan Zandian
date: Dec 11, 2021
---

In this post I will be going over various activation functions to explain their use cases and where to apply them in our design of a neural network.  

The [activation function](https://en.wikipedia.org/wiki/Activation_function) `g(z)` defines how the weighted sum of the input `z` for each layer is transformed to a range that is acceptable as an input for the next layer (or prediction output in case of the last layer). Activation functions come in many flavours with the majority of them being non-linear. We will discuss the four popular ones (Sigmoid, tahn, ReLU and Leaky ReLU) as well as softmax and linear activation for the output layer.

|	|ReLU   |Leaky ReLU  |Sigmoid  |tanh   |
|-----|---------|----------|---------|-------|
Fast to compute?|	YES	|YES |	NO |	NO |
|Simple derivative?|	YES	|YES|	YES |	YES|
|Continuous?|	NO|	NO |	YES|	YES |
|Region of uncertainty?  |	NO |	YES |	YES|	YES|
|Directional derivative? |	NO |	YES |	YES|	YES|
* inspired by [Williams, David P. (2019)](https://www.cs.ryerson.ca/~aharley/neural-networks/)


The key takeaways from this post are:
- The activation function selection is highly dependant on whether we are applying them to a hidden layer or the output layer
- The type of prediction problem limits the type of activation functions to use for the output layer.
- The best starting point for activation functions in design of neural networks

## Hidden or Output?




<br>
_References:_  

[[1](https://www.cs.ryerson.ca/~aharley/neural-networks/)] Williams, David P. "Demystifying deep convolutional neural networks for sonar image classification." (2019).