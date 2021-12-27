---
layout: post
title: Activation Functions - Which One to Use When
author: Artan Zandian
date: Dec 11, 2021
---

In this post I will be going over various activation functions to explain their use cases and where to apply them in our design of a neural network.  

The [activation function](https://en.wikipedia.org/wiki/Activation_function) `g(z)` defines how the weighted sum of the input `z` for each layer is transformed to a range that is acceptable as an input for the next layer (or prediction output in case of the last layer). Activation functions come in many flavours with the majority of them being non-linear. We will discuss the four common activation functions (Sigmoid, tahn, ReLU and Leaky ReLU) for hidden layers as well as Softmax and Linear (Identity) activation for the output layer. Since the type of prediction problem also defines the loss function to be used, I will briefly mention the respective loss function in [TensorFlow](https://www.tensorflow.org/) when exploring the activation functions.


The key takeaways from this post are:
- The activation function selection is highly dependant on whether we are applying them to a hidden layer or the output layer
- The type of prediction problem limits the type of activation functions as well as the loss function to use for the output layer.
- The best starting point for activation functions in design of neural networks  
<br>


## Hidden or Output?  
The type of the activation function in a hidden layer controls how our model learns the weights and biases from the training dataset. For this reason, we are more interested on the performance (how well it learns the parameters) and speed (how fast each epoch will take).  

|	                      |ReLU  |Leaky ReLU |Sigmoid  |tanh |
|-------------------------|------|-----------|---------|-----|
|Fast to compute?         |YES	 |YES        |NO       |NO   |
|Simple derivative?       |YES	 |YES        |YES      |YES  |
|Continuous?              |NO    |NO         |YES      |YES  |
|Region of uncertainty?   |YES   |YES        |YES      |YES  |
|Directional derivative?  |YES   |YES        |YES      |YES  |

 [1] inspired by [Williams, David P. (2019)](https://www.cs.ryerson.ca/~aharley/neural-networks/)
  

On the other hand, the activation function in the output (last) layer controls the translation of the results to the type of prediction that we expect the model to make (e.g. classification or regression).

|	                        |Linear  |Sigmoid  |Softmax |
|---------------------------|--------|---------|--------|
|Regression                 |YES	 |NO       |NO      |
|Two-class Classification   |NO  	 |YES      |NO      |
|Multi-class Classification |NO      |YES      |YES     |

## Activation Fuctions for Hidden Layers
In forward propagation we are using a linear function `(wX + b)` to calculate the `z` value for each node in a layer. Therefore, in order to allow the neural network to learn more complex patterns we will need a nonlinear activation function. This function should be differentiable to allow calculation of gradients in backpropagation.

image goes here with 4 functions
### Sigmoid
Also referred to as logistic function, this function resembles an S curve and maps the input values to a real value between 0 and 1. This function has a high computation cost but since it maps 0 to 0.5 and large positive and large negative numeers to 1 and 0 respectively, it is perfect for cases where the output is required to be translated to soft (percentage) or hard (0/1) prediction.  
$$sigmoid(z) = \frac{1}{1 + e^{-z}}$$

### Tanh
The Hyperbolic Tangent function or in short `Tanh` (read "Tan + H") looks like a stretched version of Sigmoid in a sense that it maps the input to a range betwee -1 and 1. The biggest advantange of tanh over sigmoid is that it is centerd around 0 (maps zero to zero), and because of that it typically performs better than sigmoid ([Deep Learning, 2016](https://www.deeplearningbook.org/)).
$$tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
### ReLU
Due to the simplicity and low computation cost, Rectified Linear Unit is the most commonly used activation function in deep neural networks (i.e. more than 50 layers). Another big advantage of ReLU is that unlike sigmoid and tanh it is less susceptible to the [vanishing gradients problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) which limit the training of deep networks.  
This function maps negative numbers to zero and applies an identity function to the positive numbers. 
### Leaky ReLU





## Activation Fuction for Output Layer


<br>
<br>
<br>
_References:_  

[[1](https://www.cs.ryerson.ca/~aharley/neural-networks/)] Williams, David P. "Demystifying deep convolutional neural networks for sonar image classification." (2019).