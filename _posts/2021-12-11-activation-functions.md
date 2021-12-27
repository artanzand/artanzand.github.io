---
layout: post
title: Activation Functions - Which One to Use When
author: Artan Zandian
date: Dec 11, 2021
---

In this post I will be going over various activation functions to explain their use cases and where to apply them in our design of a neural network.  

The [activation function](https://en.wikipedia.org/wiki/Activation_function) `g(z)` defines how the weighted sum of the input `z` for each layer is transformed to a range that is acceptable as an input for the next layer (or prediction output in case of the last layer). Activation functions come in many flavours with the majority of them being non-linear. We will discuss the four popular ones (Sigmoid, tahn, ReLU and Leaky ReLU) as well as softmax and linear activation for the output layer. Since the type of prediction problem also defines the loss function to be used, I will briefly mention the respective loss function in [TensorFlow](https://www.tensorflow.org/) when exploring the activation functions.




The key takeaways from this post are:
- The activation function selection is highly dependant on whether we are applying them to a hidden layer or the output layer
- The type of prediction problem limits the type of activation functions as well as the loss function to use for the output layer.
- The best starting point for activation functions in design of neural networks  
<br>
## Hidden or Output?
The type of the activation function in a hidden layer controls how our model learns the weights and biases from the training dataset. For this reason, we are more interested on the performance (how well it learns the parameters) and speed (how fast each epoch will take).  

|	                     |ReLU  |Leaky ReLU |Sigmoid  |tanh |
|------------------------|------|-----------|---------|-----|
|Fast to compute?        |YES	|YES        |NO       |NO   |
|Simple derivative?      |YES	|YES        |YES      |YES  |
|Continuous?             |NO    |NO         |YES      |YES  |
|Region of uncertainty?  |YES   |YES        |YES      |YES  |
|Directional derivative? |YES   |YES        |YES      |YES  |
* inspired by [Williams, David P. (2019)](https://www.cs.ryerson.ca/~aharley/neural-networks/)[1]

On the other hand, the activation function in the output (last) layer controls the translation of the results to the type of prediction that we expect the model to make (e.g. classification or regression).

|	                        |Linear |Sigmoid |Softmax |
|---------------------------|-------|--------|--------|
|Regression                 |YES	|NO      |NO      |
|Two-class Classification   |NO  	|YES     |NO      |
|Multi-class Classification |NO     |YES     |YES     |

## Activation Fuction for Hidden Layers
### Sigmoid

### tanh

### ReLU

### Leaky ReLU





## Activation Fuction for Output Layer


<br>
<br>
<br>
_References:_  

[[1](https://www.cs.ryerson.ca/~aharley/neural-networks/)] Williams, David P. "Demystifying deep convolutional neural networks for sonar image classification." (2019).