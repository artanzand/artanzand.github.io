---
layout: post
title: Choosing Activation Functions for Deep Learning
author: Artan Zandian
date: Dec 11, 2021
excerpt: 'In this post I will be going over various activation functions to explain their use cases and where to apply them in our design of neural networks.'
---

In this post I will be going over various activation functions to explain their use cases and where to apply them in our design of neural networks.  

The [activation function](https://en.wikipedia.org/wiki/Activation_function) `g(z)` defines how the weighted sum of the input `z` for each layer is transformed to a range that is acceptable as an input for the next layer (or the prediction output in case of the last layer). Activation functions come in many flavors with the majority of them being non-linear. We will discuss the four common activation functions (Sigmoid, Tahn, ReLU and Leaky ReLU) for hidden layers as well as Softmax and Linear (Identity) activation for the output layer. Since the type of prediction problem also defines the loss function to be used, I will briefly mention the respective loss function in a [TensorFlow](https://www.tensorflow.org/) code snippet when exploring the activation functions.


The key takeaways from this post are:
- The activation function selection is highly dependent on whether we are applying them to a hidden layer or the output layer.
- The type of prediction problem limits the choice of activation function as well as the loss function to be used for the output layer.
- The best starting point for activation functions in design of neural networks  
<br>


## Hidden or Output?  
The type of the activation function in a hidden layer controls how our model learns the weights and biases from the training dataset. For this reason, we are more interested on the performance (how well it learns the parameters) and speed (how fast each epoch will take).  

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/activation_table.PNG?raw=True"></center>
<caption><center>[1] inspired by Williams, David P. (2019)</center></caption>  
<br>
  

On the other hand, the activation function in the output (last) layer controls the translation of the results to the type of prediction that we expect the model to make (e.g. classification or regression).

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/output_activation.PNG?raw=True"></center>
<br>

## Activation Functions for Hidden Layers
In forward propagation we are using a linear function `(wX + b)` to calculate the `z` value for each node in a layer. Therefore, in order to allow the neural network to learn more complex patterns we will need a nonlinear activation function. This function should be differentiable to allow calculation of gradients in backpropagation.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/activations.PNG?raw=True"></center>


### Sigmoid
Also referred to as Logistic function, this function is an S curve and maps the input values to real values between 0 and 1. This function has a high computation cost but since it maps 0 to 0.5 and large positive and large negative numbers to 1 and 0 respectively, it is perfect for cases where the output is required to be translated to soft (percentage) or hard (0/1) predictions.  
 
$$sigmoid(z) = \frac{1}{1 + e^{-z}}$$


### Tanh
The Hyperbolic Tangent function or in short `Tanh` (read "Tan + H") resembles a stretched version of Sigmoid in a sense that it maps the input to a range between -1 and 1. The biggest advantage of tanh over sigmoid is that it is centered around 0 (maps zero to zero), and because of that it typically performs better than sigmoid ([Deep Learning, 2016](https://www.deeplearningbook.org/)).  

$$tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$  

### ReLU
Due to the simplicity and low computation cost, Rectified Linear Unit (ReLU) is the most commonly used activation function in deep neural networks (deep in contrast to shallow networks with one hidden layer). Another big advantage of ReLU is that unlike sigmoid and tanh it is less susceptible to the [vanishing gradients problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) which limits the training of deep networks. ReLU was motivated from a biological analogy with neurons where neurons would only be activated if they pass a certain threshold.   

This function maps negative numbers to zero and applies the Identity function to the positive numbers. This function is not differentiable at 0, but this is not a real problem due to the very low probability of getting an output value of exactly zero.  

$$ReLU(z) = max(0, z)$$  

### Leaky ReLU
As mentioned above ReLU maps all negative values to zero. To overcome this, Leaky ReLU shrinks the negative value by a very small multiplier like 0.01. Leaky ReLU is usually introduced at later stages of model tuning as the size of the multiplier could be a hyperparameter which could be optimized to improve model performance.  

$$LReLU(z) = max(0.01z, z)$$  

<br>

## Choosing Activation Functions for Hidden Layers

The first rule of thumb for this selection is that the same activation function is usually used for all hidden layers. This is common in practice to limit the number of hyperparameters that need to be optimized (e.g. number of layers, number of hidden units, learning rate, number of iterations, regularization parameters like dropout layers, momentum term, batch size, etc.).  

In modern neural networks, it is recommended to use ReLU as a default activation function for hidden layers ([Deep Learning, 2016](https://www.deeplearningbook.org/)) especially for Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNN). Sigmoid and tanh were very popular till 2010 but are rarely used today with the advent of faster functions like ReLU (watch [this](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1&ab_channel=3Blue1Brown) for intuition on computation cost).  

Despite being common for majority of models, Sigmoid and Tanh are still used for Recurrent Neural Networks (RNN).  
<br>
### Appropriate Initialization
Closely tied to the selection of activation functions for hidden layers is the weight initialization used for each layer. The initialization step is very critical to the model’s ultimate performance. For our two most popular activation functions, the most common are Xavier Initialization for Tanh and He initialization for ReLU function. For more details and an interactive illustration of the impact of initialization on the learning please refer to [DeepLearning.Ai notes](https://www.deeplearning.ai/ai-notes/initialization/) on this topic.
To illustrate this, consider the three-layer neural network below. You can try initializing this network with different methods and observe the impact on the learning.
<br><br>
## Activation Functions for The Output Layer
Since the output layer directly generates a prediction, the type of activation function is highly dependent on our prediction type. The three main activation functions are Linear, Sigmoid (Logistic) and Softmax. 
### Linear function
This function is also referred to as the Identity function and simply returns the input value without changing the weighted sum of the input. Because of this behavior, the activation function is perfect for regression problems where the prediction could take any real value.  

$$ g(z) = z $$  

### Softmax
Softmax is a generalization of logistic regression for multi-class classification. It normalizes the output values so that all sum up to 1. This is analogous to normalizing probabilities in a distribution and each output value could be interpreted as the probability for that specific class. This function is very similar to the argmax function in Python with a difference that argmax computes a hardmax function by selecting only one class (setting one value to 1 while the rest to zeros) from the list of many classes. For this reason, this function is perfect for multi-class predictions where the classes are mutually exclusive.

$$g(z) = \frac{e^z_i}{\Sigma_{j=1}^n e^z_j}$$  

<br>  

## Choosing Activation Function for the Output Layer
The decision tree below maps the common choices of activation function for the output layer to prediction types. Since the prediction type is also closely tied to the loss (cost) function, I am also including the corresponding loss function used in the popular TensorFlow package used for deep learning.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/output_layer.PNG?raw=True"></center>


> Non-exclusive classes: Refers to cases where the prediction could belong to more than one class. An example wouldbe when we need to pick the classes where output > 0.5 .   

> Mutually exclusive classes: Refers to cases where the model will only pick one class as the final prediction. An example would be we need to pick the one class with highest probability.  

<br>



## Wrapping up: TensorFlow Code Snippet
The below code summarizes the general layout of designing a neural network using TensorFlow and the arguments where the above activation and loss functions could be applied to. The below codes assumes that the input data is already split and scaled.  

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Mutually exclusive multi-class classification
model = Sequential()
model.add(Dense(5, activation='relu'))  # layer 1 with 5 nodes
model.add(Dense(5, activation='relu'))  # layer 2 with 5 nodes
model.add(Dense(1))  # output layer

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x=X_train, y=Y_train, epochs=1000)
```

<br>
<br>
_References:_  

[[1](https://www.cs.ryerson.ca/~aharley/neural-networks/)] Williams, David P. "Demystifying deep convolutional neural networks for sonar image classification." (2019).  
[[2](https://www.deeplearningbook.org/)] Deep Learning (Ian J. Goodfellow, Yoshua Bengio and Aaron Courville), MIT Press, 2016.  
[[3](https://www.tensorflow.org/api_docs/python/tf)] Tensorflow documentation  
[[4](https://www.deeplearning.ai/)] DeepLearning.Ai
<br>