---
layout: post
title: Neural Style Transfer
author: Artan Zandian
date: Feb 10, 2022
excerpt: "Most of the supervised deep learning algorithms optimize a cost function to get a set of parameter values (weights). With Neural Style Transfer, we optimize a cost function to get pixel values of a generated image. This algorithm is, therefore, considered an unsupervised deep learning due to absence of a labeled target."
---

In this blog I will be explaining the concept of Neural Style Transfer (NST), along with the calculations and high level code structure. For Tensorflow coding implementation details please refer to the project [repository](https://github.com/artanzand/neural_style_transfer).
<br>

# Motivation

My intention for doing this project was two-fold. First, as a deep learning enthusiast I wanted to explore if I can implement a research paper ([Gatys et al. (2015)](<https://arxiv.org/abs/1508.06576>)) in code to run on a GPU-accelerated machine (an NVIDIA product), and second, I wanted to have fun implementing a project. I have always been mesmerized by applications that would take an image and create a stylized version of it. With over 15 years of experience with graphics applications, the one thing that I knew for sure was that this is different from applying multiple filters on an image as these applications also changed the patterns of the input image. I knew that Neural Networks were somehow involved, but with my limited knowledge of deep learning I couldn't figure out how it is possible to create an output when no prediction is involved. It turned out that this can be done through unsupervised deep learning rather than usual supervised learning that neural networks are known for.

An inspiring piece of work for my project was this [video](https://www.youtube.com/watch?v=4b9PYIxmcNc&t=800s&ab_channel=CGGeek) from CG Geek who created a realistic 3D rendering of Bob Ross's famous Summer painting. Rather than being realistic the overall goal for me was to create a model that would stylize video frames of a real scenery into a painting. This would allow us to walk through a painting of choice and peek around! The first step towards this goal was clearly transferring an image into a stylized painting.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/summer.jpg?raw=True" width=400></center>
<caption><center>[5] Bob Ross - Summer painting</center></caption>
<br>

# Framework

Most of the supervised deep learning algorithms optimize a cost function to get a set of parameters (weights). With Neural Style Transfer, we optimize a cost function to get pixel values of a generated image. This algorithm is, therefore, considered an unsupervised deep learning due to the absence of a labeled target. It merges a content image (referred to as C in the code) and a style image (S in the code), to create a generated image (G in the code). The generated image G combines the content of the image C with the style of image S.
<br>

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/VGG19.png?raw=True"></center>
<caption><center>[6] VGG19 - Supervised Neural Network</center></caption>  
<br>

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/NST_diagram.JPG?raw=True"></center>
<caption><center>[7] Neural Style Transfer - Unsupervised Learning</center></caption>  
<br>

How should we read the above diagram? We are building a model in which the optimization algorithm updates the pixel values rather than the neural network's parameters. The general idea is to use the activation value of responses from different hidden layers of a convolutional network to build the stylized image. Activation values of layers capture from low level details (edges, strokes, points, corners) to high level details (patterns, objects) when going from shallow to deeper layers. This is then used to perturb the content image, which gives the final stylized image. Due to freezing of network weights, this is considered a transfer learning too.

This is how the model works. We remove the output layer in the traditional supervised neural network and choose some layers' activations to represent the content of an image (multiple outputs). We then set both content and style images as the input to the pretrained VGG network and run forward propagation. We set hidden layer activations for both (a_C and a_G) as the base values to be used for the calculation of the cost for the generated image. The generated image is the input (and the output) of the network, as in each iteration which starts from random noise, we will calculate the activation values for the generated image, find the total cost, and update the input image based on the gradients calculated in respect to pixel values. This is very exciting! Deep learning has many different types of models, and this is only one of them!  
<br>

# Building Blocks

In order to construct the final model we need some helper functions which will help with the calculation of style and content cost. In this post I will be removing the function instructions for space efficiency, but the full code is available in this [repository](https://github.com/artanzand/neural_style_transfer). We will be building the NST algorithm in three steps:

1. The content cost function
2. The style cost function
3. Total cost function  
<br>

## Content Cost

What we are targeting when performing NST is for the content in the generated image G to match the content of image C. For this we need to calculate the content cost function as per the original paper. The content cost takes a hidden layer activations (a_C and a_G) of certain layers within neural network, and measures how different are. In my experimentation with the final model, I was getting the most visually pleasing results when choosing a layer in the middle of the network. This ensures that the network captures both higher-level features and details.

$$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2 $$

where:

- n_H, n_W and n_C are the dimensions of chosen hidden layers.
- a(C) and a(G) are chosen hidden layers' activations.  
<br>

Tensorflow implementation of the cost function looks like the below.  

```py
def compute_content_cost(content_output, generated_output):
    """ """
    # Exclude the last layer output
    a_C, a_G = content_output[-1], generated_output[-1]

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, shape=(1, -1, n_C))
    a_G_unrolled = tf.reshape(a_G, shape=(1, -1, n_C))

    # Compute the cost with tensorflow
    J_content = (1 / (4 * n_C * n_H * n_W)) * tf.reduce_sum(
        tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))
    )

    return J_content
```

<br>

## Style Cost

The most critical ingredient of the style cost is a matrix called a Gram matrix which is simply the dot product of two matrices. This function measures the correlation and prevelance of patterns between activations by measuring how similar the activations of one channel (filter) are to the activations of another channel.  

$$\mathbf{G}_{gram} = \mathbf{A} \mathbf{A}^T$$  

To minimize the distance between the Gram matrix of the Generated image (G) and the Style image (S) we will then need to calculate this difference for each of our activation layers. The style cost for a given layer [l] is defined as below.  

$$J_{style}^{[l]}(S,G) = \frac{1}{(2 \times n_C \times n_H \times n_W)^2} \sum_{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2 $$  

where:

- n_H, n_W and n_C are the dimensions of chosen hidden layers.
- superscripts S and G refer to the Style and Generated images.  
<br>

Tensorflow implementation of one layer style cost is as below. Note that the constant value in front of the summations are not necessary and could dropped since the total style cost already has parameters alpha and beta that can be adjusted to simulated the same outcome. I am following original authors formula in the code below.

```python
def compute_layer_style_cost(a_S, a_G):
    """ """
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images from (1, n_H, n_W, n_C) to have them of shape (n_C, n_H * n_W)
    a_S = tf.reshape(tf.transpose(a_S, perm=[3, 0, 1, 2]), shape=(n_C, -1))
    a_G = tf.reshape(tf.transpose(a_G, perm=[3, 0, 1, 2]), shape=(n_C, -1))

    # Compute gram_matrices for both images S and G
    GS = tf.matmul(a_S, tf.transpose(a_S))
    GG = tf.matmul(a_G, tf.transpose(a_G))

    # Compute the loss
    J_style_layer = (1 / (2 * n_C * n_H * n_W) ** 2) * tf.reduce_sum(
        tf.square(tf.subtract(GS, GG))
    )

    return J_style_layer
```

<br>

## Total Cost

### Style Weights

We will get a better output image if we combine the style cost of multiple layers. To do so we will need to first identify which layers we are interested in, and then assign weights to them. To pick the appropriate layers we will first need to look at the VGG19 architecture by calling `model.layers`.  

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/VGG19_layers.JPG?raw=True"></center>
<br>

block5_conv4 in the above architecture will represent our generated image. Through my experimentations, I realized that layers in the middle are the best to be used for NST. The logic behind this is that these layers capture both the high- and low-level features which are instrumental in having a proper stylized image. Since working with layer weights are not user-friendly, I have written a function that will select the 5 layers of choice with proper weights which I found giving the best results. Here is a snippet of the code with selected layers.

```python
def get_style_layers(similarity):
    """ """
    if similarity == "style":
        style_layers = [
            ("block1_conv1", 0.4),
            ("block2_conv1", 0.3),
            ("block3_conv1", 0.2),
            ("block4_conv1", 0.08),
            ("block5_conv1", 0.02),
        ]
    ...
    return style_layers
```  

"similarity" is an optional argument in the `main()` function (default to "balanced") and accepts three options: "content", "balanced" and "style". As the names suggest, this value defines whether the generated image should be similar to either of input images or a balanced between the two.

Here is the general concept for choosing the weights. If it is desired to have a generated image that softly follows the style image, we should choose larger weights for deeper layers and smaller weights for the shallow layers. The reverse holds if we want the output image to strongly follow the style image (example above). The general intuition is that deeper layers capture higher-level concepts (i.e. the overall shape), and the features in the deeper layers are less localized in the image relative to each other.

Combining the style costs for different layers are done with the following formula where the values for lambda for each layer [l] are defined in `style_layers`.

$$J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)$$

```python
def compute_style_cost(style_image_output, generated_image_output, style_layers):
    """ """
    # Initialize the cost
    J_style = 0

    # Exclude last element of the array containing the content layer image
    a_S = style_image_output[:-1]      # a_S is the hidden layer activations
    a_G = generated_image_output[:-1]  # a_G is the hidden layer activations

    for i, weight in zip(range(len(a_S)), style_layers):
        # Compute style_cost for the current layer and add to the overall cost
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style
```

<br>

### Cost Function

The final step is to put both style and content cost together in a linear function to allow for simultaneous minimization of both. In the below function alpha and beta are hyperparameters that control the relative weighting between content and style. I am setting this to a 3 to 1 ratio in my cost function since our final goal is to create an image similar to the style image.

$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$

```python
def total_cost(J_content, J_style, alpha=10, beta=30):  
    """ """
    J = alpha * J_content + beta * J_style
    return J
```

<br>

# Putting All Together

## Transfer Learning

The idea of using a neural network trained on a different task and applying it to a new task is called transfer learning. NST uses a previously trained convolutional network and builds on top of that. We will use a 19-layer version of the VGG network from the original NST paper published by Visual Geometry Group (VGG) in 2014. This model has already been trained on the ImageNet dataset and has learned to recognize a variety of detailed features at the first hidden layers and high level features at the last layers.  
<br>

## Trainer Function

To allow for calculation of the gradients, which are in respect to the generated pixel values in case of this project, and to make an optimization step, I am using Tensorflow's `GradietTape()` method which will do the automatic differentiation for us. A minor drift from the L-BFGS optimizer which was used by original authors of the paper is the use of Adam optimizer due to Tensorflow not supporting the earlier.

```python
def trainer(generated_image, vgg_model_outputs, style_layers, optimizer, a_C, a_S):
    """ """
    with tf.GradientTape() as tape:
        # a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        # Compute content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute style cost
        J_style = compute_style_cost(a_S, a_G, style_layers)
        # Compute total cost
        J = total_cost(J_content, J_style, alpha=10, beta=30)

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(
        tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    )
```

<br>

## Optimization of the Generated Image

The main function will be in charge of doing some housekeeping items like checking for GPU, loading and preprocessing input images, loading the pretrained VGG19 model, computing the total cost and iterating the train step in order to train the model. We are locking the model weights because we are performing a transfer learning. The key step in the below function is using `tf.Variable()` to define the generated image as the value that we are trying to optimize.

```python
def main(content, style, save, similarity="balanced", epochs=500):
    """ """
    # Limit the image size to increase performance
    image_size = 400

    # Capture content image size to reshape at end
    content_image = Image.open(content)
    content_width, content_height = content_image.size

    # Load pretrained VGG19 model
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        weights="imagenet",
    )
    # Lock in the model weights
    vgg.trainable = False

    # Load Content and Style images
    content_image = preprocess_image(content, image_size)
    style_image = preprocess_image(style, image_size)

    # Randomly initialize Generated image
    # Define the generated image as as tensorflow variable to optimize
    generated_image = tf.Variable(
        tf.image.convert_image_dtype(content_image, tf.float32)
    )
    # Add random noise to initial generated image
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(
        generated_image, clip_value_min=0.0, clip_value_max=1.0
    )

    # Define output layers
    style_layers = get_style_layers(similarity=similarity)
    content_layer = [("block5_conv4", 1)]  # The last layer of VGG19

    vgg_model_outputs = get_layer_outputs(vgg, style_layers + content_layer)

    # Content encoder
    # Define activation encoding for the content image (a_C)
    # Assign content image as the input of VGG19
    preprocessed_content = tf.Variable(
        tf.image.convert_image_dtype(content_image, tf.float32)
    )
    a_C = vgg_model_outputs(preprocessed_content)

    # Style encoder
    # Define activation encoding for the style image (a_S)
    # Assign style image as the input of VGG19
    preprocessed_style = tf.Variable(
        tf.image.convert_image_dtype(style_image, tf.float32)
    )
    a_S = vgg_model_outputs(preprocessed_style)

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Need to redefine the clipped image as a tf.variable to be optimized
    generated_image = tf.Variable(generated_image)

    # Check if GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Train the model
    epochs = int(epochs)
    for i in range(epochs):
        trainer(
            generated_image, vgg_model_outputs, style_layers, optimizer, a_C, a_S
        )
        if i % 500 == 0:
            print(f"Epoch {i} >>>")

    # Resize to original size and save
    image = tensor_to_image(generated_image)
    image = image.resize((content_width, content_height))
    image.save(save + ".jpg")
    print("Image saved.")
```

<br>

# Final Thoughts

To answer my earlier question in the Motivation section, here is the final result when I stylize my Moraine Lake photo with Bob Ross's Summer painting.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/moraine_style.JPG?raw=True"></center>
<br>

The biggest limitation of this technique is being computationally expensive. My implementation of Neural Style Transfer needs to run for over 10,000 epochs to generate a reasonable stylized image which would lead to longer run times. The output below, for example, took over 10 mins with 10,000 epochs on my [Nvidia Jetson Nano](https://artanzand.github.io//Setup-Jetson-Nano/). This would be very problematic for my ultimate goal of creating a motion picture of Bob Ross painting! The two solutions that come into mind are to, first, reduce the frames per second for the video or alternatively create a GIF output instead of a video output, and second, which would require much more research, is whether I would be able to only calculate the extra pixels that appear on a new frame (compared to the previous one) and concatenate them with the already stylized pixels from the previous frame.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/moraine_GIF.gif?raw=True"></center>
<br>

## References

[1] Gatys, Leon A., Ecker, Alexander S. and Bethge, Matthias. "A Neural Algorithm of Artistic Style." CoRR abs/1508.06576 (2015): [link to paper](https://arxiv.org/abs/1508.06576)  

[2] Athalye A., athalye2015neuralstyle, "Neural Style" (2015): [Repository](https://github.com/anishathalye/neural-style)

[3] [DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes  

[4] Log0 [post](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)  

[5] Bob Ross - Summer painting: [Image source](https://blog.twoinchbrush.com/article/paint-better-mountains-upgrade-your-titanium-white/)  

[6] VGG19 - Supervised Neural Network: [Image source](https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0)  

[7] Neural Style Transfer - Unsupervised Learning: [Image source](https://arxiv.org/abs/1508.06576)

<br>
