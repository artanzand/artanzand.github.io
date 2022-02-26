---
layout: post
title: Semantic Segmentation with U-Net
author: Artan Zandian
date: Feb 24, 2022
excerpt: "In a second phase of my Neural Style Transfer model, I have applied Semantic Segmentation to exclude person figures from the style transfer. In this post, I will be going over the U-Net architecture and the project pipeline used to achieve this."
---

In a second phase of my Neural Style Transfer (NST) model (see [NST project](https://artanzand.github.io//neural-style-transfer/) and [repo](https://github.com/artanzand/neural_style_transfer)), I have applied Semantic Segmentation to exclude person figures from the style transfer (see [Segmentation repo](https://github.com/artanzand/image_segmentation_NST) for reproducible code). In this post, I will be going over the U-Net architecture and the project pipeline used to achieve this. This project is inspired by a stylized image of a child riding a bicycle which I came across with in a research paper ([Kasten et al. (2021)](https://layered-neural-atlases.github.io/)) by Adobe Research team.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/Adobe_stylized.jpg?raw=True" width=600></center>
<caption><center>[1] Cropped out figure from stylized background (Kasten et al. (2021))</center></caption>  
<br>

# Motivation

So far, we have been able to created a stylized version of an image using Neural style Transfer (see this [post](https://artanzand.github.io//neural-style-transfer/) for walkthrough of that project). This works perfect if the intention is to stylize the whole image and where the partial details of the original image do not matter. A simple use case of this is for stylizing a scenary image. However, as shown in the example image below when human figures appear in a "content" image, the results of the generated stylized image are not satisfactory. To overcome this issue, our neural network pipeline needs to somehow learn where the people are in an image and cut them out.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/nst_problem.JPG?raw=True"></center>
<caption><center>Failure in application of NST on images containing figures</center></caption>
<br>

# General Guideline

Our main objective for this project is to find masks which we would help us isolate the figure image from its background and vice versa. Masks are defined as tensors of 0's and 1's in this context, and when multiplied by by an image tensor produce our desired cutouts. For our purpose we will need two masks were one is the complement of the other. For example, if we have the figure mask as tensor Y, the background mask can be calculated by 1 - Y.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/masks.JPG?raw=True" width=400></center>
<caption><center>Figure and Background masks</center></caption>
<br>

The below diagrams shows the overall pipeline. We have two neural networks. Neural Style Transfer is in charge of creating stylized images which has already been modeled in my [previous project](https://artanzand.github.io//neural-style-transfer/), and Semantic Segmentation network, which we will walk through in this post, will be generating tensors of zeros and ones. The outputs of Semantic segmentation will create the stylized bachground and the figure cutout when multiplied by the stylized and content image respectively. The final output is a summation of the two isolated pieces.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/NST_segmentation_framework.JPG?raw=True"></center>
<caption><center>Project pipeline</center></caption>
<br>

# Semantic Image Segmentation

We will be building a special type of Convolutional Neural Networks (CNN) designed for quick and precise image segmentation. The predicted result of this network will be in shape of a label for every single pixel in an image. This is referred to as pixelwise prediction in the literature. This method of image classification is called semantic image classification and plays a critical role in self-driving cars which demand a perfect understanding of the surounding environment so that they avoid other cars and people, or change lanes.

A major source of confusion is the difference between object detection and image segmentation. The similarity is that both techniques intend to answer the question: "What objects are present in this image and what are the locations of thos objects?". The main difference is that the objective of object detection techniques like [YOLO](https://arxiv.org/abs/1506.02640) is to label objects by bounding boxes whereas semantic image segmentation aims to predict a pixel precise mask for each object in the image by labeling each pixel in the image with its corresponding class.

For reproducibility purposes and to allow for faster training of my model on the cloud, I have opted for a smaller-size [dataset](https://www.kaggle.com/nikhilroxtomar/person-segmentation) that would only label person figures. The technique can, however, be generalized to image segmentation for multiple classes, and in my post I will be referring to the respective changes in the general code and concept.

<br>

# U-Net Architecture

U-Net, named after its U-shape, was originally created in 2015 for biomedical image segmentation (Ronneberger et al. (2015) [paper](https://arxiv.org/abs/1505.04597)), but soon became very popular for other semantic segmentation tasks. Note that due to computational constraints I have reduced the number of filters and steps used in the original paper.
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/U-net.JPG?raw=True"></center>
<caption><center>U-Net Architecture used for this project</center></caption>
<br>

U-Net is based on the Fully Convolutional Network (FCN) which is comprised of two main sections; an encoder section which downsamples the input image (similar to a typical CNN network), and a decoder section which replaces the dense layer found in the output layer of CNN with transposed convolution layer (see diagram below on how transposed convolution works) that upsample the feature map back to the size of the original input image. A major downfall of an FCN is that the final feature layer of the FCN suffers from information loss due to excessive downsampling which are required for inference. Solving this issue is necessary because the dense layers destroy spatial information which is the most essential part of image segmentation tasks.  

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/transpose_conv.gif?raw=True"></center>
<caption><center>[6] Transpose Convolution</center></caption>  
<br>

U-Net improves on the FCN by introducing skip connections shown by green arrows in the U-Net diagram which are critical for preserving the spatial information. Further to that and instead of one transposed convolution at the end of the network, U-Net uses a matching number of transposed convolutions for upsampling (decoding) to those used for downsampling (encoding). This will map the encoded image back up to the original input image size. The role of skip connections is to retain information that would otherwise become lost during encoding. Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while also reducing computation cost.

To build the U-Net model we need two blocks. Each step in the U-Net diagram is considered a block.

1. Encoder block - A series of convolution layers followed by maxpooling
2. Decoder block - A transposed convolution layer followed by convolution layers  
<br>

## Encoder Block

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/encoder_block.JPG?raw=True" width=400></center>

Each decoder block is comprised of two Convolution layers with ReLU activations. As per the original [paper](https://arxiv.org/abs/1505.04597)'s instructions we will apply Dropout, and MaxPooling to some of the decoder blocks, specifically to the last two blocks of the downsampling, and our function will, therefore, need to allow for that.

The function will return two outputs:

- next_layer: The tensor that will go into the next block.
- skip_connection: The tensor that will go into the matching decoding block.

Here is a reduced version of the encoder block in Tensorflow. I am initializing my kernels with [he_normal](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal). Also note that when `max_pooling` is set to `True`, the next_layer will be the output of the MaxPooling layer, but the skip_connection will be the output of the previously applied layer. Else, both results will be identical. `max_pooling` will be set to false in the fifth encoder block where the resultant tensor is directly fed into the decoder block.

```python
def encoder_block(inputs=None, n_filters=32, dropout=0, max_pooling=True):
    """ """
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)

    # Add dropout if existing
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    # Add MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)

    else:
        next_layer = conv

    skip_connection = conv  # excluding maxpool from skip connection

    return next_layer, skip_connection
```

<br>

## Decoder Block

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/decoder_block.JPG?raw=True" width=150></center>

The decoder block takes in two arguments:  

- expansive_input - The input tensor from the previous layer, and;
- contractive_input - the input tensor from the previous skip layer in the encoder section  

The number of filters will be the same as each corresponding encoder block. We will apply transpose convolution on the expansive_input, or the input tensor from the previous layer. We will then merge the output of the transpose convolution layer with contractive_input using the `tf.concatenate()` function to create the skip connection. Skip connection is the step that allows for the capture of finer information from the earlier layers of the network where shape details were still present. We will apply two convolution layers at the end of this block.

```python
def decoder_block(expansive_input, contractive_input, n_filters=32):
    """ """
    up = Conv2DTranspose(
        filters=n_filters, kernel_size=(3, 3), strides=2, padding="same"
    )(expansive_input)

    # Merge the previous output and the contractive_input
    # The order of concatenation for channels doesn't matter
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge)
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)

    return conv
```

<br>

## Building the Model

This is where the model blocks are put together by chaining encoders, connections, and decoders. We will need to identify the number of classes (`n_classes`) of the final output image. Connecting the inputs and outputs of each layer is not hard if we follow the U-Net diagram, but let's step out for a short minute to clarify how what the number of classes should be and the desired shape of an output tensor.

In semantic segmentation, we need as many masks as we have object classes.Classes in the output tensor are defined by channels. For example, if we have 4 classes, we will have an output with the same height and width as the input image, but with 4 channels with values of each cell in a channel representing the probabilities of that pixel belonging to that channel. The intuition is that when predicting we will get the `argmax()` of the pixels in the last dimension of the tensor (the four channels in our example) to come up with the final prediction for that pixel. For the case of my model because I was only interested in identifying the pixels containing human figures, I selected a dataset that would satisfy my objective. Therefore, I will have only one class (output channel) in my model.  

channel images

Now let's go back to the model. The function is comprised of two sections. The first half is where we use the encoder_block, and chain the first output of each block to the next one. We take a note of the second output of the encoder_block for later in the Decoder section. As per the paper recommendations we will be doubling the number of filters while MaxPooling in each block will divide the height and width by two. On the fifth block, also known as the Bottleneck (for obvious reasones!), we will be turning the MaxPooling off since we will be entering the second section where we need to upsample.  

Decoder blocks in the second section will be expanding the size of the image. This time we will chain the expanded output of the previous block with the second output from the corresponding encoder block. At each step, we will half the number of filter. `conv9` is a convolution layer with he_normal initializer. At this stage, we will have an output with exactly the same shape as the input. Finally, on the output layer `conv10` we will use a convolutional layer with a kernel size of 1 to bring the number of channels to our desired number of classes (1 in case of this project).

```python
def U_Net(input_size=(320, 320, 3), n_filters=32, n_classes=1):
    """ """
    inputs = Input(input_size)

    # Encoder section
    #================
    # Double the number of filters at each new step
    # The first element of encoder_block is input to the next layer
    eblock1 = encoder_block(inputs, n_filters)
    eblock2 = encoder_block(eblock1[0], n_filters * 2)
    eblock3 = encoder_block(eblock2[0], n_filters * 4)
    eblock4 = encoder_block(eblock3[0], n_filters * 8, dropout=0.3)
    eblock5 = encoder_block(eblock4[0], n_filters * 16, dropout=0.3, max_pooling=False)

    # Decoder section
    #================
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output
    # The second element of encoder_block is input to the skip connection
    # Halving the number of filters of the previous block in each section
    dblock6 = decoder_block(
        expansive_input=eblock5[1], contractive_input=eblock4[1], n_filters=n_filters * 8
    )
    dblock7 = decoder_block(
        expansive_input=dblock6, contractive_input=eblock3[1], n_filters=n_filters * 4
    )
    dblock8 = decoder_block(
        expansive_input=dblock7, contractive_input=eblock2[1], n_filters=n_filters * 2
    )
    dblock9 = decoder_block(
        expansive_input=dblock8, contractive_input=eblock1[1], n_filters=n_filters
    )

    conv9 = Conv2D(
        n_filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(dblock9)

    # Add a 1x1 Conv2D (projection) layer with n_classes filters to adjust number of output channels
    conv10 = Conv2D(filters=n_classes, kernel_size=1, padding="same")(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
```

Next, we will instantiate the model and compile it. Adam is the optimizer of choice, but other optimizers work as well. For the loss function we use BinaryCrossentropy (since we have person(1) or not person(0)), and accuracy is our metric. Extra details about loading the data and fitting the model can be found in `train.py` in this [repo](https://github.com/artanzand/image_segmentation_NST/blob/main/src/train.py).

```python
unet = U_Net()
unet.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
```

> Note that when using more than one class, we will be dealing with a very sparse tensor, and therefore, it is recommended to use `SparseCategoricalCrossentropy()` for computational efficiency.  

```python
unet.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

<br>

# Neural Style Transfer + Segmentation

Now that we know how to get the ingredients i.e. the figure mask from Image Segmentation and the stylized image from Neural Style Transfer, it is time to put them together to create our improved generated image. The diagram below is our roadmap.

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/semantic_diagram.JPG?raw=True"></center>
<caption><center>Semantic Image Segmentation pipeline</center></caption>

This is the code I have added to the main function of `stylize.py` in the Neural Style Transfer [project](https://github.com/artanzand/neural_style_transfer/blob/main/stylize.py) to combine the results of the two. From prior to this code, we have computed and already have the variable `generated_image` as our output from NST. This is the same as the "Stylize Image" from the diagram above. Calling the `predict()` function on the content image will give the masked photo which is the multiplication of the content image with the predicted mask (0's and 1's). The reason I defined the output my predict function is more of a logistics problem as it is not possible to resize an integer tensor (the output mask is of shape 320x320).

Instead of the 1 - Y function mentioned earlier to acquire the background mask I am filtering zeros of the masked_photo. The nice thing about this is that I don't need any further resizing to bring my background mask to the same shape as the generated_image (Stylized Image). Now that we have the background mask, all we need to do is mutliplying it by the generate_image. And to wrap it up, we will sum the two masked_images to create the final output.

```python
def main(content, style, save, similarity="balanced", epochs=500):
    """ """
    ...
    # Create crop out of personage in the content image
    # Image segmentation is done using U-Net architecture (size of (320, 320))
    masked_photo = predict(content)
    # resize to current stylized image
    masked_photo = tf.image.resize(
        masked_photo, [image_size, image_size], method="nearest"
    )

    # Create background mask
    background_mask = tf.cast(masked_photo == 0, tf.float32)
    masked_background = (
        generated_image * background_mask
    )  # photo mask is 0's and 1's

    # Combine the two images (the two pieces complement eachother)
    segmented_style = masked_background + masked_photo

    # Resize stylized image to original size and save
    seg_style_image = tensor_to_image(segmented_style)
    seg_style_image = seg_style_image.resize((content_width, content_height))

    seg_style_image.save(save + ".jpg")
```

<br>

# Result and Final Thoughts

<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/evolution.gif?raw=True"></center>
<caption><center>Evolution of NST with Semantic Image Segmentation</center></caption>

This is the animation of what the final product looks like. Although I am getting my desired output, I can see two further improvements for the project.

1. Add more classes: It would be nice if the image segmentation could classify more objects. In the case of the initial inspiration image for example, it would have been nice if the user could select human and bicycle to be filtered out from the neural style transfer. This would be an easy improvement with changes in literally two words in the code (updating n_classes and changing the loss function to `SparseCategoricalCrossentropy()`) as mentioned in the previous section.  
2. Reorder the pipeline: The way I am solving this problem is not the most computationaly efficient way. My current pipeline is that I am creating the stylized background, and then I apply the mask calculated from my segmentation model. It would have been computationally more efficient if I generated the mask first, applied the mask on the original image to freeze the human figure pixels, and then do the style transfer. The computational cost saving would be proporionate to the the count / size of human figures in a certain image.
<br>

## References

[1] Kasten, Yoni, et al. "Layered neural atlases for consistent video editing." ACM Transactions on Graphics (TOG) 40.6 (2021): [link to paper](https://arxiv.org/pdf/2109.11418.pdf)  
[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.: [link to paper](https://arxiv.org/abs/1505.04597)  
[3] [DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes  
[4] Image Segmentation with DeepLabV3Plus: [link to repo](https://github.com/nikhilroxtomar/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow)
[5] Style image [reference](https://androidphototips.com/5-best-painting-apps-that-turn-your-iphone-photos-into-paintings/)  
[6] Transpose Convolution ([image reference](https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1))
