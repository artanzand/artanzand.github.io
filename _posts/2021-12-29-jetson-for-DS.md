---
layout: post
title: Jetson Nano for Education
author: Artan Zandian
date: Dec 28, 2021
excerpt: "I have been using Nvidia Jetson Nano for over 3 weeks now and strongly believe that this well-priced developer kit should be included in the requirements for Data Science programs. Let's take a look to see why."
---

It is official, I have new hobbby! I have been using Nvidia Jetson Nano developer kit (Jetson Nano) for over 3 weeks now and strongly believe that this well-priced developer kit should be included in the requirements for Data Science programs. This is not a paid product testing and I am trying my best to report my experience with the kit from the eyes of someone who has been exposed to many data science workflows and who understands the necessity of having proper computation resources.  

To explore whethere Nvidia Jetson Nano is the right product for your educational needs, I am breaking down the users into three groups and will be going over what I found beneficial for each.  


## Interested in Data Analysis?

As a data analyst one of the biggest advantages for you would get is having a separate machine for under USD$60 which runs on Linux. As the majority of large data lives on cloud with servers which run on Linux, this product will be a strong candidate to give you the opportunity to learn and brush up on your Linux skills. This is good exercise for someone who is new to cloud computing, doesn't know how remote machines work, and wants to figure out everything before subscribing for popular cloud services.  

As data analysts you are expected to set up data science tools and software on multiple machines and it is instrumental to learn how to do it on Linux. What intrigued me the most was, however, setting up a [remote connection](https://artanzand.github.io//Setup-Jetson-Nano/),  configuring my local modem to work as a router, and automating the [set up of images](https://artanzand.github.io//Setup-Jetson-Nano/) on a remote machine. Afterall, not all startup companies or even small companies have dedicated IT staff.  

One of the selling features of Jetson Nano for me was low power consumption as it is runs on 15w (2-3Amp, 5v). To put this in perspective, a modern 14-15 inch laptop typically uses about [60 watts](https://energyusecalculator.com/electricity_laptop.htm) of electricity, the equivalent of 0.05 kWh.


## Interested in Machine Learning?
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/idea_code_exp.PNG?raw=True"></center>
<br>
If you have experienced machine learning especially the development of models you know that this is a very iterative process between idea, code and experiment where you need to not only go through many models which you think would work, but also doing feature selection, hyper-parameter optimization and more. In my experience prior to working with Jetson nano because of not having a second machine to connect to remotely, my only option was to use cloud services which I turned out to be costly if you doing it for a longer period of study.   


There are many cons of just using one machine for education which I think half of my peers would agree with, but the worst of all is having a locked computer running low on resources and you wanting to open up a simple browser to do further research/study. Using Jetson Nano I was able to get similar run times to my COREi5 notebook while using Cuda to optimize the computation. This brings us to the next advantage which is the whole NVIDIA [Cuda](https://en.wikipedia.org/wiki/CUDA) echosystem with optimized packages like [CuPy](https://cupy.dev/) that allow parallel computing to speed up the already-fast Python packages like NumPy.  

CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on its own GPUs (graphics processing units). NVIDIA’s CUDA Python provides a driver and runtime API for existing toolkits and libraries to simplify GPU-based accelerated processing. As the most popular programming languages for data analytics, machine learning and deep learning applications Python is an interpreted language, but is considered too slow for high-performance computing.

[Numba](https://github.com/numba/numba) is an open-source JIT compiler from Anaconda that offers range of options for parallelising Python and NumPy code for CPUs and GPUs, often with a minimum of new syntax and jargon. CUDA Python and Numba offer the best of both worlds which are the rapid iterative development with Python combined with the speed of a compiled language (like C) which can use both CPUs and NVIDIA GPUs. Jetson nano would be your most affordable introduction to Cuda and Numba.


## Interested in Deep Learning?

This product is a no-brainer for you. Cuda not only equips you with the platform necessary to do parallel GPU computing with NumPy, but is also optimized for the two most popular deeplearning packages of Tensorflow and PyCharm. However, what I found most fascinating about Jetson nano is the community behind it. Not only you get to go through their free [courses and certification](https://developer.nvidia.com/embedded/learn/jetson-ai-certification-programs#submit_project), but you also have access to the [Hello AI World](https://github.com/dusty-nv/jetson-inference#video-walkthroughs) with a whole bunch of projects and tutorials done by learner across the globe. The tutorials and examples are very hands on. You get to collect your own data or do transfer learning on the trained data that you load, and do object detection on your live video input.

Not happy with the 128-core GPU that you get with Jetson Nano developer kit? No problem! The scalability of this family of products allow you to move your data, operating system, and all downloaded image to [another Jetson kit](https://www.nvidia.com/en-us/autonomous-machines/jetson-store/). All you need to do is to remove your microSD card and insert it in the new kit. This is the power of Linux at your fingertips! 
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/microSD.PNG?raw=True"></center>



## Downfalls
- why not do cloud computing
- probably not (limited to the size of microSD card)
- not fast for day-to-day usage because reading off of microsSD
- Gets hot really quick
- Installing the image and connecting the data cable is not a pleasant experience for Windows users. See [my post](https://artanzand.github.io//Setup-Jetson-Nano/) on the solution here.
