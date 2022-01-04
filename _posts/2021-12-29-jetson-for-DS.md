---
layout: post
title: Jetson Nano for Education
author: Artan Zandian
date: Dec 28, 2021
excerpt: "I have been using Nvidia Jetson Nano for over 3 weeks now and strongly believe that this well-priced developer kit should be included in the requirements for Data Science programs. Let's take a look..."
---

It is official. I have new hobby! I have been using Nvidia Jetson Nano developer kit (Jetson Nano) for over 3 weeks now and strongly believe that this well-priced developer kit should be included in the requirements list for Data Science programs. This is not a paid product testing, and my intention is to report my experience with the kit from the eyes of a learner who has been exposed to data science and deep learning workflows for a while now and who understands the necessity of having proper computation resources available.  

To explore whether Nvidia Jetson Nano is the right product for your educational needs, I am breaking down the usage type into three groups and will be going over what I found beneficial for each. I will sum up my review with the shortfalls I found when setting up and working with Jetson Nano.  
<br><br>

## Data Analysis

As a data analyst one of the biggest advantages you would get with Jetson Nano is having a second machine for under USD$60 which would run on Linux. As most big data lives on cloud with servers running on Linux, this product will be a strong candidate to give you the opportunity to learn and brush up on your Linux skills. This is good exercise for someone who is new to cloud computing, doesn't know how remote machines work, and wants to figure out Linux functionalities and explore software and package dependencies before subscribing for one of the popular cloud services.  

As data analysts you are expected to set up data science tools and software on multiple machines and it is instrumental to learn how to do it on Linux. What intrigued me the most was, however, setting up a [remote connection](https://artanzand.github.io//Setup-Jetson-Nano/),  configuring my local modem to work as a router, and automating the [set up of images](https://artanzand.github.io//Setup-Jetson-Nano/) on a remote machine. Afterall, not all startup or small- to mid-size companies can afford dedicated IT staff and you would be their next best option.  

One of the selling features of Jetson Nano for me was low power consumption as it is runs on 15w (3Amp, 5v). To put this in perspective, a modern 14-15 inch laptop typically uses about [60 watts](https://energyusecalculator.com/electricity_laptop.htm) of electricity, the equivalent of 0.06 kWh.
<br><br>

## Machine Learning
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/idea_code_exp.PNG?raw=True"></center>
<br>
If you have experienced machine learning, you will agree with me that the whole process is a very iterative process between idea, code and experiment where you need to not only explore performance of numerous models which you think would work, but also do feature selection, hyper-parameter optimization and more. In my experience prior to working with Jetson nano because of not having a second machine to connect to remotely, my only option was to use cloud services which turned out to be costly when I decided to do it for extended period of study/research.   

There are many cons to just using one machine for education which I think half of my peers at [UBC Master of Data Science program](https://masterdatascience.ubc.ca/) would agree with, but the worst of all is having a slow computer running low on resources and you wanting to open up a simple browser to do further research/study. The downstream effect of the frustration with a slow computer is that the learner does not get to redo the idea-code-experiment loop, and therefore, does not get to try out more resource-intensive models like ensemble models or perform hyperparameter tuning on various models to understand the bias-variance tradeoff.  

Using Jetson Nano I was able to get similar run times to my COREi5 notebook (with 8GB of RAM) while using CUDA to optimize the computation. This brings us to the next advantage of a product like this which is the whole Nvidia [CUDA](https://en.wikipedia.org/wiki/CUDA) ecosystem with optimized packages like [CuPy](https://cupy.dev/) that allow parallel computing to speed up the already-fast Python packages like NumPy.  

CUDA is a parallel computing platform and programming model developed by Nvidia for general computing on its own GPUs (graphics processing units). Nvidia’s CUDA Python provides a driver and runtime API for existing toolkits and libraries to simplify GPU-based accelerated processing. As the most popular programming languages for data analytics, machine learning and deep learning applications Python is an interpreted language but is considered too slow for high-performance computing.

[Numba](https://github.com/numba/numba) is an open-source JIT compiler from Anaconda that offers range of options for parallelizing Python and NumPy code for CPUs and GPUs, often with a minimum of new syntax and jargon. CUDA Python and Numba offer the best of both worlds which are the rapid iterative development with Python combined with the speed of a compiled language (like C) which can use both CPUs and Nvidia GPUs. Jetson nano would be your most affordable introduction to CUDA and Numba.
<br><br>

## Deep Learning

I wish I knew about this a year ago when I first started learning and implementing neural networks on my average notebook: CUDA not only equips you with the platform necessary to do parallel GPU computing with NumPy, but is also optimized for the two most popular deep learning packages of Tensorflow and PyTorch. However, what I found most fascinating about Jetson nano is the whole community of learners and instructors behind it. Not only you get to go through their free [courses and certification](https://developer.nvidia.com/embedded/learn/jetson-ai-certification-programs#submit_project), but you also have access to [Hello AI World](https://github.com/dusty-nv/jetson-inference#video-walkthroughs) with a whole bunch of [projects and tutorials](https://developer.nvidia.com/embedded/community/jetson-projects) done by learners across the globe to get inspired from. The tutorials and examples are very hands on. For example, in one of the tutorials you get to use [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) along with your newly collected data from your camera to train and implement your computer vision models and do object detection in real time using your live camera input.

Not happy with the 128-core GPU that you get with Jetson Nano developer kit? Not a problem! The scalability of this family of products allows you to move your data, operating system, and all downloaded data to [another Jetson kit](https://www.nvidia.com/en-us/autonomous-machines/jetson-store/). All you need to do is to remove your microSD card and insert it in the new kit. This is the power of Linux at your fingertips! 
<center><img src = "https://github.com/artanzand/artanzand.github.io/blob/master/_posts/img/microSD.PNG?raw=True"></center>

<br><br>

## Downfalls
With every improvement comes some downfalls or what I would like to call "lessons learned for future improvement". Jetson Nano is no exception. Unless you go with a pricier kit like NVIDIA [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/jetson-store/) Developer kit, you can't expect your Jetson Nano to do magic with high resolution image input for a longer period of time or do heavy computer gaming. Although I have tried heavy graphical simulation on the kit and it works fine for a short time, the kit gets really hot really quick if you push for extended performance! This is also the case for times when you are training a neural network. A cooling fan is not provided, but screw holes have been envisioned for those who need to add a mini fan.  

Although I mentioned earlier that the using microSD helps with the scalability of the product, it causes some issues down the line the most important of which is a slower reading and writing from memory when compared to a real SSD or hard drive. 

Jetson Nano is not meant to replace your laptop. Therefore, it does not give you a good performance when used for day-to-day tasks like streaming a video or running your favorite editing software that is not meant to run on GPU. Afterall, you are using a Linux operating system on a 2GB RAM and this limits you in terms of both CPU performance and software availability.

In terms of hardware, Jetson nano comes with no audio jack, and you will need to purchase a USB adapter if you want to check out your voice data. The most irritating aspect of configuring and using the product, however, was the power supply. Although the manual recommends a 2-3Amp 5v input, you will get an overcurrent (I know!) warning on your Linux if you provide it anything less than 3Amps. This is mainly due to the length of the cable which causes the current to drop below 2Amp. A 3Amp 5V power input combination is not common and you are restricted to their recommended [Raspberry Pi Adapter](https://www.amazon.ca/Smraza-Compatible-Raspberry-Charger-Adapter/dp/B07VGGHR6N/ref=sr_1_5?crid=1DSY0A81V4CJV&keywords=raspberry+pi+charger&qid=1641270371&s=electronics&sprefix=raspberry+pi+charger%2Celectronics%2C122&sr=1-5) or a very short power cable. I ended up using a 10cm cable to minimize the current loss.  

Last but not the least, as a Windows user I found installing the image and setting up the headless connection using a data cable very challenging. This is a well-known issue related to the default security setup in Windows firewall where it needs to be manually turned off to enable the transfer of data with an external machine. I ended up experimenting and figuring out an alternative setup via ethernet. See my post on the solution [here](https://artanzand.github.io//Setup-Jetson-Nano/).
