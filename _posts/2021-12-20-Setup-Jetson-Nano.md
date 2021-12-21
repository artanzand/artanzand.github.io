---
layout: post
title: Preferred Setup for NVIDIA Jetson Nano
author: Artan Zandian
date: Dec 20, 2021
---

Recently I had my hands on [Jetson Nano 2GB Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/education-projects/) introduced by Nvidia in Oct 2020 as a successor to the original [Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) with 4GB of RAM. Despite the difference in memory size, number of ports and power supply, the two kits almost offer the same performance using an identical 128-core NVIDIA Maxwell GPU. See below for more detail:



|       |JETSON NANO 2GB |JETSON NANO |
|-------|-----------|------------------ |
|Memory |2 GB |4 GB|
|Camera Connector|1x CSI|2x CSI|
|Power	|USB-C	|Micro-USB or DC barrel connector|
|USB	|1x USB 3.0 Type A, 2x USB 2.0 Type A, 1x Micro-USB 2.0 |	4x USB 3.0 Type A, 1x Micro-USB 2.0|
| Display	|1x HDMI	|1x HDMI, 1x DP
|Wireless Connectivity	|USB wireless adapter included |	M.2 Key E slot (adapter not included)|
|Price	| USD$59 |	USD$99|

<br>  

## Default Setup
```
ssh <username>@192.168.55.1
```

<br> 

## Preferred Setup - Ethernet
```
ifconfig
```


Replace the `<username>` and `<ethernet-ip>` with the Linux username and the actual IP acquired from the command above. After this, you will be prompted to enter you Linux password.
```
ssh <username>@<ethernet-ip>
```
Once remotely connected to Jetson Nano using the secure SSH connection, we will need a command line script to start Jupyter Lab on our remote machine. The next step is a one-time-only command to save the script to a shell file.

```
# create a reusable script
echo "sudo docker run --runtime nvidia -it --rm --network host \
    --memory=500M --memory-swap=4G \
    --volume ~/nvdli-data:/nvdli-nano/data \
    --volume /tmp/argus_socket:/tmp/argus_socket \
    --device /dev/video0 \
    nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1" > docker_dli_run.sh

# make the script executable
chmod +x docker_dli_run.sh
```

Once executed we could run the stript below to fire up Jupyter Lab.
```
# run the script
./docker_dli_run.sh
```