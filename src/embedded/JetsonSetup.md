# NVIDIA Jetson Nano Installation Guide

This document will walk you through the process of setting up your Jetson Nano with
all of the tools needed to run `Listen.py`. The 4GB model of the Jetson Nano was used
in the development of this module, so that particular SKU is recommended.

## Details
First, you'll need a Jetson Nano, a power adapter for it, some way to connect it to your network,
and a MicroSD card to serve as its hard disk. A 128 GB card was used in the development
of this module, but a smaller card should work fine depending on intended use of the space.

1. Flash the OS image onto the SD card (instructions found at https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)
2. Connect your Jetson to a mouse, keyboard, and monitor (and possibly your network), and go through the setup wizard.
3. Use git to clone the ElephantCallAI repository.
4. Install a browser on the Jetson (e.g., Chromium) for use in later steps.

The following commands can be executed individually or chained together in a simple bash script. 
Some take a significant amount of time (~45 minutes).

5. `$ sudo apt-get update`

6. `$ sudo apt-get install python3-pip`

7. `$ sudo apt-get install libffi-dev; sudo apt-get install libportaudio2; pip3 install sounddevice`

8. `$ pip3 install cython; pip3 install numpy==1.16.4` (this will take a while)

9. `$ sudo apt-get install libjpeg8-dev; pip3 install pillow; pip3 install matplotlib` (this will also take a while)

10. Now, follow the instructions on NVIDIA's Jetson forum for PyTorch installation (https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048). 
Install torch 1.8 and the corresponding version of torchvision per the instructions on the site (this may take half an hour or more)

Execute the following commands:

11. `$ pip3 install sklearn` (can take upwards of half an hour)

12. `$ pip3 install tensorboardX`

At this point, your Jetson should be completely equipped to run `Listen.py` (outside of plugging in the USB microphone).
If you are setting up multiple Jetson units for this, consider duplicating the data on the SD card of the Jetson
you used for this process and writing it to an SD card for each Jetson you are setting up. This may save a lot of time.

### Mic Test

If you want to check whether your USB mic is working, use the following command to create a
20-second long WAV file (with the same parameters used by `Listen.py`) recording it on your Jetson:

`$ arecord -D pulse -r 8000 -f S16_LE -d 20 -t wav test.wav`

Copy `test.wav`, the newly-created file, over to a machine with a speaker and play it. If you hear your voice 
(or whatever else you may have recorded), it's working.