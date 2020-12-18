## Overview
Here we have examples of Google Colaboratory (aka Colab or simply colabs) notebooks trained on various datasets. They are free GPU instances, so great for prototyping and even simple production models.

## The Tutorials
The below tutorials are based on MobileNetv2-SSD, which is a decent-performance, decent-framework object dectector which natively runs on DepthAI.  A bunch of other object detectors could be trained/supported on Colab and run on DepthAI, so if you have a request for a different object detector/network backend, please feel free to make a Github Issue!

### Tiny YOLOv3 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYolov3_Object_Detector_Training_on_Custom_Data.ipynb)

These two tutorials show how to quickly train an object detector based on the YOLOv4-tiny and YOLOv3-tiny network.

[![TinyYoloMaskDemo](https://user-images.githubusercontent.com/5244214/90792755-2a2da000-e30b-11ea-9cf3-b2f6cd4b00c4.gif)](https://www.youtube.com/watch?v=LkJi7Kp7U-o&feature=youtu.be)

### Tiny YOLOv4 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYOLOv4_Object_Detector_Training_on_Custom_Data.ipynb)

Just like the Tiny YOLOv3 tutorial above, you can train your own Tiny YOLOv4 network

### MobileNetSSDv2 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_Object_Detection_With_Custom_Data_Demo_Training.ipynb)
The tutorial notebook 
*Easy_Object_Detection_With_Custom_Data_Demo_Training.ipynb* shows how to quickly train an object detector based on the Mobilenet SSDv2 network. 

Optionally, see our documentation around this module ([here](https://docs.luxonis.com/tutorials/object_det_mnssv2_training/)) for of a guide/walk-through on how to use this notebook.  Also, feel free to jump right into the Notebook, with some experimentation it's relatively straightforward to get a model trained.

After training is complete, it also converts the model to a .blob file that runs on our DepthAI platform and modules. First the model is converted to a format usable by OpenVINO called Intermediate Representation, or IR. The IR model is then compiled to a .blob file using a server we set up for that purpose. (The IR model can also be [converted locally to a blob](https://github.com/luxonis/depthai#conversion-of-existing-trained-models-into-intel-movidius-binary-format).)

And that's it, in less than a couple of hours a fairly advanced proof of concept object detector can run on DepthAI to detect objects of your choice and their associated spatial information (i.e. xyz location). For example this notebook was used to train DepthAI to locate strawberries in 3D space, see below:

[![Real-time 3D Strawberry Detector](https://user-images.githubusercontent.com/5244214/90794705-6104b580-e30d-11ea-8275-468b300b4659.gif)](https://www.youtube.com/watch?v=Okjh2OCP-o8& "Real-Time Spatial AI to Pick Strawberries")

The above example used a DepthAI Modular Cameras Edition ([BW1098FFC](https://shop.luxonis.com/products/depthai-usb3-edition)).

### COVID-19 Mask/No-Mask Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Medical_Mask_Detection_Demo_Training.ipynb)
The *Medical Mask Detection Demo Training.ipynb* training notebook shows another example of a more complex object detector. The training data set consists of people wearing or not wearing masks for viral protection. There are almost 700 pictures with approximately 3600 bounding box annotations. The images are complex: they vary quite a lot in scale and composition. Nonetheless, the object detector does quite a good job with this relatively small dataset for such a task. Again, training takes around 2 hours. Depending on which GPU the Colab lottery assigns to the notebook instance, training 10k steps can take 2.5 hours or 1.5 hours. Either way, a short period for such a good quality proof of concept for such a difficult task. 
We then performed the steps above for converting to blob and then running it on our DepthAI module. 

Below is a quick test of the model produced with this notebook on Luxonis DepthAI Onboard Cameras Edition ([BW1098OBC](https://shop.luxonis.com/products/bw10980bc)):

[![COVID19 Mask Detector](https://user-images.githubusercontent.com/5244214/90733159-74436100-e2cc-11ea-8fb6-d4be937d90e5.gif)](https://www.youtube.com/watch?v=d_oUxDzWHd0 "COVID19 Mask/No-Mask")


### Deeplabv3 Plus Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/DeepLabv3plus%2BMobilenetv2_HighLevelDemo.ipynb)

This tutorial shows you how to train semantic-segmentation networks with Deeplabv3+ with a MobileNetv2 backbone.  Examples of such trained networks are below:

#### Aerial Semantic Segmentation for UAV disaster recovery or automated landing:
[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/102667886-c182f280-4147-11eb-904c-c8c89d9706a7.png)](https://www.youtube.com/watch?v=1EB5bpmUq1Q "Deeplabv3+ Custom Training for DepthAI")

#### Person semantic segmentation
[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/102668154-76b5aa80-4148-11eb-9a5f-67b19983af5f.png)](https://www.youtube.com/watch?v=cSDATr7Ja28 "Deeplabv3+ Custom Training for DepthAI")

This semantic segmentation could be combined with depth information from DepthAI to get the full xyz position of every voxel of the entire visible portion of a person, for example.

## Supporting Notebooks

### Image Resize in Google Drive [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/GDrive-Resize.ipynb)

This notebook operates on your set of images in Google Drive to resize them to the format needed by the training notebooks.  See our documentation [here](https://docs.luxonis.com/tutorials/object_det_mnssv2_training/#step-1-find-or-generate-images-of-the-objects-of-interest) for more details.




