## Overview
Here we have examples of Google Colaboratory (aka Colab or simply colabs) notebooks trained on various datasets. They are free GPU instances, so great for prototyping and even simple production models.

## The Tutorials
The below tutorials are based on MobileNetv2-SSD, which is a decent-performance, decent-framework object dectector which natively runs on DepthAI.  A bunch of other object detectors could be trained/supported on Colab and run on DepthAI, so if you have a request for a different object detector/network backend, please feel free to make a Github Issue!

### Easy Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Corona%20Mask%20Detection%20Demo%20Training.ipynb)
The tutorial notebook 
*Easy_Object_Detection_With_Custom_Data_Demo_Training.ipynb* shows how to quickly train an object detector based on the Mobilenet SSDv2 network. 

After training is complete, it also converts the model to a .blob file that runs on our DepthAI platform and modules. First the model is converted to a format usable by OpenVINO called Intermediate Representation, or IR. The IR model is then compiled to a .blob file using a server we set up for that purpose. (The IR model can also be [converted locally to a blob](https://github.com/luxonis/depthai#conversion-of-existing-trained-models-into-intel-movidius-binary-format).)

And that's it, in less than a couple of hours a fairly advanced proof of concept object detector can run on DepthAI to detect objects of your choice and their associated spatial information (i.e. xyz location). For example this notebook was used to train DepthAI to locate strawberries in 3D space, see below:

[![Real-time 3D Strawberry Detector](http://img.youtube.com/vi/CP-nM6CjhU4/0.jpg)](https://www.youtube.com/watch?v=CP-nM6CjhU4 "Real-Time Spatial AI to Pick Strawberries")

The above example used a DepthAI Modular Cameras Edition ([BW1098FFC](https://shop.luxonis.com/products/depthai-usb3-edition)).

### COVID-19 Mask/No-Mask Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Corona%20Mask%20Detection%20Demo%20Training.ipynb)
The *Corona Mask Detection Demo Training.ipynb* training notebook shows another example of a more complex object detector. The training data set consists of people wearing or not wearing masks for viral protection. There are almost 700 pictures with approximately 3600 bounding box annotations. The images are complex: they vary quite a lot in scale and composition. Nonetheless, the object detector does quite a good job with this relatively small dataset for such a task. Again, training takes around 2 hours. Depending on which GPU the Colab lottery assigns to the notebook instance, training 10k steps can take 2.5 hours or 1.5 hours. Either way, a short period for such a good quality proof of concept for such a difficult task. 
We then performed the steps above for converting to blob and then running it on our DepthAI module. 

Below is a quick test of the model produced with this notebook on Luxonis DepthAI Onboard Cameras Edition ([BW1098OBC](https://shop.luxonis.com/products/bw10980bc)):

[![COVID19 Mask Detector](http://img.youtube.com/vi/d_oUxDzWHd0/0.jpg)](https://www.youtube.com/watch?v=d_oUxDzWHd0 "COVID19 Mask/No-Mask")


