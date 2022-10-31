
## Overview
Here we have examples of Google Colaboratory (aka Colab or simply colabs) notebooks trained on various datasets. They are free GPU instances, so great for prototyping and even simple production models.

## Don't Want to Use Colab?  Check Out RoboFlow!

The tutorials below are great if you are a do-it-yourself type.  As all the code is there - and you can take it and run.  But what if you're the "I just want a model that works type - don't care how" type - then checkout Roboflow.  It has a clean and pain-free way to manage your dataset and do custom training.

RoboFlow also has several tutorials on training custom models for depthai, including OAK-1 and OAK-D, but the same training is applicable for any/all DepthAI devices.  See some below:

 - https://blog.roboflow.com/luxonis-oak-d-custom-model/
 - https://blog.roboflow.com/deploy-luxonis-oak/


## The Tutorials
The below tutorials cover MobileNetv2-SSD, tiny-YOLOv3, tiny-YOLOv4, YOLOv5, YOLOv7 and Deeplabv3+ (semantic segmentation).  A bunch of other object detectors and neural networks could be trained/supported on Colab and run on DepthAI, so if you have a request for a different object detector/network backend, please feel free to make a Github Issue!  Select a type of model to train from below, and get it training.

After training your own model, you can run it on DepthAI with the following:

## Running the model 
Now we use the created model, in its .blob format, to run it on DepthAI. In your DepthAI folder, go to the `resources/nn` directory. There you will see a folder called `mobilenet-ssd`. Since your model is also a mobilenet ssd, make a copy of that folder and rename it as you see fit for your model. If you trained a model for dog breeds, you could call it dog_detective. Enter the folder and delete the mobilenet-ssd.blob. Paste here your own blob from above and give it the same name as the folder, i.e. 'dog_detective.blob'. Also rename the .json files to match the folder and blob name, i.e. 'dog_detective.json' and 'dog_detective_depth.json'.

In both the .json files you will see a category called labels. Those are the default mobilenet-ssd labels trained on Pascal '07 dataset. Modify the labels according to the labels for your trained model. So if you had 4 dog breeds in the dog_detective, write those 4 breeds in the order you have them in your training data labels. Of course, remove all the unnecessary labels, except for "background". This label must remain and always on top. You should end up with "background" + your labels, i.e. 5 labels in total for the dog_detective. If the model consistently predicts a poodle as a husky for example, it most likely means your labels are not in the correct order. Don't worry, just change the order of the labels to match the detections.

Remember, you have to change the labels in both .json files.

You are ready to run your very own trained model on DepthAI. Open a terminal in your DepthAI directory and run:
`python3 depthai_demo.py -cnn <your folder name>` i.e. `python3 depthai_demo.py -cnn dog_detective`

If you wish to disable the spatial measurements associated with the detected objects, run the comand with the '-dd' (disable depth) flag.
`python3 depthai_demo.py -dd -cnn <your folder name>`


### Tiny YoloV3 and YoloV4 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb)

These tutorial shows how to train an object detector based on the YoloV4-tiny and YoloV3-tiny networks. It describes the right data format, training steps, and provides further information.

![TinyYoloV4Example](https://user-images.githubusercontent.com/56075061/143061151-07157024-4189-420d-b603-2cb3ec926bf5.png)

### YOLOv5 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb)
N
This tutorial shows you how to train the popular YOLOv5 model and then deploy it on your OAK model.

![asdfasdfdsasdf](https://user-images.githubusercontent.com/32992551/135535957-4c945ff0-1ed7-43c8-8be5-f2efebb84f19.png)

### YOLOv6 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV6_training.ipynb)

This tutorial shows you how to train the popular YOLOv6 model and then deploy it on your OAK model.

### YOLOv7 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV7_training.ipynb)

This tutorial shows you how to train the popular YOLOv7 model and then deploy it on your OAK model.

### MobileNetSSDv2 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_Object_Detection_With_Custom_Data_Demo_Training.ipynb)
The tutorial notebook 
*Easy_Object_Detection_With_Custom_Data_Demo_Training.ipynb* shows how to quickly train an object detector based on the Mobilenet SSDv2 network. 

Optionally, see our documentation around this module ([here](https://docs.luxonis.com/tutorials/object_det_mnssv2_training/)) for of a guide/walk-through on how to use this notebook.  Also, feel free to jump right into the Notebook, with some experimentation it's relatively straightforward to get a model trained.

After training is complete, it also converts the model to a .blob file that runs on our DepthAI platform and modules. First the model is converted to a format usable by OpenVINO called Intermediate Representation, or IR. The IR model is then compiled to a .blob file using a server we set up for that purpose. (The IR model can also be [converted locally to a blob](https://github.com/luxonis/depthai#conversion-of-existing-trained-models-into-intel-movidius-binary-format).)

And that's it, in less than a couple of hours a fairly advanced proof of concept object detector can run on DepthAI to detect objects of your choice and their associated spatial information (i.e. xyz location). For example this notebook was used to train DepthAI to locate strawberries in 3D space, see below:

[![Real-time 3D Strawberry Detector](https://user-images.githubusercontent.com/5244214/90794705-6104b580-e30d-11ea-8275-468b300b4659.gif)](https://www.youtube.com/watch?v=Okjh2OCP-o8& "Real-Time Spatial AI to Pick Strawberries")

The above example used a DepthAI Modular Cameras Edition ([BW1098FFC](https://shop.luxonis.com/products/depthai-usb3-edition)).

### Deeplabv3 Plus Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/DeepLabV3plus_MNV2.ipynb)

This tutorial shows you how to train semantic-segmentation networks with Deeplabv3+ with a MobileNetv2 backbone. 

You can test the trained model with [this experiment](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass).

 Examples of such trained networks are below:

#### Aerial Semantic Segmentation for UAV disaster recovery or automated landing:
[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/102667886-c182f280-4147-11eb-904c-c8c89d9706a7.png)](https://www.youtube.com/watch?v=1EB5bpmUq1Q "Deeplabv3+ Custom Training for DepthAI")

#### Person semantic segmentation
[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/109359126-25a9ed00-7842-11eb-9071-cddc7439e3ca.png)](https://www.youtube.com/watch?v=zjcUChyyNgI "Deeplabv3+ Custom Training for DepthAI")

This semantic segmentation could be combined with depth information from DepthAI to get the full xyz position of every voxel of the entire visible portion of a person, for example.

See this Gen2 API example for running this on DepthAI:
https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_person


## Supporting Notebooks

### Image Resize in Google Drive [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/GDrive-Resize.ipynb)

This notebook operates on your set of images in Google Drive to resize them to the format needed by the training notebooks.  See our documentation [here](https://docs.luxonis.com/tutorials/object_det_mnssv2_training/#step-1-find-or-generate-images-of-the-objects-of-interest) for more details.

## Deprecated tutorials

* Old YoloV3-tiny tutorial
    [Open In Colab](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYolov3_Object_Detector_Training_on_Custom_Data.ipynb)
* Old YoloV4-tiny tutorial
    [Open In Colab](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYOLOv4_Object_Detector_Training_on_Custom_Data.ipynb)
* Old MobileNetSSDv2 tutorial 
    [Open In Colab](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_Object_Detection_Demo_Training.ipynb)
* Old MobileNetSSDv2 tutorial for medical masks 
    [Open In Colab](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/Medical_Mask_Detection_Demo_Training.ipynb)
