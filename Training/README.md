# Training Tutorials

Here, we show how to train ML models on various tasks using our libraries and others. We will mainly use [LuxonisTrain](https://github.com/luxonis/luxonis-train) for the training. To enlarge datasets by generating synthetic ones or annotating unlabeled images, we will use [DataDreamer](https://github.com/luxonis/datadreamer). In addition, we also present how to train popular models like YOLO (version V3 to v8) for object detection, DeeplabV3+ for semantic image segmentation, and a simple Neural Network for image classification.

**LuxonisTrain** is a user-friendly tool designed to streamline the training of deep learning models, especially for edge devices. To read the official documentation, please refer to [here](https://rvc4.docs.luxonis.com/software/ai-inference/model-source/training/luxonis-train/).

**DataDreamer** enables you to create annotated datasets from scratch using Generative AI and foundational Computer Vision models. This allows you to train your models for edge AI applications, such as object detection, without needing real-world data. Please refer to [here](https://rvc4.docs.luxonis.com/software/ai-inference/model-source/training/datadreamer/) to read the official documentation.

## LuxonisTrain Tutorials

Coming soon!

## Additional DataDreamer Tutorials

### Object Detection Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/datadreamer/blob/main/examples/generate_dataset_and_train_yolo.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to generate an object detection dataset featuring 100 images of robots, tractors, horses, cars, people, and bears. Furthermore, the tutorial shows how to split and convert the generated dataset into YOLO format and train an [YOLOv8n](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) model using the [Ultralytics](https://github.com/ultralytics/ultralytics) library. Finally, the tutorial shows how to use our [tools](https://tools.luxonis.com/) to convert the trained model for our RVC2/3 devices.

### Helmet Detection Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/datadreamer/blob/main/examples/helmet_detection.ipynb)

**Description:** This tutorial uses **DataDreamer** to generate a helmet detection dataset with 1000 images and [Ultralytics](https://github.com/ultralytics/ultralytics) to train an [YOLOv8m](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) model on it.

### Instance Segmentation Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/datadreamer/blob/main/examples/generate_instance_segmentation_dataset_and_train_yolo.ipynb)

**Description:** This tutorial showcases the **DataDreamer**'s ability to generate an instance segmentation dataset in YOLO format, so it's out of the box ready for [YOLOv8-seg](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) to be trained on.

## Additional Tutorials

### Image Classification Tutorials

#### Simple Neural Network trained on image classification task using Tensorflow [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/image-classification/OpenVINO_OpenCV_OAK_Tensorflow_Image_Classification.ipynb)

**Description:** This tutorial shows you how to train a simple image classification model using Tensorflow and then export it.

### Object Detection Tutorials

#### Tiny YoloV3 and YoloV4 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV3_V4_tiny_training.ipynb)

**Description:** This tutorial shows how to train an object detector based on the YoloV4-tiny and YoloV3-tiny networks. It describes the right data format, training steps, and provides further information.

#### YOLOv5 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV5_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv5 model and then deploy it on your OAK model.

#### YOLOv6 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV6_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv6 model and then deploy it on your OAK model.

#### YOLOv7 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV7_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv7 model and then deploy it on your OAK model.

#### YOLOv8 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV8_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv8 model and then deploy it on your OAK model.

### Semantic Segmentation Tutorials

#### Deeplabv3 Plus Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([qq](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/semantic-segmentation/DeepLabV3plus_MNV2.ipynb))

**Description:** This tutorial shows you how to train semantic-segmentation networks with Deeplabv3+ with a MobileNetv2 backbone. You can test the trained model with [this experiment](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_multiclass). Examples of such trained networks are below:

- Aerial Semantic Segmentation for UAV disaster recovery or automated landing ([video](https://www.youtube.com/watch?v=1EB5bpmUq1Q)).
- Person semantic segmentation ([video](https://www.youtube.com/watch?v=zjcUChyyNgI), [demo app](https://github.com/luxonis/depthai-experiments/tree/master/gen2-deeplabv3_person)).

<!-- ## Don't Want to Use Colab? Check Out RoboFlow! (Legacy)

The tutorials above are great if you are a do-it-yourself type. As all the code is there - and you can take it and run. But what if you're the "I just want a model that works type - don't care how" type - then checkout Roboflow. It has a clean and pain-free way to manage your dataset and do custom training.

RoboFlow also has several tutorials on training custom models for depthai, including OAK-1 and OAK-D, but the same training is applicable for any/all DepthAI devices.  See some below:

 - https://blog.roboflow.com/luxonis-oak-d-custom-model/
 - https://blog.roboflow.com/deploy-luxonis-oak/ -->

<!-- ## Check Out RoboFlow!

If you are new to the ML world, we suggest using [RoboFlow](https://roboflow.com/) for annotating, training and deploying  ML models, especially if you are new to the ML world. They have created a [course for the DepthAI platform](https://roboflow.com/course) that is a great starting point. They also have two blog posts, [quickstart guide](https://blog.roboflow.com/deploy-luxonis-oak/) and training [custom object detection model](https://blog.roboflow.com/luxonis-oak-d-custom-model/). -->
