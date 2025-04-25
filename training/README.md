# 🚀 Training Tutorials


## 🌟 Overview

Here, we demonstrate how to train machine learning models for various tasks using both our libraries and third-party tools.

We primarily use [`luxonis-train`](https://github.com/luxonis/luxonis-train) for training AI models, along with datasets in the `LuxonisDataset` format provided by [`luxonis-ml`](https://github.com/luxonis/luxonis-ml/blob/main/luxonis_ml/data/README.md).

To augment datasets—either by generating synthetic data or annotating unlabeled images—we use [`DataDreamer`](https://github.com/luxonis/datadreamer).

Additionally, we cover how to train popular models such as:
- **YOLO (v5 to v8)** for object detection
- **DeepLabV3+** for semantic image segmentation
- A simple **neural network** for image classification

## 🔎 Table of Contents
- [🗂️ Prepare Data Using Luxonis-ML](#️-prepare-data-using-luxonis-ml)
  - [Luxonis-Train Dataset Preparation Using a Custom Generator Function](#luxonis-train-dataset-preparation-using-a-custom-generator-function-)
  - [Luxonis-Train Dataset Preparation Using Parsers](#luxonis-train-dataset-preparation-using-parsers-)
- [📚 Luxonis-Train Tutorials](#-luxonis-train-tutorials)
  - [Image Classification Training Tutorial](#image-classification-training-tutorial-)
  - [Object Detection Training Tutorial](#object-detection-training-tutorial-)
  - [Object Detection Training with DataDreamer Tutorial](#object-detection-training-with-datadreamer-tutorial-)
  - [Semantic Segmentation Training with Dataset Generation Tutorial](#semantic-segmentation-training-with-dataset-generation-tutorial-)
  - [Instance Segmentation Training with Dataset Generation Tutorial](#instance-segmentation-training-with-dataset-generation-tutorial-)
  - [Custom Segmentation Model Training Tutorial](#custom-segmentation-model-training-tutorial-)
  - [Custom Loader Detection VOC Training Tutorial](#custom-loader-detection-voc-training-tutorial-)
- [✨ Additional DataDreamer Tutorials](#-additional-datadreamer-tutorials)
  - [Object Detection Dataset Generation and YOLOv8 training](#object-detection-dataset-generation-and-yolov8-training-)
  - [Helmet Detection Dataset Generation and YOLOv8 training](#helmet-detection-dataset-generation-and-yolov8-training-)
  - [Instance Segmentation Dataset Generation and YOLOv8 training](#instance-segmentation-dataset-generation-and-yolov8-training-)
- [📘 Additional Tutorials](#-additional-tutorials)
  - [Image Classification Tutorials](#image-classification-tutorials)
    - [Simple Neural Network trained on image classification task using TensorFlow](#simple-neural-network-trained-on-image-classification-task-using-tensorflow-)
  - [Object Detection Tutorials](#object-detection-tutorials)
    - [YOLOv5 Object Detector Training](#yolov5-object-detector-training-)
    - [YOLOv6 Object Detector Training](#yolov6-object-detector-training-)
    - [YOLOv7 Object Detector Training](#yolov7-object-detector-training-)
    - [YOLOv8 Object Detector Training](#yolov8-object-detector-training-)
  - [Semantic Segmentation Tutorials](#semantic-segmentation-tutorials-)
    - [Deeplabv3 Plus Training](#deeplabv3-plus-training-)
  - [Instance Segmentation Tutorials](#instance-segmentation-tutorials)
    - [YOLO11 Training (Instance Segmentation)](#yolo11-training-)
  - [Pose Estimation Tutorials](#pose-estimation-tutorials)
    - [YOLO11 Training (Pose Estimation)](#yolo11-training--1)



**Luxonis-Train** is a user-friendly tool designed to streamline the training of deep learning models, especially for edge devices. To read the official documentation, please refer to [here](https://rvc4.docs.luxonis.com/software/ai-inference/model-source/training/luxonis-train/).

**DataDreamer** enables you to create annotated datasets from scratch using Generative AI and foundational Computer Vision models. This allows you to train your models for edge AI applications, such as object detection, without needing real-world data. Please refer to [here](https://rvc4.docs.luxonis.com/software/ai-inference/model-source/training/datadreamer/) to read the official documentation.

## 🗂️ Prepare Data Using Luxonis-ML

### Luxonis-Train Dataset Preparation Using a Custom Generator Function [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/dataset-preparation/custom_dataset_generator.ipynb)

**Description:**  
This tutorial demonstrates how to create a custom generator and use it to build a `LuxonisDataset` with [Luxonis-ML](https://github.com/luxonis/luxonis-ml/blob/main/luxonis_ml/data/README.md).  
The resulting dataset can then be used to train models with `luxonis-train`.

### Luxonis-Train Dataset Preparation Using Parsers [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/dataset-preparation/dataset_parsing.ipynb)

**Description:**  
This tutorial shows how to create a `LuxonisDataset` using built-in parsers from [Luxonis-ml](https://github.com/luxonis/luxonis-ml/blob/main/luxonis_ml/data/README.md).  
The resulting dataset can then be used to train models with `luxonis-train`.


## 📚 Luxonis-Train Tutorials

### Image Classification Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_classification_model.ipynb)

**Description:** This tutorial shows how to train an image classification model using **Luxonis-Train**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Object Detection Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_detection_model.ipynb)

**Description:** This tutorial shows how to train an object detection model using **Luxonis-Train**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Object Detection Training with DataDreamer Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_detection_model_datadreamer.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to annotate images for object detection, how to train an object detection model using **Luxonis-Train**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Semantic Segmentation Training with Dataset Generation Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_semantic_segmentation_model_datadreamer.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to enrich an existing dataset by generating more samples, train a semantic segmentation model using **Luxonis-Train**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Instance Segmentation Training with Dataset Generation Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_instance_segmentation_model.ipynb)

**Description:** This tutorial shows how to train an instance segmentation model using **Luxonis-Train**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.


### Custom Segmentation Model Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/luxonis_train_custom_model.ipynb)

**Description:**: This tutorial walks through creating a custom leaf segmentation model using **Luxonis-Train** and a Roboflow dataset. By leveraging **Luxonis-Train**’s flexible framework, we seamlessly combine new custom modules—like a UNet encoder-decoder and a specialized loss function—with existing components, then train, evaluate, and export the model (ONNX) for efficient deployment.


### Custom Loader Detection VOC Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/luxonis_train_custom_loader.ipynb)

**Description:** This tutorial demonstrates how to use a custom data loader based on `BaseLoaderTorch` to train object detection models on the VOC dataset using **LuxonisTrain**.


## ✨ Additional DataDreamer Tutorials

### Object Detection Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/datadreamer/generate_dataset_and_train_yolo.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to generate an object detection dataset featuring 100 images of robots, tractors, horses, cars, people, and bears. Furthermore, the tutorial shows how to split and convert the generated dataset into YOLO format and train an [YOLOv8n](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) model using the [Ultralytics](https://github.com/ultralytics/ultralytics) library. Finally, the tutorial shows how to use our [tools](https://tools.luxonis.com/) to convert the trained model for our RVC2/3 devices.

### Helmet Detection Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/datadreamer/helmet_detection.ipynb)

**Description:** This tutorial uses **DataDreamer** to generate a helmet detection dataset with 1000 images and [Ultralytics](https://github.com/ultralytics/ultralytics) to train an [YOLOv8m](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) model on it.

### Instance Segmentation Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/datadreamer/generate_instance_segmentation_dataset_and_train_yolo.ipynb)

**Description:** This tutorial showcases the **DataDreamer**'s ability to generate an instance segmentation dataset in YOLO format, so it's out of the box ready for [YOLOv8-seg](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) to be trained on.

## 📘 Additional Tutorials

### Image Classification Tutorials

#### Simple Neural Network trained on image classification task using Tensorflow [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/image-classification/OpenVINO_OpenCV_OAK_Tensorflow_Image_Classification.ipynb)

> [!Note]
> We are currently updating this tutorial. Some details may be outdated. We appreciate your understanding!

**Description:** This tutorial shows you how to train a simple image classification model using Tensorflow and then export it.

### Object Detection Tutorials

#### YOLOv5 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV5_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv5 model and then deploy it on your OAK model.

#### YOLOv6 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV6_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv6 model and then deploy it on your OAK model.

#### YOLOv7 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV7_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv7 model and then deploy it on your OAK model.

#### YOLOv8 Object Detector Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/object-detection/YoloV8_training.ipynb)

**Description:** This tutorial shows you how to train the popular YOLOv8 model and then deploy it on your OAK model.

### Semantic Segmentation Tutorials

#### Deeplabv3 Plus Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/semantic-segmentation/DeepLabV3plus_MNV2.ipynb)

**Description:** This tutorial shows you how to train semantic-segmentation networks with Deeplabv3+ with a MobileNetv2 backbone and then deploy it on your OAK model.

### Instance Segmentation Tutorials

#### YOLO11 Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/instance-segmentation/yolo11_instance_segmentation_training.ipynb)

**Description:** This tutorial shows you how to train the instance-segmentation YOLO11 nano model and deploy it on your OAK model.

### Pose Estimation Tutorials

#### YOLO11 Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/others/pose-estimation/yolo11_pose_estimation_training.ipynb)

**Description:** This tutorial shows you how to train the pose-estimation YOLO11 nano model and deploy it on your OAK model.

