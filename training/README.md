# Training Tutorials

Here, we show how to train ML models on various tasks using our libraries and others. We will mainly use [LuxonisTrain](https://github.com/luxonis/luxonis-train) for the training. To enlarge datasets by generating synthetic ones or annotating unlabeled images, we will use [DataDreamer](https://github.com/luxonis/datadreamer). In addition, we also present how to train popular models like YOLO (version V5 to v8) for object detection, DeeplabV3+ for semantic image segmentation, and a simple Neural Network for image classification.

**LuxonisTrain** is a user-friendly tool designed to streamline the training of deep learning models, especially for edge devices. To read the official documentation, please refer to [here](https://rvc4.docs.luxonis.com/software/ai-inference/model-source/training/luxonis-train/).

**DataDreamer** enables you to create annotated datasets from scratch using Generative AI and foundational Computer Vision models. This allows you to train your models for edge AI applications, such as object detection, without needing real-world data. Please refer to [here](https://rvc4.docs.luxonis.com/software/ai-inference/model-source/training/datadreamer/) to read the official documentation.

## LuxonisTrain Tutorials

### Image Classification Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_classification_model.ipynb)

**Description:** This tutorial shows how to train an image classification model using **LuxonisTrain**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Object Detection Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_detection_model.ipynb)

**Description:** This tutorial shows how to train an object detection model using **LuxonisTrain**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Object Detection Training with DataDreamer Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_detection_model_datadreamer.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to annotate images for object detection, how to train an object detection model using **LuxonisTrain**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Semantic Segmentation Training with Dataset Generation Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_semantic_segmentation_model_datadreamer.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to enrich an existing dataset by generating more samples, train a semantic segmentation model using **LuxonisTrain**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.

### Instance Segmentation Training with Dataset Generation Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/train_instance_segmentation_model.ipynb)

**Description:** This tutorial shows how to train an instance segmentation model using **LuxonisTrain**, its conversion using **ModelConverter**, and how to run the trained model on a Luxonis device using **DepthAI v3**.


### Custom Segmentation Model Training Tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/luxonis_train_custom_model.ipynb)

**Description:** This tutorial walks through creating a custom leaf segmentation model using **LuxonisTrain** and a Roboflow dataset. By leveraging **LuxonisTrain**’s flexible framework, we seamlessly combine new custom modules—like a UNet encoder-decoder and a specialized loss function—with existing components, then train, evaluate, and export the model (ONNX) for efficient deployment.


## Additional DataDreamer Tutorials

### Object Detection Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/datadreamer/generate_dataset_and_train_yolo.ipynb)

**Description:** This tutorial shows how to use **DataDreamer** to generate an object detection dataset featuring 100 images of robots, tractors, horses, cars, people, and bears. Furthermore, the tutorial shows how to split and convert the generated dataset into YOLO format and train an [YOLOv8n](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) model using the [Ultralytics](https://github.com/ultralytics/ultralytics) library. Finally, the tutorial shows how to use our [tools](https://tools.luxonis.com/) to convert the trained model for our RVC2/3 devices.

### Helmet Detection Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/datadreamer/helmet_detection.ipynb)

**Description:** This tutorial uses **DataDreamer** to generate a helmet detection dataset with 1000 images and [Ultralytics](https://github.com/ultralytics/ultralytics) to train an [YOLOv8m](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) model on it.

### Instance Segmentation Dataset Generation and YOLOv8 training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/main/training/datadreamer/generate_instance_segmentation_dataset_and_train_yolo.ipynb)

**Description:** This tutorial showcases the **DataDreamer**'s ability to generate an instance segmentation dataset in YOLO format, so it's out of the box ready for [YOLOv8-seg](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) to be trained on.

## Additional Tutorials

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

