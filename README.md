# DepthAI ML Training

<a name ="overview"></a>

## 🌟 Overview

This repository contains step by step tutorials on how to:

1. Train your own ML model - either on default or custom dataset
2. Convert the ML model so it's compatible with the [DepthAI](https://rvc4.docs.luxonis.com/software/) platform - conversion [docs here](https://rvc4.docs.luxonis.com/software/ai-inference/conversion/)
3. Deploy the ML model to the OAK camera, so it runs on the accelerated hardware on the edge

<!-- ## Check Out RoboFlow!

If you are new to the ML world, we suggest using [RoboFlow](https://roboflow.com/) for annotating, training and deploying  ML models, especially if you are new to the ML world. They have created a [course for the DepthAI platform](https://roboflow.com/course) that is a great starting point. They also have two blog posts, [quickstart guide](https://blog.roboflow.com/deploy-luxonis-oak/) and training [custom object detection model](https://blog.roboflow.com/luxonis-oak-d-custom-model/). -->

### What is DepthAI?

 `DepthAI` is the platform for [Spatial AI](https://docs.luxonis.com/en/latest/pages/spatial-ai/#spatialai). 4 TOPS on a tiny device which can ouput structured data of what is in the physical world - what it is, and where it is in meters (XYZ) - in real-time.


## 📜 Table of contents

- [🌟 Overview](#overview)
- [📂 Repository Structure](#repo-structure)
- [⚙️ Where to run](#run-env)
- [⚠️ Limitations](#limitations)
- [📄 License](#license)
- [🤝 Contributing](#contributing)

<a name="repo-structure"></a>

## 📂 Repository Structure

    depthai-ml-training/
    ├── datadreamer/                    - Folder with tutorials featuring DataDreamer
    │   ├── ...
    │   └── README.md                   - Short description of the DataDreamer and and links to github repository, documentation and all tutorials
    ├── luxonis-train/                  - Folder with tutorials featuring LuxonisTrain
    │   ├── ...
    │   └── README.md                   - Short description of the LuxonisTrain and and links to github repository, documentation and all tutorials
    ├── modelconverter/                 - Folder with tutorials featuring ModelConverter
    │   ├── ...
    │   └── README.md                   - Short description of the ModelConverter and links to github repository, documentation and all tutorials
    ├── others/                         - Folder containing tutorials that aren't featuring any Luxonis library
    │   ├── image-classification/       - Folder for all image classification training tutorials
    │   ├── object-detection/           - Folder for all object detection training tutorials (YOLO)
    │   ├── semantic-segmentation/      - Folder for all semantic segmentation training tutorials
    │   └── README.md                   - Short description of all available tutorials
    ├── LICENSE 
    └── README.md                       - Description of the whole repository structure, introducing key libraries and terms, etc.

<a name="run-env"></a>

## ⚙️ Where to run

### What is Google Colaboratory?

Google Colaboratory allows you to train neural models yourself using their fast GPU instances, and in some cases (depending on the dependencies of the training frameworks), even using the Google TPU - all for free!

It is exactly intended for this proof of concept and initial research.  And if you hit the limits of the free account, you can upgrade to a Pro version of Google Colab for only $10/month.

You can browse (and open/use) all the Jupyter Notebooks in this repo by clicking on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training).

<a name="limitations"></a>

## ⚠️ Limitations

TBA

<a name="license"></a>

## 📄 License

TBA

<a name="contributing"></a>

## 🤝 Contributing

TBA