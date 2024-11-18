# DepthAI ML Training

> [!IMPORTANT]  
> We are currently writing new tutorials and updating the old ones so that some details can be outdated. We appreciate your understanding!

<a name ="overview"></a>

## 🌟 Overview

This repository contains step by step tutorials on how to:

- Train your own Machine Learning (ML) model - either on default or custom dataset
- Convert the ML model so it's compatible with the [DepthAI](https://rvc4.docs.luxonis.com/software/) platform - conversion [docs here](https://rvc4.docs.luxonis.com/software/ai-inference/conversion/)

### What is DepthAI?

**DepthAI** is the platform for [Spatial AI](https://docs.luxonis.com/en/latest/pages/spatial-ai/#spatialai). 4 TOPS on a tiny device that can output structured data of what is in the physical world - what it is, and where it is in meters (XYZ) - in real-time.

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
    ├── Conversion/                         - Folder with tutorials about ML model conversion
    │   ├── ...
    │   └── README.md                       - Describes the conversion tutorials
    ├── Training                            - Folder with tutorials showing how to train a ML model
    │   ├── LuxonisTrain/                   - Folder with LuxonisTrain tutorials
    │   ├── Others/                         - Folder with tutorials featuring ML model training using other means than LuxonisTrain, such as Ultralytics or TensorFlow
    │   │   ├── ImageClassification
    │   │   ├── ObjectDetection
    │   │   └── SemanticSegmentation
    │   └── README.md                       - Describes all the training turorials
    ├── LICENSE 
    └── README.md                           - Describes the whole repository structure and key features

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

All the tutorials are licensed under the MIT license.

<a name="contributing"></a>

## 🤝 Contributing

We welcome contributions! Whether it's reporting bugs, improving documentation or tutorials, or adding new tutorials, your help is much appreciated. Please create a pull request ([here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)'s how to do it) and assign anyone from the Luxonis team to review the suggested changes. Cheers!