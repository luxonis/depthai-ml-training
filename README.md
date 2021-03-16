## Check Out RoboFlow!

So RoboFlow also has several tutorials on training custom models for OAK and OAK-D.  See some below:

 - https://blog.roboflow.com/luxonis-oak-d-custom-model/
 - https://blog.roboflow.com/deploy-luxonis-oak/

## Start with the why!

We want you to be able to train your own neural models (object detectors) for DepthAI.  

This repository contains some pre-trained models under `model-zoo` and the Jupyter Notebooks we used to produce these neural models in `colab notebooks`.

Use the `model-zoo` models if we happened to already train something you need (and see [here](https://github.com/luxonis/depthai/tree/master/resources/nn) for models that Intel already trained that run on DepthAI already) and use the `colab notebooks` if you'd like to train something new or improve upon the training we've already done.

## What is DepthAI?

It's the platform for Embedded Spatial AI and Embedded AI.  Four trillion operations on a tiny device which can ouput structured data of what is in the physical world - what it is, and where it is in meters - in real-time.

## How do I train models?  What is Google Colaboratory?

Google Colaboratory is a great tool to do so!  It allows you to train neural models yourself using their fast GPU instances, and in some cases (depending on the dependencies of the training frameworks), even using the Google TPU - all for free!

It is exactly intended for this proof of concept and initial research.  And if you hit the limits of the free account, you can upgrade to a Pro version of Google Colab for only $10/month.

You can browse (and open/use) all the Jupyter Notebooks in this repo by clicking on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training).  




