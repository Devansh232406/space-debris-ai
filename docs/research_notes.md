# Research Log

Day 1 – First CNN Implementation

Implemented a convolutional neural network in PyTorch and trained it on the CIFAR-10 dataset.

Key concepts explored

* Image representation as tensors
* DataLoader batching mechanism
* Convolutional layers for feature extraction
* Training loops with backpropagation
* Model evaluation using test accuracy

This experiment established the deep learning pipeline that will later be adapted for satellite imagery analysis and space debris detection.

Day 2 – Satellite Data Exploration

Explored the characteristics of satellite imagery and remote sensing datasets.

Key concepts studied

* Satellite images contain large scenes with multiple objects.
* High resolution imagery requires object detection models rather than simple classification.
* Image normalization and data augmentation improve model performance.

Experiments performed

* Loaded and visualized image datasets using PyTorch.
* Applied normalization and augmentation transforms.
* Analyzed image tensor structure.

Observations

Object detection models such as YOLO will likely be required to detect small debris objects in satellite imagery.


Day 3 Update

Implemented a bounding box visualization experiment to understand how object detection models represent object locations.

Used a CIFAR10 image and manually drew a bounding box using matplotlib patches to simulate detection output.

This experiment helped illustrate how detection models produce spatial predictions alongside class labels.

