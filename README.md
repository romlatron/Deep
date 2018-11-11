# DEEP: face detection
A deep learning powered face recognition app.
Written for the INSA Lyon DEEP Project by Martin Haug, Romain Latron, Anh Pham, Benoit Zhong.

## Goals
The goals of this project are:
 - select and prepare training and evaluation datasets
 - select an architecture for the neural networks
 - implement a detector for larger images
 - visualize and evaluate results

## Dataset
We used a greyscale dataset with 36x36 images of faces and non-faces (Google dataset). We added data from ImageNet and with automated searches on Bing.

## Neural networks
We tried several architectures to train on the dataset.
### Standard Neural Network
We adapted the neural network from the PyTorch tutorial [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to take 36x36 images as input.
This neural network is quite simple and can be run on CPUs.

### Advanced neural network
We added a convolution layer to the previous architecture. It remains simple and can be run on CPUs.

### GoogleNet/Inception
From the paper [Going Deeper with Convolution](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf).

We tried to implement one module of the GoogleNet to study the result.
Due to the complexity of this network, we need a GPU to run it (or some time with a CPU).

### Pre-trained models
PyTorch has a lot of pre-trained models: [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)

## Detector
To detect faces in bigger pictures, we first convert that picture into a greyscale image.
Then we create 36x36 boxes that we send to the neural network as an input.
To create those boxes, we go through the picture at different sizes (we reduce the size by a `shrink_factor` each time) and create boxes for every section of the image.
If the neural networks detect a face (value above a threshold), we create a rectangle patch for that square to display to the user.

### Intersections
As squares are overlapping, we can have intersections between squares. To solve that problem, we only keep the square with the highest score.

## Results and examples
TODO: add examples for same photo for each neural network

### Standard Neural Network
```
Accuracy of the network on the 10000 test images: 95 %
Precision of noface : 95 %
Recall of noface : 99 %
F1 score of noface : 97 %
Precision of  face : 98 %
Recall of  face : 57 %
F1 score of  face : 73 %
```

### Advanced neural network
Results on the validation dataset (training on the first dataset):
```
Accuracy of the network on the 10000 test images: 97 %
Accuracy of noface : 99 %
Accuracy of  face : 82 %
```

Results on the validation dataset (training on the new dataset):
```
Accuracy of the network on the 10000 test images: 98 %
Precision of noface : 98 %
Recall of noface : 99 %
F1 score of noface : 98 %
Precision of  face : 98 %
Recall of  face : 84 %
F1 score of  face : 90 %%
```

### GoogleNet/Inception
Results on the validation dataset:
```
Accuracy of the network on the 10000 test images: 95 %
Precision of noface : 95 %
Recall of noface : 99 %
F1 score of noface : 97 %
Precision of  face : 97 %
Recall of  face : 56 %
F1 score of  face : 71 %
```

### Pre-trained models
Todo

## Improvements ideas
Some improvements can include:
 - Collecting more data for the training
 - Have as many face and noface images
 - More layers for the neural network
 - Dataset with colors instead of greyscale
