# DEEP: A face detection solution
Face recognition app powered by deep learning.
Written for the INSA Lyon DEEP Project by Martin Haug, Romain Latron, Anh Pham, Benoit Zhong.

## User Instructions
### Detection
Run the `deep-script.py` file on a Python3 interpreter. You always have to specify one image path as an argument, this is the image that the face detection will run on. The image may be stored in any of the formats accepted by PIL (JPEG, GIF, PNG, PGM, ...) and should either be in color or grayscale. In order to use the provided model to scan for faces, specify `-m model` as an additional argument. You can, of course, use your own model files instead of our file.

The application will run on the image and output a visualization in a new window. Faces will be shown by the circles drawn around them. Green circles indicate that the match confidence is high and backed by multiple detections. Yellow circles in varying shades show additional hits, where less saturated shades mean a lower confidence.

If you feel like there are too few matches, you should specify the `--outliers` switch. That way, matches that would otherwise have been discarded are displayed as squares (the redder, the more confident). While this mode may result in more matches, especially in images where the faces are small it also augments the possibility for false positives.

By default, the model converts the images to grayscale to match the training images. You can use the `--color` switch to run the network on the original (colored) image instead.  Our tests - while not exhaustive - show similar result quality.

### Training
Run the `deep-script.py` file on a Python3 interpreter with the `-t path/to/training` and `-e path/to/test` arguments specified. Your model will be saved at the path specified with the `-m new_model` argument. You still have to provide an image path as an unnamed argument, however, the file does not have to exist for the training to complete successfully. If you omit the model path, the training will run anyway, evaluate on the test set, search faces on the input image and terminate without saving the trained model. There is an `--epoch n` switch (defaults to 15) with which you can change the epoch count.

Please note that both your test and training folders must contain two subfolders named `0` containing images without faces and `1` containing images with faces. Ensure that there are no other subdirectories.

### Evaluation
In addition to detecting images, you can batch test your own validation set of images. Call the program with the arguments `deep-script.py -e path/to/test -m model_to_test detector_image.jpg` to obtain output like ours in the section results and examples.

### Other remarks
The script depends on several packages. If you can't run it because some of them are not found, try to run `pip install numpy matplotlib skikit-learn Pillow` and install PyTorch stable using [these instructions](https://pytorch.org/get-started/locally/).
If you got a CUDA-compatible NVidia GPU and the matching software the application will use your GPU to accelerate execution. If you think you have a compatible setup but the program tells you that it runs on CPU try to update your display drivers, [CUDA software](https://developer.nvidia.com/cuda-downloads) and ensure that you installed the right version of PyTorch.

An overview of the command line arguments may be obtained by calling `deep-script.py -h`.

## Goals
The goals of this project are:
 - select and prepare training and evaluation datasets
 - select an architecture for the neural networks
 - implement a detector for larger images
 - visualize and evaluate results

## Dataset
We used a greyscale dataset with 36x36 images of faces and non-faces. We added data from ImageNet and with automated downloads from the Microsoft Azure Bing Search v7 API and Wikimedia Commons because we felt that the provided training set for the non-face class had too many macro images, likely obtained through data augmentation on very few images.

Because the network runs on more zoomed-out imagery during detection this was not sufficient for training our models adequately. With the added landscape, building, structure, and body shots the performance significantly increased. Fortunately, the non-face category is the easier one to collect images for because there is almost no manual labeling involved.

## Neural networks
We tried several architectures to train on the dataset.
### Standard Neural Network
We adapted the neural network from the PyTorch tutorial [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to take 36x36 images as input.
This neural network is quite simple and can be run on CPUs.

### Advanced neural network
We added a convolution layer to the previous architecture. It remains simple and can be run on CPUs.

### GoogleNet/Inception
From the paper [Going Deeper with Convolution](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf).

We tried to implement one Inception-module of the GoogleNet to study the result.
Due to the complexity of this network, we need a GPU to train it (or plenty of time with a CPU). 

### Pre-trained models
We used ResNet18 from the [pre-trained and pre-implemented models](https://pytorch.org/docs/stable/torchvision/models.html) that ship with TorchVision. After downloading a model which is trained to the ImageNet 1000 dataset we delete the last fully connected layer and replace it with a randomly initialized one with two exits. Starting with the downloaded parameters, we continue by training with our own dataset. In this iteration of our program, we added an evaluation at the end of each epoch: The model with the best accuracy on the test set will be saved temporarily until an epoch produces a better one. Finally, the best model is returned and saved. That way we are able to limit overfitting.

## Detector
To detect faces in bigger pictures, we first convert that picture into a greyscale image unless the color switch is specified.
Then we create 226x226 crops by sliding a window of decreasing size over the image and taking overlapping snapshots that we use as input for the neural network. The crop area on the original image, however, can be a lot smaller than 226x226 - the minimum border length is calculated in accordance with the number of pixels in the image. This makes the crop creation somewhat dynamic: If a square takes up a very small surface ratio on a high-resolution image, it is unlikely that it contains a face - by not checking it false positives are avoided and performance is significantly improved.

The neural network matches are then post-processed: We ignore all matches below a certain score. We fine-tuned this threshold for each model individually for maximal effectivity. After that, we run a DBSCAN clustering algorithm on our results, treating the matches as a cloud of points. Their size is being taken into account by calculating the eps value for DBSCAN from the median of the match border lengths. That way, more points will be clustered on an image where the faces take up a large part of the surface. This method accounts better for the variance of face size than using the image resolution to calculate the parameter.

Each calculated cluster will then be overlayed as a green circle onto the image - their center is the respective cluster centroid and their radius is calculated by an average of the width and height of the bounding box of that cluster.

After that, significant outliers are added to the image as yellow circles: We calculate the median of the outlier scores. If it is above a threshold value, we add all the outliers with a score greater than that median. Overlapping circles are discarded by calculating circle intersections and if one circle contains the other for each 2-combination of circles. These overlaps are than saved as graph edges between nodes (the circles). We then group the graph into connected subgraphs and render the item with the max score as a representative of that class.

Finally, the other outliers are rendered as red squares if the switch is specified.

## Results and examples

### Standard Neural Network
Results on the validation dataset:
```
Accuracy of the network on the 10000 test images: 95 %
Precision of noface : 95 %
Recall of noface : 99 %
F1 score of noface : 97 %
Precision of  face : 98 %
Recall of  face : 57 %
F1 score of  face : 73 %
```
![alt text](https://github.com/romlatron/Deep/blob/master/results_previous.png)
### Advanced neural network

Results on the validation dataset:
```
Accuracy of the network on the 10000 test images: 98 %
Precision of noface : 98 %
Recall of noface : 99 %
F1 score of noface : 98 %
Precision of  face : 98 %
Recall of  face : 84 %
F1 score of  face : 90 %
```
![alt text](https://github.com/romlatron/Deep/blob/master/results_advanced.png)
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
![alt text](https://github.com/romlatron/Deep/blob/master/results_inception.png)
### ResNet18 and transfer learning
Results on the validation dataset:
```
Accuracy of the network on the 10000 test images: 99 %
Precision of noface : 99 %
Recall of noface : 99 %
F1 score of noface : 99 %
Precision of  face : 99 %
Recall of  face : 96 %
F1 score of  face : 98 %
```
![alt text](https://github.com/romlatron/Deep/blob/master/results_transferLearning.png)
## Improvements ideas
Some improvements can include:
 - Collecting more data for the training, notabely color photos and faces
 - Parity between class image count in training set
 - Try other models
