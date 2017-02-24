# **Traffic Sign Recognition**

This is the second project I am doing as part of Udacity's Self-Driving-Car Nanodegree. After learning the theoretical concepts of *neural networks* and *deep learning* I started exploring [TensorFlow](https://www.tensorflow.org/). Following that path I used the newly gained knowledge to build up a *deep neural network* in order to recognize 43 different German traffic signs. Utilizing different preprocessing and data augmentation techniques my final model got an accuracy of **98.6%** on the validation set and an accuracy of **96.0%** on the test set. See below for more detail :)


**The goals / steps of this project are the following:**
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/chart_count_classes.png "Bar chart"
[image2]: ./images/random_examples.png "Random examples"
[image3]: ./images/grayscale.png "Grayscale"
[image4]: ./images/equalize_histogram.png "Equalize histogram"
[image5]: ./images/normalized.png "Center normalize"
[image6]: ./images/transform1.png "Transformation 1"
[image7]: ./images/transform2.png "Transformation 2"
[image8]: ./images/model_scheme.png "Model scheme"


# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! Following below I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/traffic-sign-classifier) and if you're looking specifically for the projects code, [here is a link to the implementation](https://github.com/thoomi/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb) in a IPython notebook.


---

### Data Set Summary & Exploration

#### 1. Basic Dataset Summary

The code for this step is contained in the second code cell of the IPython notebook.  

I mainly used the numpy library to calculate summary statistics of the traffic
signs data set:

* Size of the training set: **34799**
* Size of the validation set: **4410**
* Size of the test set: **12630**
* Shape of a traffic sign image: **(32, 32, 3)**
* Number of unique classes/labels:  **43**
* Mean of the training set: **82.677**
* Standard deviation of the training set: **67.850**


#### 2. Exploratory Dataset Visualization

The code for this step is contained in the third code cell of the IPython notebook.  

Here is a bar chat visualizing the number of examples for each class. Some classes are obviously very underrepresented. In order to help the model to generalize on those classes one needs to increase their number of examples (exp. through data augmentation).

![Bar chart showing count of individual classes][image1]

Below is a visualization of one example for each individual class. As we can see, the pictures have very different brightness values and possible need to be normalized.

![Random pick of traffic sign images][image2]

---

### Design and Test a Model Architecture

#### 1. Preprocessing

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because as Sermanet & LeCun show in their [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the color makes almost no difference in the networks accuracy and as a side effect we also reduce the computation cost.

Here is an example of a traffic sign image before and after grayscaling:

![Color to grayscale][image3]

As a second step, the images histogram has been equalized in order to get a constant brightness level of all images. The example below shows the conversion from grayscale to an equalized histogram:

![Grayscale with equalized histogram][image4]

The last step normalizes the images by subtracting the mean and dividing by the standard deviation as LeCun suggests in his paper about [efficient backpropagation](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). See below an example:

![Normalized image][image5]


#### 2. Training Data Augmentation

The fifth code cell of the IPython notebook contains the code for augmenting the data set.

I decided to generate additional data because some classes in the training data set are underrepresented. In order to help the network to generalize and correctly classify those classes i augmented the original data set by generating new images with small and random transformations (translate, rotate). At different stages in the process of finding the right parameters i tried various numbers and finally settled with four additional images for each original image. This gives a good compromise between accuracy and training speed.

Here are some examples of original images and their augmented counterparts:

![Transformation examples][image6]
![Transformation examples][image7]

In summation, this leads to a new training set size of **173995** and the parameters for the transformations are as follows:

* Translation: +- 2 Pixel (x and y direction)
* Rotation: +- 10 Degree

#### 3. Model Architecture

The code for my final model is located in the sixth cell of the IPython notebook.

My final model consisted of the following layers and is highly inspired by the Sermanet & LeCun model:

![Transformation examples][image8]

And here is a table with a little bit more detail:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x512    |
| RELU					|		 										|
| Fully connected		| flattend, combined L1 + L1 + L3		 	    |
| RELU					|		 										|
| Droput				| Keep probability: 0.5		 			    	|
| Fully connected  		| Outputs 43									|
| Softmax       		|              									|


#### 4. Model Training

The code for training the model is located in the seventh cell of the IPython notebook.

To train the model, i used the **Adam optimizer** and the following hyperparameters:

* Learning rate: **0.001**
* Batch size: **100**
* Epochs: **50**
* Weight initialization with truncated normal distribution with **mu = 0** and **sigma = 0.1**


#### 5. Solution Design

The code for calculating the accuracy of the model is located in the seventh cell of the Ipython notebook.

I started of with a standard LeNet5 architecture and tried to tune the hyperparameters from there. First with only grayscaling the images and after that I tried various input normalization techniques which gave me a about 1% higher accuracy. The further equalization of the images histogram, to decouple the model from brightness effects, added another 1-2 % accuracy on the validation set.  

After playing a lot with the hyperparameters and not getting results above 95% I decided to increase the convolution layer sizes and to connect the convolution outputs of each layer to a big fully connected layer. Finally i got results above 97 %. To prevent the model from over fitting because of this big last layer, I added a dropout layer with a keep probability of 50 %.

My final model results were:
* Validation set accuracy of **98.6%**
* Test set accuracy of **96.0%**


I kept track of the process in order to help me finding good parameters and not
trying everything twice. Here is the log book of my progress torwards the final model:


16.02.2017  

    Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    Epochs: 30 | Batchsize: 150 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: no
    Preprocessing: no
    Validation Accuracy: 0.892

16.02.2017

    Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    Epochs: 30 | Batchsize: 150 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: (X_train / 255.0) - 0.5
    Preprocessing: no
    Validation Accuracy: 0.921

17.02.2017

    Standard LeNet5 (C32 - C64 - FC120 - FC84 - SM43)
    Epochs: 30 | Batchsize: 200 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: (X_train / 255.0) - 0.5
    Preprocessing: grayscale
    Validation Accuracy: 0.931

18.02.2017

    Standard LeNet5 (C32 - C64 - FC256 - FC84 - SM43)
    Epochs: 30 | Batchsize: 200 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: subtract mean, divide by std deviation
    Preprocessing: grayscale, equalized Histogram
    Validation Accuracy: 0.957


20.02.2017

    Standard LeNet5 (C32 - C64 - FC256 - FC84 - SM43)
    Epochs: 30 | Batchsize: 200 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: subtract mean, divide by std deviation
    Preprocessing: grayscale, equalized Histogram, 4 x additional data (rotated, translated)
    Validation Accuracy: 0.969


22.02.2017

    Standard LeNet5 (C32 - C64 - FC512 - FC84 - SM43)
    Epochs: 30 | Batchsize: 100 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: subtract mean, divide by std deviation
    Preprocessing: grayscale, equalized Histogram, 4 x additional data (rotated, translated)
    Validation Accuracy: 0.973

22.02.2017

    Advanced LeNet5 (C16 - C32 - FC512 - SM43) [connecting both convolution outputs to the FC512]
    Epochs: 30 | Batchsize: 100 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: subtract mean, divide by std deviation
    Preprocessing: grayscale, equalized Histogram, 4 x additional data (rotated, translated)
    Validation Accuracy: 0.986

---


### Test a Model on New Images

#### 1. Acquiring New Images

Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to any particular qualities of the images or traffic signs in the images that may be of interest, such as whether they would be difficult for the model to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Performance on New Images

Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail
as described in the "Stand Out Suggestions" part of the rubric).

The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Model Certainty - Softmax Probabilities

The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
