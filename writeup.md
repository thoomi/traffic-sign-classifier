# **Traffic Sign Recognition**

This is the second project I am doing as part of Udacity's Self-Driving-Car Nanodegree. After learning the theoretical concepts of *neural networks* and *deep learning* I started exploring [TensorFlow](https://www.tensorflow.org/). Following that path I used the newly gained knowledge to build up a *deep neural network* in order to recognize 43 different German traffic signs. Utilizing different preprocessing and data augmentation techniques my final model got an accuracy of **98.6%** on the validation set and an accuracy of **96.0%** on the test set. See below for more detail :)


---


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
[image8]: ./examples/placeholder.png "Traffic Sign 5"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup & Project Files

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

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

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|


#### 4. Model Training

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used an ....

#### 5. Solution Design

Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

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
