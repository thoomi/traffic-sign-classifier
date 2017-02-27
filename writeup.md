# **Traffic Sign Recognition**

This is the second project I am doing as part of Udacity's Self-Driving-Car Nanodegree. After learning the theoretical concepts of *neural networks* and *deep learning* I started exploring [TensorFlow](https://www.tensorflow.org/). Following that path I used the newly gained knowledge to build a *deep neural network* in order to recognize 43 different German traffic signs. Utilizing different preprocessing and data augmentation techniques my final model got an accuracy of **98.7%** on the validation set and an accuracy of **96.0%** on the test set. See below for more detail :)


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
[image9]: ./images/chart_count_classes_augmented.png "Bar char augmented"


# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! Following below I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/traffic-sign-classifier) and if you're looking specifically for the projects code, [here is a link to the implementation](https://github.com/thoomi/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb) as IPython notebook.


---

### Data Set Summary & Exploration

#### 1. Basic Dataset Summary

The code for this step is contained in the second code cell of the IPython notebook.  

I mainly used the numpy library to calculate summary statistics of the [traffic
sign data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip):

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

Below is a visualization of one example for each individual class. As we can see, the pictures have very different brightness values and possibly need to be normalized.

![Random pick of traffic sign images][image2]

---

### Design and Test a Model Architecture

#### 1. Preprocessing

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale. As Sermanet & LeCun show in their [paper][paper02], the color makes almost no difference on the models accuracy because of the similar colors and shapes between the traffic signs. A positive side effect of this is the reduced computation cost.

Here is an example of a traffic sign image before and after grayscaling:

![Color to grayscale][image3]

As a second step, the images histogram has been equalized in order to get a constant brightness level of all images. The example below shows the conversion from grayscale to an equalized histogram:

![Grayscale with equalized histogram][image4]

The last step normalizes the images by dividing by 255 and subtracting 0.5 as LeCun suggests in his paper about [efficient backpropagation][paper03]. See below an example:

![Normalized image][image5]


#### 2. Training Data Augmentation

The fifth code cell of the IPython notebook contains the code for augmenting the data set.

I decided to generate additional data because some classes in the training data set are underrepresented. In order to help the network to generalize and correctly classify those classes I augmented the original data set by generating new images with small and random transformations (translate, rotate, shear). At different stages in the process of finding the right parameters, I tried various numbers and finally settled with a maximum of 2500 images for each class. This gives a good compromise between accuracy and training speed.

Here are some examples of original images and their augmented counterparts:

![Transformation examples][image6]
![Transformation examples][image7]

In summation, this leads to a new training set size of **107500** and the parameters for the transformations are as follows:

* Translation: +- 2 Pixel (x and y direction)
* Rotation: +- 10 Degree
* Shear: +- 5 Degree


The new image per class distribution looks like:

![Augmented data bar chart][image9]

#### 3. Model Architecture

The code for my final model is located in the sixth cell of the IPython notebook.

My final model consisted of the following layers and is highly inspired by the Sermanet & LeCun model:

![Transformation examples][image8]

And here is a table with more detail:

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

To train the model, I used the **Adam optimizer** and the following hyperparameters:

* Learning rate: **0.001**
* Batch size: **100**
* Epochs: **50**
* Weight initialization with truncated normal distribution with **mu = 0** and **sigma = 0.1**


#### 5. Solution Design

The code for calculating the accuracy of the model is located in the seventh cell of the IPython notebook.

I started off with a standard [LeNet5 architecture][paper01] and tried to tune the hyperparameters from there. First with only grayscaling the images and after that I tried various input normalization techniques which gave me a about 1% higher accuracy. The further equalization of the images histogram, in order to decouple the model from brightness effects, added another 1-2% accuracy on the validation set.  

After playing a lot with the hyperparameters and not getting results above 95%, I decided to increase the convolution layer sizes and to connect each convolutions output to a big fully connected layer. Finally i got results above 97%. To prevent the model from overfitting caused by the big last layer, I added a [dropout layer][paper04] with a keep probability of 50%.

My final model results were:
* Validation set accuracy of **98.7%**
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
    Normalization: (X_train / 255.0) - 0.5
    Preprocessing: grayscale, equalized Histogram
    Validation Accuracy: 0.957


20.02.2017

    Standard LeNet5 (C32 - C64 - FC256 - FC84 - SM43)
    Epochs: 30 | Batchsize: 200 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: (X_train / 255.0) - 0.5
    Preprocessing: grayscale, equalized Histogram, Additional data (rotated, translated)
    Validation Accuracy: 0.969


22.02.2017

    Standard LeNet5 (C32 - C64 - FC512 - FC84 - SM43)
    Epochs: 30 | Batchsize: 100 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: (X_train / 255.0) - 0.5
    Preprocessing: grayscale, equalized Histogram, Additional data (rotated, translated)
    Validation Accuracy: 0.973

22.02.2017

    Advanced LeNet5 (C16 - C32 - FC512 - SM43) [connecting both convolution outputs to the FC512]
    Epochs: 30 | Batchsize: 100 | LearningRate: 0.001 | Initialization: truncated_normal
    Normalization: (X_train / 255.0) - 0.5
    Preprocessing: grayscale, equalized Histogram, Additional data (rotated, translated)
    Validation Accuracy: 0.987

---


### Test a Model on New Images

#### 1. Acquiring New Images

Here are five German traffic signs I found online while browsing through Google Street View in Berlin:

[example_image1]: ./examples/example1.png "Example 1"
[example_image2]: ./examples/example2.png "Example 2"
[example_image3]: ./examples/example3.png "Example 3"
[example_image4]: ./examples/example4.png "Example 4"
[example_image5]: ./examples/example5.png "Example 5"

![Example 1][example_image1]
![Example 1][example_image2]
![Example 1][example_image3]
![Example 1][example_image4]
![Example 1][example_image5]

* The first image is a sign painted on the street. Because of that it has quite a perspective distortion and might be difficult to classify.

* The second image is a different version of the German 30 km/h speed limit sign. I wanted to see if the network is able co classify it correctly, but I guess due to the additional characters on the sign this might be hard.

* The third image contains a yield sign partly overlapped by a tree. The overlap could influence the classification probability and the model might not be as certain.

* The fourth image is pretty standard and the model should classify it with high certainty.

* The fifth image is a 120 km/h speed limit display. The colors are quite different compared to the original speed limit sign and I guess the model will have problems classifying it correctly. Mainly because of the black background and the yellow characters.


#### 2. Performance on New Images & Model Certainty

The code for making predictions on my final model is located in the tenth cell of the IPython notebook.

##### 2.1 Analysis of the Performance on New Images

[prediction_image1]: ./examples/prediction1.png "Prediction 1"
[prediction_image2]: ./examples/prediction2.png "Prediction 2"
[prediction_image3]: ./examples/prediction3.png "Prediction 3"
[prediction_image4]: ./examples/prediction4.png "Prediction 4"
[prediction_image5]: ./examples/prediction5.png "Prediction 5"


![Prediction 1][prediction_image1]

The first image is actually a *Children crossing* sign but my trained model classified it as a *Wild animals crossing* sign. This is obviously a wrong prediction and in my opinion caused by the perspective distortion. Additionally it is surprising the model has a certainty of a 100% on the wrong sign. I guess this might be a indicator for an overfitted network.


![Prediction 2][prediction_image2]

As this second sign belongs to a class on which the model was not explicitly trained the false prediction is not a big surprise. But it is very similar to a standard *30 km/h limit* sign and I had some hope it could classify it correct. As the bar chart indicates it got the general speed limit type right, but classified it as an *50 km/h limit* sign.

![Prediction 3][prediction_image3]

With this third example prediction, the model got it right. Although the the traffic sign is partly overlapped by a tree it got a certainty of a 100% on the correct class.

![Prediction 4][prediction_image4]

The fourth image was also classified correctly. This wasn't a particularly hard image and as there is no perspective distortion and the sign is fully visible we got a certainty of a 96.96% on the correct class.

![Prediction 5][prediction_image5]

This last example image is consists of a *120 km/h speed limit* display on a German Autobahn. The model thinks it is a *Stop* sign with a certainty of a 100%. The second prediction is the correct class but the model does not even consider it with 1%. This might be caused by the yellow characters, as they appear very bright (white) in the preprocessed image. Additionally the inner background of the sign is dark and not white as usual.


##### 2.2 Summary of the Performance on New Images

The model has its Problems given very distorted images and might be overfitted in some way because it is predicting 100% certainty on wrong images. A good generalized model should distribute the probabilities more even. However given *good* example images with small distortions or small overlaps it performs very well.


##### 2.3 Accuracy on all example images

The model was able to correctly guess 15 of the 18 traffic signs, which gives an accuracy of **83.33%**. This compares to the accuracy on the test set of 96.0%. The difference is caused by using some really hard problems for the classifier like the 120 km/h speed limit display or the highly distorted children crossing sign. But still okay as the model was not explicitly trained on them.

---

### Appendix

#### Papers
[paper01]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[paper02]: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
[paper03]: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
[paper04]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

- [Gradient-Based Learning Applied to Document Recognition [LeCun et al. 1998]][paper01]
- [Traffic Sign Recognition with Multi-Scale Convolutional Networks [Sermanet & LeCun, 2011]][paper02]
- [Efficient BackProp [LeCun et al., 1998]][paper03]
- [Dropout:  A Simple Way to Prevent Neural Networks from
Overfitting [Srivastava et al., 2014]][paper04]

#### Blogs & Tutorials
[blog01]: https://campushippo.com/lessons/build-a-tensorflow-traffic-sign-classifier-at-95-accuracy-214727f24
[blog02]: https://medium.com/@andrew.d.wilkie/self-driving-car-engineer-diary-4-c75150cf93d5#.lrmjqqbb5
[blog03]: https://chatbotslife.com/intricacies-of-traffic-sign-classification-with-tensorflow-8f994b1c8ba#.40pwknr15
[blog04]: https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad#.7sk6yag6z
[blog05]: http://navoshta.com/traffic-signs-classification/

- [Learn Build a TensorFlow Traffic Sign Classifier [machinehorizon]][blog01]
- [Self-Driving Car Engineer Diary — 4 [Andrew Wilkie]][blog02]
- [Intricacies of Traffic Sign Classification with TensorFlow [Param Aggarwal]][blog03]
- [(98.8% solution) German sign classification using deep learning neural networks [Vivek Yadav]][blog04]
- [Learn Build a TensorFlow Traffic Sign Classifier [Alex Staravoitau]][blog05]

#### Tools:
[tool01]: https://www.tensorflow.org/
[tool02]: http://matplotlib.org/
[tool03]: http://opencv.org/

 - [TensorFlow][tool01]
 - [Matplotlib][tool02]
 - [OpenCV][tool03]
