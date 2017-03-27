#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/data_hist.png "Visualization"
[image2]: ./examples/fake_data_hist.png "Fake Visualization"
[image3a]: ./examples/before.png "Before"
[image3b]: ./examples/after.png "After"
[image3c]: ./examples/rotated.png "Rotation"
[image4]: ./internet_images/keep_right_38.jpg "Keep Right"
[image5]: ./internet_images/pedestrian_27.jpg "Pedestrian"
[image6]: ./internet_images/priority_12.jpg "Priority"
[image7]: ./internet_images/row_11.jpg "Right of Way"
[image8]: ./internet_images/speed_50_2.jpg "Speed 50"
[image9]: ./internet_images/speed_60_3.jpg "Speed 60"
[image10]: ./internet_images/stop_sign_14.jpg "Stop Sign"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/grantsrb/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

#### A Basic Summary of the Data

The code for this step is contained in the fourth code cell of the IPython notebook.  

I used numpy and matplotlib to calculate and produce summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image dimensions = (32, 32, 3)
* Number of classes = 43

The code for the following visualization is contained in the fifth and seventh code cell of the IPython notebook.

Here is a visualization of the data set. The first bar chart shows an unequal distribution of data. This can potentially make the classifier favor certain classifications over others. The second bar chart displays the distribution of data after rotated data was added. Images from classifications with fewer than 1000 samples were rotated -9, -4, 4, and 9 degrees and added to the dataset.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### Preprocessing Image Data

The preprocessing is defined in code cell 6 and applied to the data sets in code cell 7.

I converted the images to grayscale because this seemed to give better results than the color counterparts. It was also recommended by LeCun in the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks." I never tried an inception net with the color images due to time restrictions. All nets I did try, however, performed better when the samples were grayscale.

After converting to grayscale, I normalized the images by subtracting the average pixel value in the training set and dividing by the pixel standard deviation. This reduced the pixel values and centered them around the origin. In theory, this helps the optimizer find the minimum cost during optimization. The gradient descent doesn't need to work as hard to find the theoretical minimum because values do not need to be changed as much to significantly affect the prediction values. I tried a few trainings with non-normalized images and they did not perform as well as when the data was normalized.

As described in the last section, to alleviate the unequal data distribution, I added fake data to the training set by adding rotated images from underrepresented classes. They were rotated by -9, -4, 4, and 9 degrees if their class had fewer than 1000 samples. Four blank traingles are created when rotating images. I tried both leaving the rotation space blank and making the rotation space a random value. The random seemed to give better results.

Here is an example of a traffic sign image before and after preprocessing followed by a rotation of -9 degrees.

![alt text][image3a]
![alt text][image3b]
![alt text][image3c]


#### Model Architecture

The code for my final neural net model is located in the 10th, 11th, and 12th cells of the ipython notebook.

My final model was a derivative of the LeNet architecture. It consisted of the following layers all simply feeding forward to the next layer:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x84 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x84 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Dropout	      	| 0.5 probability 				|
| Fully connected		| 800x120, outputs x120        									|
| RELU					|												|
| Fully connected		| 120x84, outputs x84        									|
| RELU					|												|
| Fully connected		| 84x43, outputs x43        									|




#### Neural Net Training

The code for training the model is located in the 14th cell of the ipython notebook.

I left the batch size as 128 because changing it didn't seem to significantly affect results.

To train the model I used the momentum optimizer under the theory that it was less likely to get caught in a local minimum. I also tried the Gradient Descent Optimizer and the Adam Optimizer. Empirically, the momentum optimizer seemed to produce the best results, but I did not have time to fully explore the differences.

Instead of manually setting the epoch count, I created a function to monitor the cost of the training set during training. When the cost was no longer decreasing, the function would decay the the momentum. After 5 decays or 40 epochs, the training would cease. The learning rate was initially set to 0.05 but was decreased to 0.01 at the 8th epoch and further decreased to 0.001 at the 14th epoch.

The dropout probability was an interesting parameter because its optimal value was dependent on the architecture of the neural net. The best validation accuracy of 97% and test accuracy of 96% were found when dropout was performed once with a probability of 0.5 between the convolutions and fully connected layers.

#### 4. Creative Approach

The code for calculating the accuracy of the model is located in the 15th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.86%
* validation set accuracy of 97.37%
* test set accuracy of 95.78%

The first model architecture I tried was the LeNet architecture as recommended by the Udacity Self Driving Car class. With preprocessing and multiple tests with different hyperparameters, I could only get about a 93% accuracy on the validation set. 

I read LeCun's paper on traffic sign recognition. They described improved results by feeding the first and second activations of the net into the final fully connected layer. I tried this and found the results to be about the same. 

I tried adding and removing convolution layers and fully connected layers each of which produced validation accuracies ranging from 85-93.

I then adjusted parameter counts within the layers. I increased the initial convolution depth to 64 from 6 in the simple feed forward LeNet architecture. This drastically improved results to about a 97% accuracy on the validation set. I then tried increasing the second convolution depth but this gave worse results.

At this point, it seems worth noting that given perfect optimization, increasing convolution depths should only improve results. This is because an activation can always be set to 0 and have no effect on the subsequent layers. This was difficult to realize in practice. I believe this most likely was due to the difficulty of optimization. The more parameters there are, the longer training takes and the more local minimums there are to get caught on.

I continued to experiment with different depths and found that a first layer depth of 84 and a second of 32 gave slightly better results than 64 followed by 16. I tried improving further by feeding the first and second activations into the last layer. This, however, decreased the validation accuracy, so I kept the simple forward pass architecture.

I then created a decay function for the momentum and set epoch intervals for changing the learning rate. It was obvious that different learning rates and momenta would perform better at different points in the training process.

After checking the certainties of my model's predictions, I noticed that is was very sure of each of its predictions even when wrong. A less aggressive learning rate with a longer training session would theoretically help this issue, but I did not have time to explore this idea further.

At this point I was aware that experimentation on most of my ideas for net architecture would take longer to test than I had time for. I started focussing on preprocessing to improve results further.


### Model on New Images

#### Five German Traffic Signs (found on the web)

Here are five German traffic signs:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

For many of the images, the brand that covers a portion of the image could make it more difficult for the classifier to perform. During preprocessing, the presence is less obvious, however it could still have a significant impact.

The first image (Keep Right sign) could be difficult to classify due to the sharp edge on what is supposed to be a circular sign. The model, however, gave a correct prediction, so this must not have been a significant feature for the model.

The Pedestrian sign is extremely similar to the German General Warning sign which can prove difficult for the classifier. Both signs are triangular with a long, thin, black detail in the triangle.

The speed signs would potentially be difficult to classify because the numbers are the only distinguishing characteristics between each of them. The outer circle is shared by a number of signs.


#### Model Predictions (Web Images vs Test Set)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep Right     			| Keep Right 										|
| Pedestrian					| General Caution											|
| Priority					| Priority											|
| Right of Way					| Right of Way											|
| 50 km/h	      		| 50 km/h					 				|
| 60 km/h	      		| Round About Mandatory					 				|
| Stop Sign      		| Stop sign   									|

Total Accuracy: 71.43

The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71%. This compares poorly to the accuracy on the test set of 96%. This may be in part due to the company logos accross the front of some of the images and also potentially from the formatting of the jpg files to fit the 32x32x3 shape.

Noteably, 50 km/h sign was predicted correctly whereas the 60 km/h sign was not. The 60 km/h sign was predicted as a Roundabout Mandatory sign which has a circular shape. This indicates two things. The first is that the 60 km/h sign has a low reliability of being properly classified. The second indication is that when a Round About Mandatory sign is predicted, there is a low reliability that it is a Round About Mandatory sign. Put more concisely, the 60 km/h classification has a low accuracy whereas the Round About Mandatory classification has a low precision.

For the Pedestrian sign, the prediction of General Caution is incorrect. In the classifier's defense, the two signs look paticularly similar when heavily pixelated. Regardless, the prediction indicates a low accuracy for pedestrian sign and low precision for the General Caution sign.


#### Model Certainty on Web Images

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.


##### Keep right
![alt text][image4]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep right | 0.999881 |
| Yield | 0.000119031 |
| Right of Way | 1.38229e-07 |
| Road work | 4.5846e-08 |
| Priority road | 5.63471e-09 |

The Keep Right sign was accurately predicted with a high degree of certainty (99.9%).

##### Pedestrians
![alt text][image5]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General caution |  0.997157 |
| Pedestrians |  0.00157825 |
| Wild animals crossing |  0.00033264 |
| Road narrows on the right |  0.000276632 |
| Go straight or left |  0.000252305 |

The Pedestrians sign was incorrectly classified with a relatively high degree of confidence (99.7%). The Pedestrian sign was the next highest probability, but only with a confidence of 0.16%. 


##### Priority
![alt text][image6]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road | 0.999943 |
| No passing| 3.44303e-05 |
| Yield| 1.11475e-05 |
| No entry| 5.83228e-06 |
| Roundabout mandatory| 2.70981e-06 |

The Priority Road sign was correctly predicted with a relatively high degree of confidence (99.99%).


##### Right of Way at the Next Intersection
![alt text][image7]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right of Way | 0.999077 |
| Beware of ice/snow | 0.000458759 |
| Pedestrians | 0.000433949 |
| Children crossing | 2.96425e-05 |
| Double curve | 2.69535e-07 |


The Right of Way sign was correctly predicted with a relatively high degree of confidence (99.9%).


##### 50 km/h
![alt text][image8]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (50km/h) | 0.666202 |
| Speed limit (30km/h) | 0.315747 |
| Speed limit (70km/h) | 0.00427005 |
| Speed limit (100km/h) | 0.00349992 |
| Speed limit (80km/h) | 0.00300471 |


The 50 km/h sign was accurately predicted with 66.6% confidence. Each of the other top 4 predictions are been speed signs which indicates that the classifier is noticing the similarities between the speed signs. Given that 30 km/h has a prediction confidence of 31%, the classifier likely thinks 30 and 50 look similar.


##### 60 km/h
![alt text][image9]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roundabout mandatory | 0.626408 |
| Priority road | 0.169607 |
| End of no passing | 0.139186 |
| Keep right | 0.0221789 |
| No entry | 0.00824869 |

The 60 km/h sign was incorrectly classified as a Roundabout Mandatory sign with a confidence of 62.6%. 60 km/h does not make the top 5 predictions. The incorrect predictions are mostly circular signs except the Priority Road sign which is square. This indicates a low accuracy for 60 km/h and a low degree of precision for the Roundabout Mandatory sign.


##### Stop Sign 
![alt text][image10]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop | 0.425476 |
| Yield | 0.121781 |
| Turn right ahead | 0.0875134 |
| Keep left | 0.0601183 |
| Keep right | 0.0329059 |

The Stop Sign was correctly classified with a confidence of 42.5%. The next nearest prediction, is a yield sign which is triangular. The classifier must have noticed a similarity of the straight edges of the octagonal shape of the stop sign.


