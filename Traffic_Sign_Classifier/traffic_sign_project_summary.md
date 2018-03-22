

#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/gray.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"




###Data Set Summary & Exploration

####1.  a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 31367
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of classes in the training, validation and test set.First,each class is not uniform distribution in training or test data set;Second,i use sklearn.model_selection.StratifiedShuffleSplit split the origin train data set,so the distribution of classes is the same between the training and valid data set.Third,i found the distribution of classes is very different between the training and test data set,that indicate the model trained by training data set maybe bad in generalizing in test data set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the traffic sign most important feature is the shape,not the color.In contrast,the color maybe the noise when train the model,because its real traffic images which the images quality was effected by photoing/light etc.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the origin images scale is (0,255),is too large for neural network's inputs.The big size of the scale will cause the loss function to large.and high variance will cause a lot of searching to find a good sulotion in mathematically.So i use (x-128)/128 function to scale origin images to (-1,1),it doesn't change the content of images and it make much easier for the optimization to proceed numerically. 
Then,reshape the image [32,32] to [32,32,1] for adapting for the placeholder shape [None,32,32,1].

I don't generate fake images.
Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe my final model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten				| outputs 400									|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Fully connected		| logits=outputs 43        						|
 


####3. Describe how  trained  model. 

To train the model, I used an AdamOptimizer,batch size=128,epochs=30,learning rate=0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.981
* test set accuracy of 0.917

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The Yan LeCun's LeNet architecture was my first architecture.Because the problem is the same between MNIST data and German traffic signs classification.They have the almost same size inputs,only the traffic signs are colorful.But the most important features are the same is identified by shape.Since LeNet are good at MNIST data set,then must be good at German traffic sigs classification.
* What were some problems with the initial architecture?
No problem.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The training accuracy is higher than the validation accuracy,maybe i should add dropout layer.
* Which parameters were tuned? How were they adjusted and why?
I decrease epochs from 50 to 30,because the validation accuracy of epochs=50 is lower than the validation accuracy of epochs=30. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The most important design choice is used the 1*1 kernel size,the subsampling is very useful.
Different conv layer extract different features,the previous layer extract the lines or edges,the last layer extract the shape of the signs.So maybe this is why convolution neural network works well in images' problems. 
The dropout layer can defend overfitting,to increase the generalization of the model.
If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
Because the most important features are the same:the shape between MNIST and German traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
First,the accuracy on the training,validation are very near,and near to the benchmark;
Second,the accuracy on the test are not too far away from accuracy on validation,it prove the model have good generalization.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because the bottom of the images have other shape,its big noise.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No parking     		| Priority road							        |
| Stop Sign				| Priority road									|
| 60 km/h	      		| 60 km/h					 				    |
| No entry			    | No entry     							        |
| yield 			    | Priority road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares favorably to the accuracy on the test set of is too low,maybe because the real images i chose have too large noise.For examples,the No.1 and No.2 signs have the obivous background.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority road sign (probability of 0.6), but the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Priority road   								| 
| .40     				| Roundabout mandatory 							|
| .00001				| Speed limit (50km/h)							|
| .0000	      			| General caution					 			|
| .0000				    | No entry      							    |


For the second image, the model is relatively sure that this is a Priority road sign (probability of 0.999),but the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Priority road   								| 
| .00092     			| Speed limit (60km/h) 							|
| .00006				| Right-of-way at the next intersection			|
| .00000	      		| No passing					 			    |
| .00000				| Beware of ice/snow      						|
For the third image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of 1), the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (60km/h)   						| 
| .00000     			| Speed limit (50km/h)							|
| .00000				| Speed limit (80km/h)		                    |
| .00000	      		| End of speed limit (80km/h)					|
| .00000				| Slippery road     						    |
For the 4th image, the model is relatively sure that this is a No entry sign (probability of 1), the image does contain a No entry  sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry   								    | 
| .00000     			| Speed limit (20km/h)							|
| .00000				| Stop		                                    |
| .00000	      		| Turn right ahead					 			|
| .00000				| Speed limit (70km/h)    						|
For the last image, the model is relatively sure that this is a Priority road sign (probability of .97734),the second prediction is Yield(only probability of .02241) the image does  contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97734        		| Priority road   								| 
| .02241     			| Yield							                |
| .00014				| Keep right		                            |
| .00006	      		| Speed limit (50km/h)					 	    |
| .00000				| Speed limit (30km/h)    						|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




