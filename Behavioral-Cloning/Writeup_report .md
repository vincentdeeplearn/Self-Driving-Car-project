
# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Grayscaling"
[image3]: ./examples/left_bias.jpg "Recovery Image"
[image4]: ./examples/right_bias.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/left.jpg "Left camera Image"
[image7]: ./examples/right.jpg "Right camera Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the Nvidia's pipeline.The model consists of a convolution neural network with  3x3 filter sizes and depths between 24 and 64 (model.py lines 108-112) 

The model includes RELU layers to introduce nonlinearity (code line 108--122), and the data is normalized in the model using a Keras lambda layer (code line 106). and i alse add RELU layers to the FC layers(code line 116-122)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 118 120).I also try use regularizers to FC layers for kenel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.001),but the model training loss is higher than use Dropout only,maybe use two regular method is too strict to punish the parameters? Just use only use one way is better. 

I also try add Dropout after Flatten layers,the val_loss is good,but when i test in the simulator,the model don't study how to turn......

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18). The model was tested by running  through the simulator and ensuring that the vehicle could stay on the track with speed 30.

#### 3. Model parameter tuning

I used  an adam optimizer, so the learning rate was tuned reducing to 0.0008,it is key points.I had try more than 20 times just use the default LR=0.001,but validation loss(MSE) is always higher then 0.02,and always can not turn  when testing in the simulater.Then i set the LR lower,the model study how to turn. (model.py line 125).
And this time,add EarlyDropping callbacks to the model,so it will stop when the val_loss increasing.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I just used the Udacity's data set.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the 'good' model that was certified,for example:Alexnet,Vggnet,etc.

My first step was to use a convolution neural network model similar with  Alexnet, I thought this model might be appropriate because Alexnet is a classic net and do not have too much layers,maybe it is good start points from using it.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model(Alexnet) had a high mean squared error both on the training set and the validation set. That implied that the model was underfitting. 
I think maybe because Alexnet was to solve classification questions,it use cross entropy method,but this project output was a continous values.
To combat the underfitting, I modified the model, i used the Nvidia's net so that had a low MSE on the training set,but got high MSE on validation set,and bad performance in simulator test.It's overfitting.

Then I add Dropout layers in the full connect layers.then the validation loss reduce obviously.

The final step was to run the simulator to see how well the car was driving around track one. The mainly problems were turning  where the vehicle fell off the track.Obivously the model does not study how to turn around.At first,i was think maybe i should collect more turning data,but when i add more turning data(collect by myself) to my model it's make the validation loss worse. To improve the driving behavior in these cases, I reduced the default Learning rate 0.001 to 0.0008,then i got it.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I also try the model pipeline:from comma.ai,i just copy the model to train,but it have higher loss,so i quit tuning it.

#### 2. Final Model Architecture

The final model architecture (model.py lines 72-87) consisted of a convolution neural network with the following layers and layer sizes:
Conv1  strides=(3,3)  filters=24

Conv2  strides=(3,3)  filters=36

Conv3  strides=(3,3)  filters=48

Conv4  strides=(3,3)  filters=64

Conv5  strides=(3,3)  filters=64

FC1   100-dims

Dropout1  droprate=0.2

FC2   50-dims

Dropout2  droprate=0.2

FC3   10-dims

FC4   1-dims

I also use the plot_model want to plot the model graph.I have install pydot and graphviz successfully,but i don't know why it still import error:

ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to from the sides back to the center. These images show what a recovery looks like starting from below :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also used the left/right camera images and correction angles  that this would augment the data set and maybe teach the model from the edges of the tracks back to center. For example, here is an image that catched by left/right images:

![alt text][image6]
![alt text][image7]


After the collection process, I had 19284 number of data points. I then preprocessed this data by:

1:I reference Vivek Yadav's blog.First,bright the image randomly;then,Horizontal and vertical shift the image.Third,Crop the image(remove 50 pixels top,25 pixels bottom) ,then resize image to 64*64.Finally,flip the iamge with probility 0.5.

2:produce a generator.read the image one by one,use cv2.cvtcolor change the color space from BGR to RGB and use correction angles for reading the left/right images.

3:normalize the image to scale to (-1.0,1.0)


4:All of the preprocess step was done by the generator,then can use GPU to preprocess data but not CPU.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the training loss is no longer reduce. I used an adam optimizer so that manually training the learning rate wasn't necessary.


```python

```
