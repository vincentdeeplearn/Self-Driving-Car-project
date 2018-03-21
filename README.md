# Self-Driving-Car-project
The project i made about self driving car.It's mainly focus in computer Vision and Deep Learning.
# 1. Lane-lines-detect
First,i use the gradient of image and different color space to find the lane lines,and get the perpective image to valid the curve of the lane lies,then warp back to the images. 
# 2.Clone-Learning
i use the data gain from the simulator matually. Then,use different argument data technology apply to the raw data.then,i use the pipelines from Nvidia to train the model to learn the drive by the feature of images.then,i got it.
# 3.Car-detection
I used the histgram and HOG features of images to be the features of the veichles.i use the svm model to train the model.then apply a sliding windows to the real video to detect the viechles. 
