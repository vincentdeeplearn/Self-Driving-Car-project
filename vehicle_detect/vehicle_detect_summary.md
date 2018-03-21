
## Vehicle detect summary

---

The steps of this project are the following:

* normalize the features and randomize a selection for training and testing.
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use  trained classifier to search for vehicles in images.
* Run the pipeline on a video stream  and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ../examples/carsnotcars.png
[image2]: ../examples/hogvisualize.png
[image3]: ../examples/findcars.png
[image4]: ../examples/findcars1.png
[image5]: ../examples/last20frames.png
[image6]: ./examples/labels_map.png
[image7]: ../examples/thefinal.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1.  how  i extracted HOG features from the training images.

The code for this step is contained in the ** 5 ** code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored  different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from vehicle images and displayed them to get a feel for what the `skimage.hog()` output looks like.
I  refered from the Udacity forum. It suggested  orient= 11.That's gain more features from each single image.

Here is an example using the gray image and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]
here is image with HOG parameters of `pix_per_cell= 8 cell_per_block = 1 orient = 9`:
<img src="../examples/hog1.png" width="400">
here is image with HOG parameters of `pix_per_cell= 16 cell_per_block = 1 orient = 9`:
<img src="../examples/hog2.png" width="400">
here is image with HOG parameters of `pix_per_cell= 8 cell_per_block = 2 orient = 11`:
<img src="../examples/hog3.png" width="400">

#### 2.  how  settled on my choice of HOG parameters.

The code for this step is contained in the **7** code cell of the IPython notebook .  

I tried various combinations of parameters and run it in the model.the main improvement is orient changed from 9 to 11.Because the main feature in this project is gradient of direction,so more larger orient means gaining more feature information.

Final params i choose:
* color_space = 'YUV' - YUV resulted in far better performance than RGB, HSV and HLS
* orient = 11  # HOG orientations - I tried 9 . 11 is more better.
* pix_per_cell = 16 - I tried 8 and 16 and finally chose 16 since it signficantly decreased computation time
* cell_per_block = 2 - 
* hog_channel = 'ALL' -  ALL resulted in far better performance than any other individual channel

#### 3. how  trained a classifier using your selected HOG features (and color features).

The code for this step is contained in the ** 7 ** code cell of the IPython notebook . 
I first trained a linear SVM using different various colorspace.All of the accuracy is nearly 98%,but bad in genelizing the real image.Finally,I chose MLPclassifier,the test accuracy is  99.35% 

The code for this step is contained in the ** 5 ** code cell of the IPython notebook .
this time i not only use HOG features,i add color_histogram(nbins=32) and bin_spatial(size=(16,16)) features to model.it's make the model more robust.

### Sliding Window Search

#### 1.I  implemented a sliding window search.  decide what scales to search and how much to overlap windows?

The code for this step is contained in the ** 6 ** code cell of the IPython notebook .
I decided to search random window positions(y_start = 400,y_end = 660) at scale=1.2 or 1.3 in test images,but i found the best scale=1.5 for my model:
windows size=64.Overlap windows:cells_per_step = 2(Instead of overlap, equal overlap=0.75)

![alt text][image3]

#### 2. some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for this step is contained in the ** 9 ** code cell of the IPython notebook .
Ultimately I searched on 1.5 scales , use YUV 3-channel HOG features and color histogram features / bin spatial features for the  feature vector.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to my final video output.  
Here's a [link to my video result](../MLP-2018vehicle_detection.mp4)
The code for vehicle detection pipeline is in cell ** 12 ** of the IPython notebook.


#### 2. how i  implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I recorded the positions of positive detections in each frame of the video. I combined detection over 20 frames (or using the number of frames available if there have been fewer than 20 frames before the current frame). From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I found best performance with threshold parameter of 18.  I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

The code for this is in the vehicle detection pipeline is in cell ** 7 and 12 **

### Here are 20 frames and their corresponding heatmaps:Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 20 frames:

![alt text][image5]


![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion




### Two problems that I faced were:

1. I found that the test accuracy in the classifier was not a good predictor of actual performance in the video. Most model combinations had an accuracy of 98%+ but only a few had good performance in the video. This was a bit surprising. I think this is because I didn't put in extra work in making sure that examples in training and testing were distinct. As a result the model overfit to the training data. To identify the best model, I tested performance in the video. 
And i confused by the model between the SVM and MLP,its have almost 98%+ accuracy in validation dataset,but SVM classifier bad in real video,MLP classifier perform better,i still don't know why.

2. Once the video pipeline was working, it was detection false positives in some frames and not detecting the car in other frames. Careful tuning of num of frames over which windows are added and thresholding parameter were needed. Ideally there should be a way of modifying these parameters for different sections of the video.

My biggest concern with the approach here is that it relies heavily on tuning the parameters for window size, scale, hog parameters, threshold etc. and those can be camera/track specific. I am afraid that this approach will not be able to generalize to a wide range of situations. And hence I am not very convinced that it can be used in practice for autonomously driving a car. 

Here are a few  other situations where the pipeline might fail:

1. I am not sure this model would perform well when it is a heavy traffic situations and there are multiple vehicles. You need something with near perfect accuracy to avoid bumping into other cars or to ensure there are no crashes on a crossing. 

2. The model was slow to run. It took 6-7 minutes to process 1 minute of video. I am not sure this model would work in a real life situation with cars and pedestrians on thr road. 

To make the code more robust we can should try the following:

1. Reduce the effect of time series in training test split so that the model doesn't overfit to training data

2. Instead of searching for cars in each image independently, we can try and record their last position and search in a specific x,y range only



```python


```
