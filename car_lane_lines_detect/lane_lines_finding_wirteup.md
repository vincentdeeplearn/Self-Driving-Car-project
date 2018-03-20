
## Writeup

---

** Lane Finding Project**

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_undist1.jpg "Undistorted"
[image2]: ./output_images/undistort_realimage.png "Undistor test image"
[image3]: ./output_images/thresholding.png "Binary Example"
[image4]: ./output_images/warped2.png "Warp Example"
[image5]: ./output_images/polyfitlines1.png "Fit Visual"
[image6]: ./output_images/backtoorigin1.png "Output"
[video1]: ./project_video1.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/p4_final.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
First,i load the dist and mtx,the camera calibration coefficients;then,i use cv2.undistort apply to the origin images.(the in[3] in p4_final.ipynb)
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried to combine all of these thresholding,but it not work good.
I found that gradient thresholding using sobel operator in x-direction was the most effective. For color thresholding I used the s-channel to identify yellow and white lines better under different lightening conditions.
So,I combined the sobel X thresholding and S-color channels thresholding from HLS .(thresholding steps at in[4] in p4_final.ipynb;combined function step at in[7] in p4_final.ipynb).  

For this video the Yellow-thresholds and White-thresholds is the best way,its more clearer,less noise.but when i use this combined thresholding,its always raise error for my pipelines.For shorten time,so i still use my former combined thresholding.

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the line 60 in In[4]  code cell of the P4_final.ipynb).  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. 
I identified 4 src points that form a trapezoid on the image and 4 dst points such that lane lines are parallel to each other after the transformation. The dst points were chosen by trial and error but once chosen works well for all images and the video since the camera is mounted in a fixed position.
Then,use cv2.getPerspectiveTransform to gain the M matices,use cv2.warpPerspective() to gain the perspective images.

I use the mentor's src and  dst points,its more clearer and less noise.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 545, 460      | 0, 0        | 
| 735,460      | 1280, 0      |
| 1280,700     | 1280, 720      |
| 0, 700      | 0,720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

Function:  polyfit_warp_image() this is for the first video image to detect. (in In[11] code cell of the P4_final.ipynb)
Since the video is continuous,if wo know the last left_fit and right_fit,then no need to for over the image again to find the lane lines.
So i set function:polyfit_warp_continuous() to fit the lanelines if wo know former 2nd order polynomial rate values.(in In[13] code cell of the P4_final.ipynb)

1、Take a histogram of the bottom half of the warped_image;

2、use np.argmax() find the peak of the left and right halves of the histogram;

3、set a sliding window (80 pixel width 80 pixel high)

4、Identify the x and y positions of all nonzero pixels in the image;

5、Step through the windows search one by one,Identify the nonzero pixels in x and y within the window,if the pixels in the windows >minpix(70),then calculate the mean position and record it.

6、Extract left and right line pixel positions

7、Use left and right line pixel positions,Fit a second order polynomial to left lanes and right lanes;

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this  in In[14] code cell of the P4_final.ipynb

1、Choose the maximum y-value(y_eval), corresponding to the bottom of the image;

2、Use curve rate to calculate the left and right X values;
the formula is :x = ay2 + by + c

3、Fit new polynomials to x,y in world space(pixel was changed to meter in each x/y dimension)

4、Calculate the new radii of curvature
the radius of curvature formula is:

radius = (1 + (2a y_eval+b)2)1.5 / abs(2a)

5、We assume the camera is mounted exactly in the center of the car. We first calculate the bottom of left and right lane and hence the center of the lane. The difference between the center of the image (680 positions) and the center of the lanes is the offset (in pixels). The calculation was then converted to meters:3.7/850 meters per pixel in x dimension.
Because the gaps between my perspective lane lines pixel is approximatly 850 pixels,the real world the distance is 3.7m.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step  in In[17] code cell of the P4_final.ipynb.

Once we have lane lines identified and polynomial fit to them, we can again use cv2.warpPerspective and the Minv matrix to warp lane lines back onto original image. We also do a weightedadd to show the lane lines on the undistorted image. 

I have used cv2.putText to display radius of curvature and offset from center on this image.(The code for adding text to image is in In[16] code cell of the P4_final.ipynb)

Here is an example of my result on a test image:

![alt text][image6]


#### 7.Sanity Check

I implemented a function to sanity check if the lane lines were being properly identified. This function did the following 3 checks:
* Left and right lane lines were identified (By checking if np.polyfit returned values for left and right lanes)
* If left and right lanes were identified, there average seperation is in the range 150-430 pixels
* If left and right lanes were identified then there are parallel to each other (difference in slope <0.1)

If an image fails any of the above checks, then identified lane lines are discarded and last good left and right lane values are used.
The code for sanity check is in In[20] code cell of the P4_final.ipynb
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I added the following checks to this pipeline:
* If this is the first image then the lane lines are identified using a histogram search (checked using a counter variable)
* Otherwise the previous left and right fit are used to narrow the search window and identify lane lines (function implemented in line 12 in In[21] code cell of the P4_final.ipynb)
* The left and right fits identified were stored in a variable and if sanity check described above was failed, then the last good values of left and right fit were used.(function implemented in line 19 in In[21] code cell of the P4_final.ipynb)

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems i faced in this project:
1、how to choose the src points and dst points.I spend two weeks to handle with it;First,i design a function like fit the lane lines function to find the 4-src points,but because the Discontinuous of the right lane lines,So i always lose to find the right points,so if the function failed i have to set fixed right src points.
And so if the camera was not in the middle of the image,or tingle,then the pipeline will fail. 
2、the shadow and car in the images is big problem for my pipeline to find the sobel thresholding.So if encounter more complex environmets,the pipeline will fail to dectect the lane lines.
3、I apply the pipelines to the challenge_video,works bad.

The way to make it more robust:
1、i just use the former imformations of lane lines when the pineline failed to find right lines.The better way is use the average of the former imformations,i can use the Prior Knowledge.Take an average over n past measurements will make the pineline more robust.
2、first,complete the discontinuous curve lane lines,it's pity for Houghtransform it hard to detect curve;then,define a function to detect the curve lane lines.find the 4-src points,and set the 4-dst points,then warp to perspective images. Such way,maybe can keep the pipelines more robust.



```python

```
