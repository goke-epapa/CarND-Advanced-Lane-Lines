## Writeup for Advanced Lane Finding Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Calibration Image"
[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/distortion_corrected.jpg "Distortion Corrected"
[image4]: ./output_images/binary_combo.jpg "Binary Example"
[image5]: ./output_images/warped_image_with_points.jpg "Warped image with points"
[image6]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image7]: ./output_images/output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the  `camera_cal/camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world.

Then for each calibration image: I convert the image to grayscale using `cv2.cvtColor()` and find the chessboard corners using `cv2.drawChessboardCorners()`. If the corners are found, I append the object points to the list of valid object points and I also append the found corners to the image points list (2D).

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

**Raw Image**
![alt text][image0]

**Undistored Image**
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

I loaded the pickle file with the camera matrix and distortion coefficient using the `load_camera_calibration()`, then `cv2.undistort()` is used to undistort by passing the image, camera matrix and distortion coefficients as arguments.

Here is an example of a distortion-corrected image

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of x derivative, magnitude and saturation thresholds to generate a binary image (thresholding steps in `get_binary_pixels_of_interest` method in `lane_finder.py`).  Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 104 through 108 in the file `lane_finder.py`.  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. The src points are hardcoded and for destination points, the points are calculated relative to an `offset`. The points `src` and `dst` points are shown below:

```python
src = np.float32([
	[730,450],
	[1180,img_size[1]],
	[190,img_size[1]],
	[590,450]])

offset = 190 # offset for d	st points
dst = np.float32([
	[1180, 0],
	[1180, img_size[1]],
    [offset, img_size[1]],
    [offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 730,  450     | 1180, 0       | 
| 1180, 720     | 1180, 720     |
| 190,  720     | 190, 720      |
| 590,  450     | 190, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image  and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this step in the function `measure_curvature_real()` in my code in `lane_finder.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `visualise()` in my code in `lane_finder.py`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used the sliding window technique to detect the lane line the first time, then subsequently, I just searched around the previously detected line.

My pipeline will fail in very sunny weather conditions where the lane lines become very faint because the lane lines are not easily detected by the current approach. My recommendation would be to use a more advanced technique to detect the lane line.

I found it hard to process the challenge video, majorly because the line in the middle of lane is detected as a lane line.


