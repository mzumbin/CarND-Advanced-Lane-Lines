

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output/calib_output.png "Road Transformed"
[image3]: ./output_images/pipeline.png "Binary Example"
[image4]: ./output_images/parallel.png "Warp Example"
[image5]: ./output_images/convolution_find.png " Convolution Search"
[image6]: ./output_images/poly_and_points.png "fitting a Poly"
[video1]: ./project_video.mp4 "Video"


---




###1. Camera Calibration



The code for this step is contained  in `proc.py` in function   
`
def calibrate():
`
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


####2.Color and gradient pipeline
The pipeline to get the binary image uses a combination of histogram equalization , filtering and a dilation to get more points of the binary image

```python 
    #color and gradient threshold
    edges = p.gradient_pipeline(color_rgb)
    # filter some noise
    color_rgb = cv2.medianBlur(color_rgb, 5)
    kernel = np.ones((5, 5), np.uint8)
    # dilate to get more points to be fed in the polynomail fiting
    edges = cv2.dilate(edges, kernel, iterations=1)
```
I used a combination of color and gradient thresholds to generate a binary image. Illustrated in the  the function bellow
 `gradient_pipeline` function  file `proc.py`.  

```python
def gradient_pipeline(color_img_rgb):

    # Sobel x
    sobel_x =abs_sobel_thresh(color_img_rgb, orient='x', thresh_min=20, thresh_max=100)
    s_grad = hls_select(color_img_rgb, (80, 255))
    v_grad = hsv_select(color_img_rgb, (50, 255))
    # Combine the the  binary thresholds
    combined_binary = np.zeros_like(sobel_x)
    combined_binary[ ((s_grad == 1) & (v_grad == 1)) | (sobel_x ==1)] = 1
    return combined_binary
```


Here's an example of my output for this step.

![alt text][image3]

####3. Perspective Transform

The code to find a perspective transform need a picture with two parallel land lines to find the transform. The method works fitting two lines and find a point where they intercept, then a trapezoid is build as code above expose:  

```python
def find_line(col_left, row_left, col_right, row_right,img_shape):

    x0 = np.zeros(2)
    # fit left line first degree polynomial
    res = least_squares(line, x0, loss='cauchy', f_scale=5, args=(col_left, row_left))
    line_left = res.x
    # fit right line first degree polynomial
    line_right = least_squares(line, x0, loss='cauchy', f_scale=5, args=(col_right, row_right)).x

    x,y =iterception(line_left, line_right)
    #test if is vanish point
    if abs(x - img_shape[1]/2)< 5:
        good_value = True
    else:
        good_value = False
     # building the trapezoid   
    top_y = y + 50# top y coordinate,  50 pixels above where lines intercept
    #where line intercept at top_y
    top_left_x = (top_y - line_left[0])/line_left[1]

    top_right_x = (top_y - line_right[0])/line_right[1]

    bottom_y = img_shape[0] - 20
   #where line intercept at bottom_y
    bottom_left_x = (bottom_y - line_left[0])/line_left[1]
    bottom_right_x = (bottom_y - line_right[0])/line_right[1]
    
    src = np.float32([[bottom_left_x, bottom_y], [top_left_x, top_y], [top_right_x, top_y], [bottom_right_x,bottom_y]])

    quart_x =img_shape[1]/4
    max_y = img_shape[0]
    dst = np.float32([[quart_x, max_y], [quart_x, 0], [3*quart_x, 0], [3*quart_x, max_y]])

    H = cv2.getPerspectiveTransform(src, dst)
    inv_h = cv2.getPerspectiveTransform(dst, src)
    return  H, inv_h , good_value
```

I verified that my perspective transform was working as expected by warping an image with parallel test image and its warped counterpart to verify that the lines appear parallel in the warped image above.

![alt text][image4]

####4. Finding the Lanes and Fitting a polynomial

First the land finding algorithm uses a variation of convolution lane proposed in the lessons. The function return the points contained in the bins, another modification is that, the bin stays foot if number of points do not reach threshold. This points are fitted using a robust least squares using the Cauchy loss function.  The code for search the centroid bins is the function in proc.py file.
```
def find_window_centroids(warped, window_width, window_height, margin):
```

**output from convolution search**
![alt text][image5]


After the lanes are initialized and if lanes founded are  valid ( this is checked measuring  distances from each order) the search for the lane points uses only a small area around the last fitted polynomial instead of full warped area. This points are append to a list containing the last n  valid frames,  The points in this list are used to fit the lanes again using a least square with Cauchy loss function. Fitting a polynomial against n last frames and using a small constant in the error function (5 pixel) result in a smooth variation in the polynomial coefficients. 

Code for finding with previous polynomial is in file `connected.py`

```python
 leftx, lefty, rightx, righty = p.find_lanes_points(current_left_fit, current_right_fit, warped)
    dif = np.mean(rightx) - np.mean(leftx)
    #test if lines makes sense
    if len(leftx) > 50 and len(rightx) > 50:
        if abs(dif - 650) < 80 :
            # add last found points to list wich has last points from last n good frames
            lx_points_list = p.window_list(lx_points_list, leftx)
            ly_points_list = p.window_list(ly_points_list, lefty)
            rx_points_list = p.window_list(rx_points_list, rightx)
            ry_points_list = p.window_list(ry_points_list, righty)
            ploty, left_fitx, right_fitx, new_left_fit, new_right_fit, mad_l, mad_r, mean_l, mean_r = p.fit_lanes(
                np.concatenate(lx_points_list),
                np.concatenate(ly_points_list),
                np.concatenate(rx_points_list),
                np.concatenate(ry_points_list),
                current_left_fit,
                current_right_fit,
                warped)
            bad = 0
        else:

            bad += 1
            # bad  results for 6 consecutive frames,  search again using convolution
            if bad > 6:
                find_with_convolution = True
            ploty, left_fitx, right_fitx, new_left_fit, new_right_fit, mad_l, mad_r, mean_l, mean_r = p.fit_lanes(
                leftx,
                lefty,
                rightx,
                righty,
                current_left_fit,
                current_right_fit,
                warped)
            

```
*** above an exemple of a polynomial fitted and points used from small area around last fitted polynomial.***

code in proc.py
```
def find_lanes_points(left_fit, right_fit, binary_warped, margin=13):
```
![alt text][image6]


####5.Curvature and Center Calculation 

I used calculation method proposed in the lessons, the code can be found in the functions above in `proc.py` file
```python
def calc_curvature( ploty, leftx, rightx): 
```
and 
```
def calc_center(leftx, rightx, warped):
```  
in file in `proc.py`

####6.Final Results


Can be seen in the videos bellow

 [project video result](./project_output.avi)
 [challenge video result](./challenge_output.avi)

---

###Discussion
The final results are good but it still perform poorly in the hard challenge video.
The approach using a window  with points from last n frames to smooth 
the results and small constant in the error will likely fail if the car ran faster and consequently lanes varies rapidly. Using a convolution the find the lanes do not  function if there is large area with noise. The algorithm will pick this area instead of the lanes.

One approach two have a better initialization, is would be search for bands with certain area and with  constant distance between the bands.
