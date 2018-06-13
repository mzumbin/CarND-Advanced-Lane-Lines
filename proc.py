import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    if len(img.shape) == 2:
        gray =contrast(img)
    else:
        gray = contrast(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel =  contrast(hls[:,:,2])
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def hsv_select(img, thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = contrast(hsv[:, :, 2])
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel >= thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output


def gradient_pipeline(color_img_rgb):

    # Sobel x
    sobel_x =abs_sobel_thresh(color_img_rgb, orient='x', thresh_min=20, thresh_max=100)
    s_grad = hls_select(color_img_rgb, (80, 255))
    v_grad = hsv_select(color_img_rgb, (50, 255))
    # Combine the the  binary thresholds
    combined_binary = np.zeros_like(sobel_x)
    combined_binary[ ((s_grad == 1) & (v_grad == 1)) | (sobel_x ==1)] = 1
    return combined_binary


def divide_left_right_points_index(bin_img):
    midle = int(bin_img.shape[1]/2)
    rows,cols = np.nonzero(bin_img)
    cols_gt_midle = np.where(cols >= midle)
    cols_right = cols[cols_gt_midle]
    rows_right = rows[cols_gt_midle]

    cols_lt_midle = np.where(cols < midle)
    cols_left = cols[cols_lt_midle]
    rows_left = rows[cols_lt_midle]

    return rows_left, cols_left, rows_right, cols_right


def calibrate():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
# Make a list of calibration images
    images = glob.glob('camera_cal/cal*.jpg')  # Step through the list and search for chessboard corners

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

           # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            write_name = 'corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/test_undist.jpg', dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_cal/cam_dist_pickle.p", "wb"))
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)


def read_camera_instrinsics():
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open("camera_cal/cam_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    return mtx, dist


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def roundF(n):
   return int(round(n))


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height), max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def window_point_index(width, height, img_ref, center, level):
    row_start = int(img_ref.shape[0] - (level + 1) * height)
    row_end = int(img_ref.shape[0] - level * height)

    col_start = max(0, int(center - width / 2))
    col_end = min(int(center + width / 2), img_ref.shape[1])

    index_row,index_col = np.where( img_ref[row_start:row_end, col_start:col_end] > 0)
    return index_row + row_start, index_col + col_start


def find_window_centroids(warped, window_width, window_height, margin):

    rols_list_left = []
    cols_list_left = []

    rols_list_right = []
    cols_list_right = []
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    row_l,col_l = window_point_index(window_width,window_height,warped,l_center,0)
    rols_list_left.append(row_l)
    cols_list_left.append(col_l)

    row_r, col_r = window_point_index(window_width, window_height, warped, r_center, 0)
    rols_list_right.append(row_r)
    cols_list_right.append(col_r)

    window_centroids.append((l_center, r_center))

       # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l = conv_signal[l_min_index:l_max_index]
        non_zero_l = l[l > 0].shape[0]
        if non_zero_l >30:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r = conv_signal[r_min_index:r_max_index]
        non_zero_r = r[r > 0].shape[0]
        if non_zero_r >30:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        # Add what we found for that layer

        row_l, col_l = window_point_index(window_width, window_height, warped, l_center, level)
        rols_list_left.append(row_l)
        cols_list_left.append(col_l)

        row_r, col_r = window_point_index(window_width, window_height, warped, r_center, level)
        rols_list_right.append(row_r)
        cols_list_right.append(col_r)
        window_centroids.append((l_center, r_center))

    return window_centroids, (rols_list_left, cols_list_left), (rols_list_right, cols_list_right)


def centroid(warped, color, Minv):
    # Read in a thresholded image

    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    window_centroids, (rols_list_left, cols_list_left), (rols_list_right, cols_list_right) = find_window_centroids(warped, window_width, window_height, margin)


    rols_left = np.concatenate(rols_list_left, axis=0)
    cols_left = np.concatenate(cols_list_left)
    rols_right = np.concatenate(rols_list_right)
    cols_right = np.concatenate(cols_list_right)

    poly_l = fit(rols_left, cols_left, np.zeros(3))
    poly_r = fit(rols_right, cols_right, np.zeros(3))
    plot_poly(poly_l,poly_r,warped,rols_left,cols_left,rols_right,cols_right)
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    p_l = np.poly1d(poly_l)
    p_r = np.poly1d(poly_r)
    render_lanes(p_l(ploty), p_r(ploty), ploty,warped,color,Minv)
      # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channle
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warped = 166* warped
        warpage = np.array(cv2.merge((warped, warped, warped)),np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    #Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()




def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def find_lines_with_last_fit(binary_warped, left_fit, right_fit,margin=100):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    #left_fit = np.polyfit(lefty, leftx, 2)
    left_fit = fit(lefty, leftx, left_fit)
    p_l = np.poly1d(left_fit)
    mad_l = mean_absolute_error(p_l(lefty), leftx)
    right_fit = fit(righty, rightx, right_fit)
    p_l = np.poly1d(right_fit)
    mad_r = mean_absolute_error(p_l(righty), rightx)
    #right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return ploty, left_fitx, right_fitx, left_fit, right_fit , mad_l , mad_r


def find_lanes_points(left_fit, right_fit, binary_warped, margin=13):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_lanes(leftx, lefty, rightx, righty, left_fit, right_fit, binary_warped):
    left_fit = fit(lefty, leftx, left_fit)
    p_l = np.poly1d(left_fit)
    mad_l = mad_error(p_l(lefty), leftx)
    mean_l = mean_absolute_error(p_l(lefty), leftx)
    right_fit = fit(righty, rightx, right_fit)
    p_l = np.poly1d(right_fit)
    mad_r = mad_error(p_l(righty), rightx)
    mean_r =mean_absolute_error(p_l(righty), rightx)
    # right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return ploty, left_fitx, right_fitx, left_fit, right_fit, mad_l, mad_r,mean_l,mean_r


def poly_nd(x, t, y):
    return x[2]+t*x[1]+t*t*x[0] - y


def mad_error(prediction, y):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.abs(prediction-y) # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def fit(x, y, last_poly):
    res_robust = least_squares(poly_nd, last_poly, loss='cauchy', f_scale=3, args=(x, y))
    return res_robust.x
#calibrate()
#mtx , dist = read_camera_instrinsics()
#print(str(mtx))


def plot_poly(left_fit, right_fit, binary_warped, left_row, left_col, right_row, right_col):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    binary_warped = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8)
    binary_warped[left_row, left_col] = [255, 0, 0]
    binary_warped[right_row, right_col] = [0, 0, 255]

   # plt.imshow(binary_warped)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, binary_warped.shape[1])
    #plt.ylim(binary_warped.shape[0], 0)
    return  binary_warped


def render_lanes(left_fitx, right_fitx, ploty, warped, undist, Minv):
# Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

     # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def window_list(l, to_append, max_size=11):
    l.append(to_append)
    if len(l) > max_size:
        return l[-max_size:]
    else:
        return l


def contrast(image):
    phi = 1
    theta = 1
    maxIntensity = 255.0  # depends on dtype of image data
    # Decrease intensity such that
    # dark pixels become much darker,
    # bright pixels become slightly dark
    newImage1 = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2
    return newImage1


def equalize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output


def calc_curvature( ploty, leftx, rightx):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty) * ym_per_pix
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad, right_curverad


def calc_center(leftx, rightx, warped):
    xm_per_pix = 3.7 / 700
    camera_center = (leftx[-1]+rightx[-1])/2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    if center_diff <= 0:
        side_pos = 'left'
    else:
        side_pos = 'right'

    return center_diff, side_pos
