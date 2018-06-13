import cv2
import matplotlib.pyplot as plt
import proc as p
import numpy as np
import regressor as r

plt.interactive(True)


def calc_homography(img):
    size_x = img.shape[1]
    size_y = img.shape[0]
    bot_width = 0.76
    mid_width = 0.05
    height_pct = 0.69
    bottom_trim = 0.93
    mid_half = size_x / 2.
    offset = size_x * 0.25
    top_y = size_y * height_pct
    x_left = (mid_half - (size_x * mid_width) / 2)
    x_right = (mid_half + (size_x * mid_width) / 2)
    x_left_bot = mid_half - (size_x * bot_width / 2)
    x_right_bot = mid_half + (size_x * bot_width / 2)
    bottom_y = size_y * bottom_trim
    src = np.float32([[x_left, top_y], [x_right, top_y], [x_right_bot, bottom_y], [x_left_bot, bottom_y]])
    dst = np.float32([[offset, 0], [size_x - offset, 0], [size_x - offset, size_y], [offset, size_y]])
    h = cv2.getPerspectiveTransform(src, dst)
    inv_h = cv2.getPerspectiveTransform(dst, src)
    return h, inv_h


def find_good_homography(color_rgb):
    edges = p.gradient_pipeline(color_rgb)
    #edges = cv2.GaussianBlur(edges, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    #cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow('warped', 255*edges)
    shape = edges.shape
    vertices = np.array([[(110, shape[0]), (592, 433), (674, 433), (1180, shape[0])]], dtype=np.int32)
    edges = p.region_of_interest(edges, vertices)
    rows_left, cols_left, rows_right, cols_right = p.divide_left_right_points_index(edges)
    H, inv_h, good_value = r.find_line(cols_left, rows_left, cols_right, rows_right, edges.shape)
    if True:
        warped = cv2.warpPerspective(edges, H, (edges.shape[1], edges.shape[0]))
        cv2.imshow('warped', 255 * warped)
        print('asd')
        return  H, inv_h


from sklearn.metrics import mean_absolute_error
mtx, dist = p.read_camera_instrinsics()

## image wih straight lines
# line 10 striah line to challenge video, use test_images/straight_line2.png for project video
to_find_homography = cv2.imread('line10.png')
to_find_homography = cv2.undistort(to_find_homography, mtx, dist, None, mtx)
to_find_homography = cv2.cvtColor(to_find_homography, cv2.COLOR_BGR2RGB)

h,inv_h = find_good_homography(to_find_homography)
color_bgr = cv2.imread('line10.png')
color_bgr = cv2.undistort(color_bgr, mtx, dist, None, mtx)

color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


edges = p.gradient_pipeline(color_rgb)
plt.imshow(edges)
# Threshold it so it becomes binary
# h,inv_h=r.find_line(cols_left,rows_left,cols_right,rows_right,edges.shape)

#connectivity = 4
# Perform the operation
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.GaussianBlur(edges, (5, 5), 0)

plt.imshow(edges)
shape = edges.shape

warped = cv2.warpPerspective(edges, h, (edges.shape[1], edges.shape[0]))


plt.imshow(warped)
p.centroid(warped, color_rgb, inv_h)
# plt.imshow(warped)
# output = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
# num_labels = output[0]
# The second cell is the label matrix
# labels = output[1]

# plt.imshow(labels)
# The third cell is the stat matrix
# stats = output[2]
# gt_area = stats[:, 4] > 30
# output[gt_area, :] = 0
# plt.imshow(output)
# plt.imshow(stats)
# The fourth cell is the centroid matrix
# centroids = output[3]
cap = cv2.VideoCapture('/home/marcelo/CarND-Advanced-Lane-Lines/harder_challenge_video.mp4')
current_left_fit = 0
current_right_fit = 0
find_with_convolution = True

mads_l = []
mads_r = []
lx_points_list = []
ly_points_list = []
rx_points_list = []
ry_points_list = []
kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0

kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])


fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output2.avi', fourcc, 30.0, (edges.shape[1], edges.shape[0]))
bad = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    color_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    color_rgb = cv2.undistort(color_rgb, mtx, dist, None, mtx)
    color_rgb_undist_clone = np.copy(color_rgb)
    # histogram equalization YUV in Y coordinate
    color_rgb = p.equalize(color_rgb)
    #color and gradient threshold
    edges = p.gradient_pipeline(color_rgb)
    # filter some noise
    color_rgb = cv2.medianBlur(color_rgb, 5)
    kernel = np.ones((5, 5), np.uint8)
    # dilate to get more points
    edges = cv2.dilate(edges, kernel, iterations=1)
    #edges = cv2.GaussianBlur(edges, (5, 5), 0)
    warped = cv2.warpPerspective(edges, h, (edges.shape[1], edges.shape[0]))
    cv2.imshow('edges', 255 * warped)
    # plt.imshow(warped)
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    if find_with_convolution:
        window_centroids, (rols_list_left, cols_list_left), (
        rols_list_right, cols_list_right) = p.find_window_centroids(
            warped, window_width, window_height, margin)

        rols_left = np.concatenate(rols_list_left, axis=0)
        cols_left = np.concatenate(cols_list_left)
        rols_right = np.concatenate(rols_list_right)
        cols_right = np.concatenate(cols_list_right)

        current_left_fit = p.fit(rols_left, cols_left, np.zeros(3))
        current_right_fit = p.fit(rols_right, cols_right, np.zeros(3))
        find_with_convolution = False
        #p.plot_poly(current_left_fit, current_right_fit, warped, rols_left, cols_left, rols_right, cols_right)
        # ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        # p_l = np.poly1d(poly_l)
    # p_r = np.poly1d(poly_r)
    # result = p.render_lanes(p_l(ploty), p_r(ploty), ploty, warped, color_rgb, inv_h)

    # ploty, left_fitx, right_fitx, new_left_fit, new_right_fit, mad_l, mad_r = p.find_lines_with_last_fit(warped, current_left_fit, current_right_fit)
    leftx, lefty, rightx, righty = p.find_lanes_points(current_left_fit, current_right_fit, warped)
    dif = np.mean(rightx) - np.mean(leftx)
    #test if lines makes sense
    if len(leftx) > 50 and len(rightx) > 50:
        if abs(dif - 650) < 50 :
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


    curv_l, curv_r = p.calc_curvature(ploty, left_fitx, right_fitx)
    center_diff, side_pos = p.calc_center(left_fitx, right_fitx, warped)

    points_image = p.plot_poly(new_left_fit, new_right_fit, warped, lefty, leftx, righty, rightx)
    points_image = cv2.warpPerspective(points_image, inv_h, (points_image.shape[1], points_image.shape[0]))

    result = p.render_lanes(left_fitx, right_fitx, ploty, warped, color_rgb_undist_clone, inv_h)
    result = cv2.addWeighted(result, 1, points_image, 0.5, 0)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.putText(result, 'Radius of curvature  left lane= ' + str(round(curv_l, 3)) +'m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(result, 'Radius of curvature  right lane= ' + str(round(curv_r, 3)) + 'm', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm '+side_pos , (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

  #  cv2.putText(result, 'mad left= ' + str(round(mean_l, 3)), (50, 160),
  #              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
  #  cv2.putText(result, 'mad right= ' + str(round(mean_r, 3)), (50, 190),
  #              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(result, 'med dif= ' + str(round(dif, 3)), (50, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('frame', result)
    current_right_fit = new_right_fit
    current_left_fit = new_left_fit
    out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

