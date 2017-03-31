import cv2
import matplotlib.pyplot as plt
import proc as p
import numpy as np
import regressor as r
# Read the image you want connected components of
src = cv2.imread('test_images/straight_lines1.jpg',0)
color_bgr = cv2.imread('test_images/straight_lines1.jpg')
color_rgb  =cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
edges =p.gradient_pipeline(color_rgb)

# Threshold it so it becomes binary
#h,inv_h=r.find_line(cols_left,rows_left,cols_right,rows_right,edges.shape)
ret, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# You need to choose 4 or 8 for connectivity type

connectivity = 4
# Perform the operation
edges = cv2.GaussianBlur(edges,(5,5),0)
plt.imshow(edges)
shape = edges.shape
vertices = np.array([[(80,shape[0]),(592, 433), (674, 433), (1180,shape[0])]], dtype=np.int32)
edges = p.region_of_interest(edges,vertices)
rows_left, cols_left, rows_right, cols_right =p.divide_left_right_points_index(edges)
h,inv_h=r.find_line(cols_left,rows_left,cols_right,rows_right,edges.shape)
warped = cv2.warpPerspective(edges, h, (edges.shape[1],edges.shape[0]))
#plt.imshow(warped)
p.centroid(warped,color_rgb,inv_h)
#plt.imshow(warped)
#output = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
#num_labels = output[0]
# The second cell is the label matrix
#labels = output[1]

#plt.imshow(labels)
# The third cell is the stat matrix
#stats = output[2]
#gt_area = stats[:, 4] > 30
#output[gt_area, :] = 0
#plt.imshow(output)
#plt.imshow(stats)
# The fourth cell is the centroid matrix
#centroids = output[3]
cap = cv2.VideoCapture('/home/marcelo/CarND-Advanced-Lane-Lines/project_video.mp4')
current_left_fit = 0
current_right_fit = 0
first_frame = True
mads_l = []
mads_r = []
lx_points_list = []
ly_points_list = []
rx_points_list = []
ry_points_list = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("false false")

    color_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    edges = p.gradient_pipeline(color_rgb)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    warped = cv2.warpPerspective(edges, h, (edges.shape[1], edges.shape[0]))

   # plt.imshow(warped)
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    if first_frame:
        window_centroids, (rols_list_left, cols_list_left), (rols_list_right, cols_list_right) = p.find_window_centroids(
        warped, window_width, window_height, margin)

        rols_left = np.concatenate(rols_list_left, axis=0)
        cols_left = np.concatenate(cols_list_left)
        rols_right = np.concatenate(rols_list_right)
        cols_right = np.concatenate(cols_list_right)

        current_left_fit = p.fit(rols_left, cols_left, np.zeros(3))
        current_right_fit = p.fit(rols_right, cols_right, np.zeros(3))
        first_frame = False
    #p.plot_poly(poly_l, poly_r, warped, rols_left, cols_left, rols_right, cols_right)
    #ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
   # p_l = np.poly1d(poly_l)
    #p_r = np.poly1d(poly_r)
    #result = p.render_lanes(p_l(ploty), p_r(ploty), ploty, warped, color_rgb, inv_h)

    ploty, left_fitx, right_fitx, new_left_fit, new_right_fit, mad_l, mad_r = p.find_lines_with_last_fit(warped, current_left_fit, current_right_fit)
    p
    #mads_l.append(mad_l)
    #mads_r.append(mad_r)
    result = p.render_lanes(left_fitx, right_fitx, ploty, warped, color_rgb, inv_h)
    cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    current_right_fit = new_right_fit
    current_left_fit =  new_left_fit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()