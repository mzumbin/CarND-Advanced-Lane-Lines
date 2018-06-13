from matplotlib import pyplot as plt
import numpy as np
import cv2
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


np.random.seed(42)



poly = PolynomialFeatures(2)
reg = LinearRegression(PolynomialFeatures(2))

ransac_estimator = RANSACRegressor(reg, min_samples=3,random_state=42)
RANSACRegressor()
regressor = HuberRegressor()
model = make_pipeline(poly, regressor)

x=np.linspace(0, 5, 200)
y=x*x
model.fit(x.reshape(-1, 1), y)
ransac_estimator.fit(x.reshape(-1, 1), y)
#egressor.warm_start = True
#egressor.fit(x.reshape(-1, 1),y)

#rint('oi')
x0 = np.zeros(3)
def fun(x,t, y):
    return x[2] + t*x[1]+ t*t*x[0] -y
from scipy.optimize import least_squares

def mad_error(prediction, y):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.abs(prediction-y) # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

res_robust = least_squares(fun, x0, loss='cauchy', f_scale=0.1, args=(x, y))
pred = np.poly1d(res_robust.x[::-1])
pred = pred(x)

mad = mad_error(pred,y)
print(str(mad))
def line (x,t,y):
    return x[0]+t*x[1]-y


def iterception(line1,line2):
    m1 = line1[1]
    b1 = line1[0]

    m2 = line2[1]
    b2 = line2[0]

    x = (b2 - b1)/(m1 - m2)
    y = m1 *x +b1
    return x,y


def find_line(col_left, row_left, col_right, row_right,img_shape):

    x0 = np.zeros(2)
    res = least_squares(line, x0, loss='cauchy', f_scale=5, args=(col_left, row_left))
    line_left = res.x
    line_right = least_squares(line, x0, loss='cauchy', f_scale=5, args=(col_right, row_right)).x

    x,y =iterception(line_left, line_right)
    if abs(x - img_shape[1]/2)< 5:
        good_value = True
    else:
        good_value = False
    top_y = y + 50# top y coordinate,  50 pixels above where lines intercept
    top_left_x = (top_y - line_left[0])/line_left[1]

    top_right_x = (top_y - line_right[0])/line_right[1]

    bottom_y = img_shape[0] - 20

    bottom_left_x = (bottom_y - line_left[0])/line_left[1]
    bottom_right_x = (bottom_y - line_right[0])/line_right[1]

    src = np.float32([[bottom_left_x, bottom_y], [top_left_x, top_y], [top_right_x, top_y], [bottom_right_x,bottom_y]])

    quart_x =img_shape[1]/4
    max_y = img_shape[0]
    dst = np.float32([[quart_x, max_y], [quart_x, 0], [3*quart_x, 0], [3*quart_x, max_y]])

    H = cv2.getPerspectiveTransform(src, dst)
    inv_h = cv2.getPerspectiveTransform(dst, src)
    return  H, inv_h , good_value









