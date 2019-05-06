import pickle
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    takeX = 1 if orient == 'x' else 0
    takeY = 1 if orient == 'y' else 0

    sobel = cv2.Sobel(gray, cv2.CV_64F, takeX, takeY)

    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt((sobelx ** 2) + (sobely ** 2))

    scaled_sobel = np.uint8(255 * mag / np.max(mag))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    dir_of_grad = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(dir_of_grad)
    binary_output[(dir_of_grad >= thresh[0]) & (dir_of_grad <= thresh[1])] = 1

    return binary_output

def saturation_threshold(img, sthresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary_output = np.zeros_like(s_channel)
    s_binary_output[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    return s_binary_output

def color_threshold(img, sthresh=(0, 255), rthresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary_output = np.zeros_like(s_channel)
    s_binary_output[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    r_channel = img[:,:,2]
    r_binary_output = np.zeros_like(r_channel)
    r_binary_output[(r_channel > rthresh[0]) & (r_channel <= rthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary_output == 1) | (r_binary_output == 1)] = 1

    return s_binary_output


def warp_image(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 35
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds =((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if(len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if(len(good_right_inds) > minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(leftx, lefty, 2)
    right_fit = np.polyfit(rightx, righty, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

dist_pickle = pickle.load(open('./camera_cal/calibration_pickle.p', 'rb'))

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

test_images = glob.glob('./test_images/test*.jpg')

for index, filename in enumerate(test_images):
    img = cv2.imread(filename)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Pre process image and generate binary pixels of interest
    pre_process_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, 'x', thresh=(60, 100))

    gradx = abs_sobel_thresh(img, 'x', thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))

    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) & (mag_binary == 1))] = 1

    s_binary = saturation_threshold(img, sthresh=(90, 255))

    pre_process_image[(combined == 1) & (s_binary == 1) | (combined == 1) ^ (s_binary == 1) ] = 255

    # defining perspective transformation area
    img_size = (img.shape[1], img.shape[0])

    # Source points - defined area of lane line edges
    ax = 536
    ay = 469
    bx = 737
    by = 469
    cx = 1140
    cy = 665
    dx = 138
    dy = 662
    src = np.float32([[ax,ay],[bx, by],[cx, cy],[dx, dy]])

    # 4 destination points to transfer
    dst = np.float32([[dx, ay],[cx, by],
                      [cx, cy],[dx, dy]])

        # Source points - defined area of lane line edges
    src = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

    # 4 destination points to transfer
    offset = 300 # offset for dst points
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])

    # Perform the transformation
    warped_image = warp_image(pre_process_image, src, dst)

    # Find lane pixels and fit a polynomial to the lane lines
    fitted_lane_img = fit_polynomial(warped_image)

    result = fitted_lane_img

    write_fname = './test_images/tracked' + str(index + 1) + '.jpg'
    cv2.imwrite(write_fname, result)

# img = plt.imread('./test_images/straight_lines1.jpg')
# plt.imshow(img)
# plt.show()