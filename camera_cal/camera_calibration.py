import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

images = glob.glob('calibration*.jpg')

print(images)

# Parameters

nx = 9
ny = 6

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in the image plane

objp = np.zeros((ny * nx, 3), np.float32)
objp[:,:2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2) # x, y coordinates

for index, filename in enumerate(images):
    print(filename)
    img = cv2.imread(filename)

    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If corners are found, add object points and image points
    if ret:
        print('working on ', filename)

        objpoints.append(objp)
        imgpoints.append(corners)

        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        corner_img_fname = 'corners_found' + str(index) + '.jpg'
        cv2.imwrite(corner_img_fname, img)

        # draw the detected corners on the image

    plt.imshow(img)


img = cv2.imread('calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))
