## This code takes the known intrinsic params of the camera and outputs the Rotation & Translation vectors.

import numpy as np
import cv2 as cv
import glob

def draw(img, corners, imgpts):
    corner = tuple(int(el) for el in corners[0].ravel())
    img = cv.line(img, corner, tuple(int(el) for el in imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(int(el) for el in imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(int(el) for el in imgpts[2].ravel()), (0,0,255), 5)
    return img

## Input Intrinsic Camera Parameters
#left
# mtx_1 = ([906.789900045893,-1.48120980115051,652.531687107407],[0,914.235393453131,348.016355255329],[0,0,1])
# dist_1 = ([-0.364841324031623,0.190085683807103,0,0.000000,0.000000])
#right
mtx_1 = ([905.092280449818,-1.44151624935301,668.597751702037],[0,912.965034309910,366.148526103497],[0,0,1])
dist_1 = ([-0.358686975494141,0.173997072572930,0, 0, 0])
mtx = np.float32(mtx_1)
dist = np.float32(dist_1)

## Input Chessboard Params
l = 8 #w
b = 6 #h

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((b * l, 3), np.float32)
objp[:,:2] = np.mgrid[0:b, 0:l].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
#print(axis)

for fname in glob.glob('test-modified.jpeg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow('gray',gray)
    ret, corners = cv.findChessboardCorners(gray, (b,l),None)
    
    if ret == False:
        print("\n Something is Wrong, Chessboard not detected")
    
    else:
        corners2 = corners
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  # for subpixel accuracy
        
        # Find the rotation and translation vectors using PnP
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d

        ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_ITERATIVE)
        # ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_EPNP)
        # ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_IPPE) # Infinitesimal Plane-Based Pose Estimation
        # ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_SQPNP)

        ## ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_P3P) #-> Doesnot Work
        ## ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_DLS) #-> falls back to EPNP
        ## ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_UPNP) #-> falls back to EPNP
        ## ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_AP3P) #-> Doesnot Work
        ## ret1,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist, flags=cv.SOLVEPNP_IPPE_SQUARE) #-> Doesnot Work. Infinitesimal Plane-Based Pose Estimation. This is a special case suitable for marker pose estimation.
        
        
        print("\n Rotation Vectors:")
        print(rvecs)
        print("\n Translation Vectors:")
        print(tvecs)

        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # print(imgpts)
        # print(jac)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'-after-pnp.png', img)
cv.destroyAllWindows()