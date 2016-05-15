Intro
=====

This is a collection of code I've put together to perform object tracking of a single deformable colour object using camshift algorithm from OpenCV along with SURF for robust tracking. 2D Kalman filter is applied on the predictions to smoothen out the errors and make the prediction more robust. 

References:

G. R. Bradski. "Computer vision face tracking for use in a perceptual user interface", Intel Technology Journal, 2nd Quarter, 1998. 

Liu, Q., Tang, L.B. and Zhao, B.J. (2012) Meanshift Tracking Algorithm with Adaptive Tracking Window. System Engineering and Electronics, 34, 409-412.

Herbert, B., Andreas, E., Tinne, T., et al. (2006) Speeded-Up Robust Features. Computer Vision and Image Understanding, 404-417.


Requirements
============

All code in this package requires the OpenCV library (known working 
version is 3.1):
https://github.com/Itseez/opencv
https://github.com/Itseez/opencv_contrib.git

Building
========

To build everything, use make:

	mkdir build; cd build
    cmake .. 
    make

This should produce an executable tracking.

License
=======

See the file LICENSE for more information on the legal terms of the use 
of this package.
