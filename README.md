# 475Heterogen-Comp-Sig-Processing-CourseProject
Course Project for 4750 HETEROGEN COMP-SIG PROCESSING. 

Language: Python, PyOpenCL



1. Descriptions of the project, sub-modules, and files

This project aims at parallelize the SIFT algorithm. We has referred to the SIFT code in OpenCV, which is squential program for CPU. We divide the whole project into six parts, and use six kernels to implement them. 

There are six files in total:
    README.txt, 
    lenel.png (source image for SIFT algorithm), 
    optimize.cpp & optimize.py (contains the code for our optimized kernel)
    naive.cpp & naive.py (contains the code for naive version, only output time for comparison)


2. Instructions to compile and run the program

Upload the image (lenel.png) and program (optimize.cpp & optimize.py) to the server, and they should be put in same path.

Run optimize.py file directly, and will generate 3 images.

naive.cpp & naive.py is used for time comparison only.


3. Expected output

optimize.cpp & optimize.py:

   the running time for each kernels, and verification for some of them

   the number of local extrema and keypoints

   3 images, one for our program, one for OpenCV SIFT, one for OpenCV SURF

naive.cpp & naive.py:

   running time for kernels
