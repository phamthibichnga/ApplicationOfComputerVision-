'''
Edge detection with gradient
Step 1: Load image in gray-scale mode

Step 2: Compute derivative with respect to x: Dx

Step 3: Compute derivative with respect to y: Dy

Step 4: Compute gradient norm: D = sqrt(Dx^2 + Dy^2)

Step 5: Visualize result D
'''

# Load img
from cmath import sqrt
import cv2
import numpy as np
  
# path
path = r"9-ro.jpeg"
 
img = cv2.imread(path)
print('Shape of img :', img.shape)

# Load image in grayscale 
img_grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print('Shape of img grayscale :', img_grey.shape)

# Resize image 
img_grey = cv2.resize(img_grey, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
print('Shape of resize img grayscale :', img_grey.shape)

# derivative in the x-direction - horizontal 
filterX = np.array([[-1, 0, 1], 
           [-1, 0, 1], 
           [-1, 0, 1]])

# derivative in the y-direction - vertical 
filterY = np.array([[1, 1, 1], 
           [0, 0, 0], 
           [-1, -1, -1]])

D = img_grey.copy()

# filterX.shape = filterY.shape
for i in range(0, img_grey.shape[0] - filterX.shape[0] + 1):
    for j in range(0, img_grey.shape[1] - filterX.shape[1] +1):
        Ix = np.sum(img_grey[i:i + filterX.shape[0], j:j + filterX.shape[1]] * filterX)/3 # Compute derivative with respect to x: Dx
        Iy = np.sum(img_grey[i:i + filterX.shape[0], j:j + filterX.shape[1]] * filterY)/3 # Compute derivative with respect to y: Dy
        D[i, j] = np.sqrt(np.square(Ix) + np.square(Iy)) #  Compute gradient norm: D = sqrt(Dx^2 + Dy^2)
    
# Displaying the image
# cv2.imshow('image', img_grey)
cv2.imshow('image', D)
cv2.waitKey(0)
