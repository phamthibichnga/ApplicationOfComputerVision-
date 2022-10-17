
import numpy as np
import cv2
from imageio import imread, imwrite
import imageio
from scipy.ndimage import convolve
import matplotlib.pyplot as plt 


def Calculate_EnergyMap(img):
    # Create energy map using Sobel filter (convolutional kernel) on every channel. 
    # 2 filter for 2 different directions of an image 
    filter_dx = np.array([[1, 2, 1], 
                      [0, 0, 0], 
                      [-1, -2, -1]]) 

    filter_dy = np.array([[1, 0, -1], 
                     [2, 0, -2], 
                     [1, 0, -1]])

    # This convert it from 2D filter to 3D filter, filter for each channel R,G, B
    filter_dx = np.stack([filter_dx] * 3, axis=2)
    filter_dy = np.stack([filter_dy]*3, axis=2)

    # print(filter_dx)
    # print(filter_dy)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_dx)) + np.absolute(convolve(img, filter_dy))


    # Sum the ennergies in red, green, bue channels
    energy_map = convolved.sum(axis=2)
    # plt.imshow(energy_map, cmap='gray')
    # plt.show()
    
    return energy_map

# Finding the seam with least energy
def Minimum_Seam_matrix(img):
    row, column, channel = img.shape
    energy_map = Calculate_EnergyMap(img)
    
    # Create matrix to save minimum energy value of each pixel
    Minimun = energy_map.copy()
    
    # Backtrack save the position to find the seam 
    Backtrack = np.zeros_like(Minimun)
    
    for i in range(1, row):
        for j in range(column):
            if j == 0:
                idx = np.argmin(Minimun[i-1, j:j+2])
                Backtrack[i, j] = idx # In function, [i-1, j - 1 : j + 2] formated to 0-n, so we need to + j -1 to have correct idx
                min_energy = Minimun[i-1, idx]
            else:  
                idx = np.argmin(Minimun[i-1, j - 1 : j + 2])
                Backtrack[i, j] = idx + j - 1 # In function, [i-1, j - 1 : j + 2] formated to 0-n, so we need to + j -1 to have correct idx 
                min_energy = Minimun[i-1, idx + j - 1]

            Minimun[i, j] += min_energy
    return Minimun, Backtrack
            
# Delete the pixels from the seam with the least energy
def Delete_SeamLine(img):
    row, column, channel = img.shape
    Minimun, Backtrack = Minimum_Seam_matrix(img)
    
    # Create a (r, c) matrix filled with the value True 
    # Remove pixels from the image which have False value
    
    mask = np.ones((row, column), dtype=np.bool_) 
    
    # find the index have smallest element in the last row of M 
    j = np.argmin(Minimun[-1])
    
    for i in reversed(range(row)):
        # mark the index to delete 
        mask[i, j] = False
        j = int(Backtrack[i, j])
        
    # Convert mask to 3D channel 
    mask = np.stack([mask]*3, axis=2)
    
    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((row, column - 1, 3))
    
    return img

def Crop_Column_Scale(img, scale):
    row, column, _ = img.shape
    column_new = int(column*scale)
    
    for i in range(column - column_new):
        img = Delete_SeamLine(img)
        
    return img
    
def main():
    scale = 0.7 # Value: 0->1 (0-100%)
    # Load image 
    img = imageio.v2.imread("artcloud.png") 
    print(img.shape)
    plt.imshow(img)
    plt.show()
    
    image_cropped = Crop_Column_Scale(img, scale)
    plt.imshow(image_cropped)
    plt.show()
    
if __name__ == '__main__':
    main()
