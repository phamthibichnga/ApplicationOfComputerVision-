import cv2
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt


#Reading the source and sub images
img_c = cv2.imread(r"9ro.png")
template = cv2.imread(r'ro1.png',0) #Reading sub image in grey mode
img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) 


def Normalised_Cross_Correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor = np.sum(roi*target)
    nor = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
    #print('Nor : ', nor)
    return cor / nor 


def matching(img, target):
    # initial parameter
    height,width=img.shape
    tar_height,tar_width=target.shape

    # Set image, target and result value matrix
    img=np.array(img, dtype="int")
    target=np.array(target, dtype="int")
    NccValue=np.zeros((height-tar_height+1,width-tar_width+1))

    # calculate value using filter-kind operation from top-left to bottom-right
    for y in range(0,height-tar_height+1):
        for x in range(0,width-tar_width+1):
            # image roi
            roi=img[y:y+tar_height,x:x+tar_width]
            # calculate ncc value
            NccValue[y,x] = Normalised_Cross_Correlation(roi,target)      
    return NccValue

# Draw bounding box around the object
def draw_boudingbox(index):
    for i in range(index.shape[0]):
        #print('index :', index[i])
        cv2.rectangle(img_c, (index[i][1],index[i][0]), ((index[i][1],index[i][0])[0] + template.shape[1], (index[i][1],index[i][0])[1] + template.shape[0]), (0,0,255) , 5)
        
        
# Call the template matching function
NCCValue = matching(img, template)

# Select top k index have max value 
n = 9
index = np.argwhere(np.isin(NCCValue, np.sort(NCCValue, axis=None)[-n:]))
#print(index)  

# Draw bouding box       
draw_boudingbox(index)

plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
plt.show()
