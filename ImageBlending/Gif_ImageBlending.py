# Image Blending with Gif

import numpy as np
import cv2
import imageio

# Read foreground image
fg = cv2.imread(r"girl_with_wreath.jpg")
#print('Kich thuoc anh foreground: ', fg.shape)

# Read mask image
mask = cv2.imread(r"girl_with_wreath-removebg.png", cv2.IMREAD_UNCHANGED)
#print('Kich thuoc anh mask: ', mask.shape)

url = "https://media2.giphy.com/media/aEWqT7q0sLVjW/giphy.gif?cid=ecf05e47up509f2dhvxr5rdjdlv3kdppe0zwubs0d7m713mf&rid=giphy.gif&ct=g"
frames = imageio.mimread(imageio.core.urlopen(url).read(), '.gif')

bgs = [frame[:, :, 0:3] for frame in frames]
for i in range(len(bgs)):
    bgs[i] = cv2.resize(bgs[i], (464, 600)) # (464, 600) is the size of foreground


# Gif blending
results = []
alpha = 0.7
for i in range(len(bgs)):
    result = fg.copy()
    result[mask[:,:,3] != 0] = alpha * result[mask[:,:,3] != 0]
    bgs[i][mask[:,:,3] == 0] = 0
    bgs[i][mask[:,:,3] != 0] = (1-alpha)*bgs[i][mask[:,:,3] != 0]
    result = result + bgs[i]
    results.append(result)

imageio.mimsave("D:/Downloads/result.gif", results)  # save the result