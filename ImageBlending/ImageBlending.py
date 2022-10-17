# Image Blending 
# Image Blending is mixing two images and foreground image mask of the corresponding pixel values to create a new target image.
# Mask image : image that removed background

import numpy as np
import cv2

# Read foreground image
fg = cv2.imread(r"girl_with_wreath.jpg")
print('Kich thuoc anh foreground: ', fg.shape)

# Read background image
bg = cv2.imread(r"jungle.png")
print('Kich thuoc anh background: ', bg.shape)

# Read mask image
mask = cv2.imread(r"girl_with_wreath-removebg.png", cv2.IMREAD_UNCHANGED)
print('Kich thuoc anh mask: ', mask.shape)

# Resize background image follow the size of foreground image
bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
print('bg sau resize : ', bg.shape)
#mask = cv2.resize(mask, (fg.shape[0], fg.shape[1]))

# Show image 
#cv2.imshow('Foreground', fg)
#cv2.imshow('Background', bg)
#cv2.imshow('Maskground', mask)
# cv2.waitKey(0)


# a = np.ones((mask.shape[0],mask.shape[1] , 3))
# a = np.cumsum(np.ones((mask.shape[0],mask.shape[1] , 1)))
# print('a: ', a.shape)


# # Blending
result = fg.copy()

# alpha = np.array([1 - abs(mask.shape[1]/2 - np.cumsum(np.ones((mask.shape[0],mask.shape[1] , 1)))) / (mask.shape[0]/2)])
# #alpha = cv2.resize(alpha, (fg.shape[1], fg.shape[0], 3))
# alpha = np.reshape(alpha , (600, 464, 3))
# print('kich thuoc : ', alpha.shape)
# x = alpha*fg
# #print(x)

# result = alpha*fg

# a = alpha * fg 
# cv2.imshow('Result', a)
# cv2.waitKey(0)

alpha = [[]]





# Ý tưởng: Để ảnh có thể chuyển màu nhạt dần ra ngoài rìa
# Chọn điểm ảnh trung tâm (weight//2 nếu vật thể nằm giữa khung hình)
# Từ điểm ảnh trung tâm, duyệt y từ 0-> weight, y < điểm trung tâm thì giảm alpha
# Nếu y > điểm trung tâm, tăng alpha
# Vì thế, ảnh có thể chuyển màu nhạt dần ra rìa
for x in range(mask.shape[0]): # height
    alpha = 0.3
    for y in range(mask.shape[1]):
        if (mask[x,y,3] != 0):
            if (y < mask.shape[1]//2): # Nếu y < điểm trung tâm => giảm alpha để tăng trọng số của ảnh effect
                alpha = alpha - 1/(10*y)
                if alpha < 0:
                    alpha = 0
            if (y > mask.shape[1]//2):
                alpha = alpha + 0.007 # Nếu y > điểm trung tâm => tăng alpha để giảm trọng số ảnh hưởng của effect 
                if alpha > 1: # Nếu alpha vượt ngưỡng màu => đặt alpha = 1
                    alpha = 1
            result[x,y] = alpha * fg[x,y] + (1 - alpha) * bg[x,y]



# alpha = 0.3
# p = mask.shape[1]//2
# result = fg.copy()
# result[mask[:,:,3] != 0] = [alpha - 1/(10*2*p), 0][alpha - 1/(10*2*p) and  alpha - 1/(10*2*p) > 0.3] * fg[mask[:,:,3]!=0] + (1 - [alpha + 1/(10*2*p), 0][alpha - 1/(10*2*p) > 0.3]) * bg[mask[:,:,3]!=0]


# result[mask[:,:,3] != 0] = alpha * fg[mask[:,:,3]!=0] + (1 - alpha) * bg[mask[:,:,3]!=0]



cv2.imshow('Result', result)
cv2.waitKey(0)
