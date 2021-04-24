
import cv2

# Reading the image
filename = 'images/style.in4.jpg'
source = cv2.imread(filename,1)

#scaleX is scale factor in x direction
#scaleY is scale factor in y direction
scaleX = 0.6
scaleY = 0.6


# Scaling Down the image 0.6 times
scaleDown = cv2.resize(source, None, fx= scaleX, fy= scaleY, interpolation= cv2.INTER_LINEAR)

# Scaling up the image 1.8 times
scaleUp = cv2.resize(source, None, fx= scaleX*3, fy= scaleY*3, interpolation= cv2.INTER_LINEAR)

#Cropped Image
crop = source[50:150,20:200]

# Displaying all the images
cv2.imshow("Original", source)
cv2.imshow("Scaled Down", scaleDown)
cv2.imshow("Scaled Up", scaleUp)
cv2.imshow("Cropped Image",crop)

cv2.waitKey(0)
