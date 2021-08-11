#Paying around in Open CV
#11 August 2021


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
cnt=0


img = cv.imread('D:\Downloads 2.0\sheep2.jpg')
img = cv.resize(img,(600,600))
cv.imshow('Original',img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
thresh = cv.bitwise_not(thresh)
cv.imshow("Thresh",thresh)

contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
objects = str(len(contours))
text = "Obj:"+str(objects)
cv.putText(thresh, text, (10, 25),  cv.FONT_HERSHEY_SIMPLEX,0.4, (240, 0, 159), 1)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow("opening",opening)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations = 3)


# sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
cv.imshow("Distance Transform", dist_transform)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region ( back - for )
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow("Unknown Pixels",unknown)

# Marker labelling
ret, markers = cv.connectedComponents(opening)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0


#Watershed
markers = cv.watershed(img,markers)
number= len(markers)
print("Number: "+ str(number))
img[markers == -1] = [255,0,0]

cv.imshow("markers",img)

#

print("Count: "+cnt)
cv.waitKey(0)
cv.destroyAllWindows()