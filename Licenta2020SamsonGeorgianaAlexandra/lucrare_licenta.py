import pip
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


"""
from sklearn.cluster import MeanShift
X = img
clustering = MeanShift().fit(X)
MeanShift(bandwidth=1, bin_seeding=False, cluster_all=True, min_bin_freq=1,
     n_jobs=None, seeds=None)

cv2.imshow('X',clustering)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################

"""

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

image = cv2.imread('download.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

"""
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)


faces = faceCascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex-20,ey-20),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
blur = cv2.blur(gray, (3, 3)) # blur the image
#ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

#try more prcised treshold
a = gray.max()
ret, thresh = cv2.threshold(gray, a/2+60, a, cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap = 'gray')

# Finding contours for the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create hull array for convex hull points
hull = []

# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(image, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(image, hull, i, color, 1, 8)

print(thresh.shape[1])
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
x, y, z = image.shape
drawing = drawing.reshape(x, y, z).astype(np.uint8)
print(image.shape)
cv2.imshow('img',image)
cv2.waitKey(0)

cv2.destroyAllWindows()

for i in range(len(contours)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(image, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(image, hull, i, color, 2, 8)
    image[np.where((image == [0, 0, 0]).all(axis=2))] = [0, 33, 166]
    #cv2.colorChange()

cv2.imshow('output.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
print(image.shape)

imgYCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(imgYCC,'Schimbat culorile',
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    lineType)

cv2.imshow("Face", imgYCC)
cv2.waitKey(0)



#####################################   Kmeans over an image

#matplotlib inline
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
from collections import Counter
import pprint
import imutils


image = cv2.imread('fata.jpg')
plt.figure(figsize = (5, 8))
plt.imshow(image)
plt.waitforbuttonpress()
print(image.shape)

x, y, z = image.shape
image_2d = image.reshape(x*y, z)

print(image_2d.shape)

##################################### Apply cluster
kmeans_cluster = cluster.KMeans(n_clusters=7)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

#to get all the items in the first cluster just get the indexes int km_labels with the

value = 1

index = [x[0] for x, value in np.ndenumerate(cluster_labels) if value==1]

#cluster_point=[178.00089892, 205.55825233, 242.76362496]

#for i in cluster_centers:
#   i = [255, 20, 255]

f = open("out", "w")
f.write(str(index))

################################# Recreate image with standards dimensions

plt.figure(figsize = (15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z).astype(np.uint8))
plt.waitforbuttonpress()

"""

#######################################################################################################################################################
# the biggest contour
# load the image

from PIL import Image

image = cv2.imread('download.jpg')

# red color boundaries [B, G, R]
lower = [1, 0, 50]
upper = [80, 60, 220]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask

mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

ret,thresh = cv2.threshold(mask, 40, 255, 0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were founded
    cv2.drawContours(output, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    #c = max(contours, key = cv2.contourArea)

    contours = sorted(contours, key=cv2.contourArea, reverse = True)

    # try to find by perimeter but doesn't work
    # perimeter = max(contours, key = cv2.arcLength(contours, False))

    #x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    img_copy = image.copy()


    final = cv2.drawContours(img_copy, contours, contourIdx=-1, color=(255,0,255), thickness=2)
    #cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

f = open("out.txt", "w")
f.write(str(contours))#stick all contours and after that discard

# show the images
cv2.imshow("Result", np.hstack([image, output]))

cv2.waitKey(0)
cv2.imshow("Result", np.hstack([img_copy, output]))
cv2.waitKey(0)
