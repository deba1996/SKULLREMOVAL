import cv2 
import numpy as np
#import necessary packages
input_image = cv2.imread('Image 6.JPG', cv2.IMREAD_COLOR)#read image
gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)#convert to grayscale
cv2.imshow('Brain with Skull',gray)
kernel = np.ones((3,3), np.uint8)       # set kernel as 3x3 matrix from numpy
#Create erosion and dilation image from the original image

erosion_image = cv2.erode(input_image, kernel, iterations=1)#apply erosion
dilation_image = cv2.dilate(input_image, kernel, iterations=1)#apply dilation

#cv2.imshow('Erosion', erosion_image)
#cv2.imshow('Dilation', dilation_image)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
#apply otsu thresholding
#cv2.imshow('Applying Otsu',thresh)


edges = cv2.Canny(input_image,250,200)#apply edge detection method
cv2.imshow('Canny Edged', edges)


colormask = np.zeros(input_image.shape, dtype=np.uint8)
colormask[thresh!=0] = np.array((0,0,255))
blended = cv2.addWeighted(input_image,0.7,colormask,0.1,0)# blend the image
#cv2.imshow('Blended', blended)


ret, markers = cv2.connectedComponents(thresh)#apply connected components

#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
#Get pixels which correspond to the brain
brain_mask = markers==largest_component

brain_out = input_image.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)
#cv2.imshow('Connected Components',brain_out)

gradient_image = cv2.morphologyEx(input_image, cv2.MORPH_GRADIENT, kernel)#apply gradient
#cv2.imshow('gradient', gradient_image)


brain_mask = np.uint8(brain_mask)
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Closing', closing)

brain_out = input_image.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[closing==False] = (0,0,0)
#cv2.imshow('Connected Components',brain_out)

image = input_image-brain_out

cv2.imshow('result',image)



cv2.waitKey(0)       #wait for a key to exit





