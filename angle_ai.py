import cv2
import numpy as np


img_path = "images/picture3.png"

# Load the image
img = cv2.imread(img_path, 0)

# Threshold the image to get a binary black and white image
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Save the threshholded image
cv2.imwrite('thresh.jpg', thresh)

# Find the contours of the droplet
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Fit a circle to the contours
(x,y),radius = cv2.minEnclosingCircle(contours[0])

# Draw the circle on the image
cv2.circle(img,(int(x),int(y)),int(radius),(0,255,0),2)

# Save the circle drawn image
cv2.imwrite('circle.jpg', img)

# Find the contact angle
angle = np.arctan(radius/((x,y)-contours[0][0][0]))

# Print the contact angle
print(angle)

# Draw the contours on the image
img_contours = cv2.drawContours(img, contours, 0, (0, 255, 0), 3)

# Save the contour drawn image
cv2.imwrite('contours.jpg', img_contours)
