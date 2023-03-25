import numpy as np
import statistics
import cv2
import matplotlib.pyplot as plt

# read in and greyscale image
img = cv2.imread("eye.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# KMeans clustering to differentiate iris
clusters = 2
max_iter = 100  # clustering iterations
epsilon = 0.2  # accuracy
attempts = 10

inner_eye = np.float32(img)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
_, _, (centers) = cv2.kmeans(inner_eye, clusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
clusters = [round(statistics.mean(centers[0])), round(statistics.mean(centers[1]))]

# visualization of clustering result
for i in range(len(inner_eye)):
    for j in range(len(inner_eye[i])):
        diff = 1024
        tmp = 0
        for item in clusters:
            if abs(inner_eye[i][j] - item) < diff:
                diff = abs(inner_eye[i][j] - item)
                tmp = item
        inner_eye[i][j] = tmp

plt.imshow(inner_eye)
plt.show()
