import cv2
import numpy as np
import glob

import scipy.cluster.vq as sc


images = [cv2.imread(file) for file in glob.glob("D:/luddy_SICE/spring 22/Computer Vision/Assignments/sdabhane-sparanjp-athakulk-a2/part1-images/*.jpg")]

print(type(images))
print(len(images))

descriptors = np.array([])
for pic in images:
    kp, des = cv2.ORB_create(nfeatures=1000).detectAndCompute(pic, None)
    descriptors = np.append(descriptors, des)

print(len(descriptors))

desc = np.reshape(descriptors, (len(descriptors)//64,64))
desc = np.float32(desc)

#print(desc)


new_m=cv2.kmeans(desc, K=500,bestLabels=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
print(new_m)