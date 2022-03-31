#!/l/python3.5.2/bin/python3
#
# This is just some messy code to show you how to use the
# ORB feature extractor.
# D. Crandall, Feb 2022

#https://stackoverflow.com/questions/39527947/how-to-calculate-score-from-orb-algorithm
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import os
import sys
from sklearn.cluster import AgglomerativeClustering
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import dendrogram

print(len(sys.argv))
images ={}
k = int(sys.argv[2])
arr = os.listdir(sys.argv[3])
for i in range(len(arr)):
    image1 = Image.open(sys.argv[3]+"\\"+arr[i])
    image1 = np.asarray(image1)
    images[i] = image1


print(arr)
print(len(arr))

number_of_matches_matrix = np.zeros(shape=(len(images), len(images)))
for i in range(len(images)):
    for j in range(i+1, len(images)):
        orb_1 = cv2.ORB_create()
        kp1, des1 = orb_1.detectAndCompute(images[i],None)
        orb_2 = cv2.ORB_create()
        kp2, des2 = orb_2.detectAndCompute(images[j],None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        #matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = bf.knnMatch(des1,des2,k=2)
        #matches = matcher.match(des1,des2)

        good = []
        for first_closest,second_closest in matches:
            if first_closest.distance/second_closest.distance < 0.8 :
                good.append(first_closest)
        """
        final_points={}
        final_descriptors={}
        for match in good:
           query_index=match.queryIdx
           train_index=match.trainIdx
           final_points[kp1[query_index]]=kp2[train_index]

        print('final_points:',final_points)
        print(len(final_points))
        """
        #print(len(good))
        number_of_matches_matrix[i][j] = len(good)
        number_of_matches_matrix[j][i] = number_of_matches_matrix[i][j]

clustering = AgglomerativeClustering(n_clusters=k).fit(number_of_matches_matrix).labels_
print(clustering)
print(len(clustering))
#dendrogram(clustering, above_threshold_color='#bcbddc',orientation='top')

z= zip(arr,clustering)        
new_list=list(z)
#print(new_list)
#print(len(new_list))

res = sorted(new_list, key = lambda x: x[1])
print(res)


