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

#print(len(sys.argv))
images ={}
k = int(sys.argv[2])
arr = os.listdir(sys.argv[3])
for i in range(len(arr)):
    image1 = Image.open(sys.argv[3]+"/"+arr[i])
    image1 = np.asarray(image1)
    images[i] = image1


points_dictionary={}

for i in range(len(arr)):
    for j in range(i+1, len(arr)):
        image1 = sys.argv[3]+"/"+arr[i]
        image2 = sys.argv[3]+"/"+arr[j]
        points_dictionary[(image1,image2)]=None
#print(points_dictionary)
#print(arr)
#print(len(arr))


number_of_matches_matrix = np.zeros(shape=(len(images), len(images)))
for i in range(len(images)):
    for j in range(i+1, len(images)):
        image1 = sys.argv[3]+"/"+arr[i]
        image2 = sys.argv[3]+"/"+arr[j]
        #print(images[i])
        #print(images[j])
        orb_1 = cv2.ORB_create()
        kp1, des1 = orb_1.detectAndCompute(images[i],None)
        orb_2 = cv2.ORB_create()
        kp2, des2 = orb_2.detectAndCompute(images[j],None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(des1,des2,k=2)
        good = []
        for first_closest,second_closest in matches:
            if first_closest.distance/second_closest.distance < 0.75 :
                good.append(first_closest)
        number_of_matches_matrix[i][j] = len(good)
        points_dictionary[(image1,image2)]=good

        image1 = sys.argv[3]+"/"+arr[j]
        image2 = sys.argv[3]+"/"+arr[i]
        orb_3 = cv2.ORB_create()
        kp3, des3 = orb_3.detectAndCompute(images[j],None)
        orb_4 = cv2.ORB_create()
        kp4, des4 = orb_3.detectAndCompute(images[i],None)
        bf1 = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches1 = bf1.knnMatch(des3,des4,k=2)
        good1 = []
        for first_closest,second_closest in matches1:
            if first_closest.distance/second_closest.distance < 0.75 :
                good1.append(first_closest)
        number_of_matches_matrix[j][i] = len(good1)
        points_dictionary[(image1,image2)]=good1


#print(points_dictionary)



clustering = AgglomerativeClustering(n_clusters=k).fit(number_of_matches_matrix).labels_

z= zip(arr,clustering)        
new_list=list(z)


res = sorted(new_list, key = lambda x: x[1])


dictionary_list_1={}
dictionary_list_2={}
for i in range(k):
    dictionary_list_1[i]=[]

for i in range(0,len(res)):
    dictionary_list_1[res[i][1]].append(res[i][0])

for i in range(0,len(res)):
    dictionary_list_2[res[i][0]]=res[i][1]



total_pairs=len(clustering)*(len(clustering)-1)
true_positives=0
true_negatives=0

#print(dictionary_list_2)
count=0
for i in range(0,len(res)):
    for j in range(i+1, len(res)):
        count+=2
        i1=res[i][0]
        i2=res[j][0]
        i_1 = i1.replace("_", "")
        i_2 = i2.replace("_", "")
        im1 = ''.join([i for i in i_1 if not i.isdigit()])
        im2 = ''.join([i for i in i_2 if not i.isdigit()])
        if im1==im2 and dictionary_list_2[i1]==dictionary_list_2[i2]:
            true_positives+=1
        elif im1!=im2 and dictionary_list_2[i1]!=dictionary_list_2[i2]:
            true_negatives+=1

        i3=res[j][0]
        i4=res[i][0]
        i_3 = i3.replace("_", "")
        i_4 = i4.replace("_", "")
        im3 = ''.join([i for i in i_3 if not i.isdigit()])
        im4 = ''.join([i for i in i_4 if not i.isdigit()])
        if im3==im4 and dictionary_list_2[i3]==dictionary_list_2[i4]:
            true_positives+=1
        elif im3!=im4 and dictionary_list_2[i3]!=dictionary_list_2[i4]:
            true_negatives+=1


print(count)

print(true_positives)
print(true_negatives)

#accuracy=(true_positives+true_negatives)/count
accuracy=(true_positives+true_negatives)/total_pairs
print(accuracy)



filename=sys.argv[4]

list_of_cluster_indexes=list(dictionary_list_1.keys())
#print(list_of_cluster_indexes)

for i in range(0,len(list_of_cluster_indexes)):
    line_here=dictionary_list_1[list_of_cluster_indexes[i]]
    with open(filename, 'a') as f:
        for j in range(len(line_here)):
            f.write(line_here[j]+" ")
        f.write("\n")
#    print(line_here)
    #with open(filename, 'a') as f:

