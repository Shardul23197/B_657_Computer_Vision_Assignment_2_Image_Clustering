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

#print(len(sys.argv))
images ={}
arr=[]
for file in glob.glob(sys.argv[3]):
    images[os.path.basename(file)]=cv2.imread(file)
    arr.append(os.path.basename(file))
k = int(sys.argv[2])

points_dictionary={}

for i in range(len(arr)):
    for j in range(i+1, len(arr)):
        image1 = list(images.keys())[i]
        image2 = list(images.keys())[j]
        points_dictionary[(image1,image2)]=None
#print(points_dictionary)
#print(arr)
#print(len(arr))


number_of_matches_matrix = np.zeros(shape=(len(arr), len(arr)))
for i in range(len(images)):
    for j in range(i+1, len(images)):
        
        orb_1 = cv2.ORB_create()
        kp1, des1 = orb_1.detectAndCompute(list(images.values())[i],None)
        orb_2 = cv2.ORB_create()
        kp2, des2 = orb_2.detectAndCompute(list(images.values())[j],None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(des1,des2,k=2)
        good = []
        for first_closest,second_closest in matches:
            if first_closest.distance/second_closest.distance < 0.75 :
                good.append(first_closest)
        number_of_matches_matrix[i][j] = len(good)
        points_dictionary[(image1,image2)]=good

        orb_3 = cv2.ORB_create()
        kp3, des3 = orb_3.detectAndCompute(list(images.values())[j],None)
        orb_4 = cv2.ORB_create()
        kp4, des4 = orb_3.detectAndCompute(list(images.values())[i],None)
        bf1 = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        matches1 = bf1.knnMatch(des3,des4,k=2)
        good1 = []
        for first_closest,second_closest in matches1:
            if first_closest.distance/second_closest.distance < 0.75 :
                good1.append(first_closest)
        number_of_matches_matrix[j][i] = len(good1)
        points_dictionary[(image1,image2)]=good1


#print(points_dictionary)
#print(number_of_matches_matrix)


clustering = AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='complete').fit(number_of_matches_matrix).labels_

z= zip(arr,clustering)        
new_list=list(z)



res = sorted(new_list, key = lambda x: x[1])
#print(res)

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

#print(dictionary_list_1)
#print(dictionary_list_2)
count=0
for i in range(0,len(res)):
    for j in range(i+1, len(res)):
        count+=2
        i1=res[i][0]
        i2=res[j][0]
        i_1 = i1.replace("_", "")
        i_2 = i2.replace("_", "")
        #print(i_1,i_2)
        im1 = ''.join([i for i in i_1 if not i.isdigit()])
        im2 = ''.join([i for i in i_2 if not i.isdigit()])
        #print(im1,im2)
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
        #print(dictionary_list_2[i3])

        if im3==im4 and dictionary_list_2[i3]==dictionary_list_2[i4]:
            true_positives+=1
        elif im3!=im4 and dictionary_list_2[i3]!=dictionary_list_2[i4]:
            true_negatives+=1


#print(count)
print(total_pairs)

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

