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


images ={} 

for file in glob.glob("D:/luddy_SICE/spring 22/Computer Vision/Assignments/sdabhane-sparanjp-athakulk-a2/part1-images/*.jpg"):
   images[str(os.path.basename(file))]=cv2.imread(file)

# matches = []
# for i in images:
#    for j in images:
#       matches[i] = j
#       break
#    break


#img = cv2.imread("D:/luddy_SICE/spring 22/Computer Vision/Assignments/sdabhane-sparanjp-athakulk-a2/part1-images/eiffel_18.jpg")
#img1 = cv2.imread("D:/luddy_SICE/spring 22/Computer Vision/Assignments/sdabhane-sparanjp-athakulk-a2/part1-images/eiffel_19.jpg")

# print(len(images))
# print(images)
#print(img1.shape)

#for name,img in images.keys(),images.values():
#   print(name,img)

# you can increase nfeatures to adjust how many features to detect 
#orb = cv2.ORB_create()

# detect features 
#(keypoints, descriptors) = orb.detectAndCompute(img, None)
# print(len(keypoints))
# print(len(descriptors))
#(keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
# print(len(keypoints1))
# print(len(descriptors1))
matches=[]
for i in images:
   for j in images:
      if i==j:
         pass
      else: 
         orb_1 = cv2.ORB_create()
         kp1, des1 = orb_1.detectAndCompute(images[i],None)
         kp2, des2 = orb_1.detectAndCompute(images[j],None)
         bf = cv2.BFMatcher()
         matches = bf.knnMatch(des1,des2,k=2)
         good = []
         for m,n in matches:
            if m.distance/n.distance<0.75:
               good.append(m)
         final_points={}
         final_descriptors={}
         for match in good:
            query_index=match.queryIdx
            train_index=match.trainIdx
            final_points[kp1[query_index]]=kp2[train_index]
            #final_descriptors[des1[query_index]]=des2[train_index]

         print('final_points:',final_points)
         print(len(final_points))


         matches[i][j]=matches[j][i]
         
print('hello')
print(len(matches))

   #print('final_descriptors:',final_descriptors)

# if len(keypoints)>=len(keypoints1):
#    range_value=len(keypoints1)
# else:
#    range_value=len(keypoints1)

# dictionary_match={}
# dictionary_match_1={}
# dictionary_match_2={}
# dictionary_match_3={}

# for i in range(0, len(keypoints)):
#    dictionary_match[i]=[]

# for i in range(0, len(keypoints)):
#    dictionary_match_1[i]={}

# for i in range(0, len(keypoints)):
#    dictionary_match_2[i]={}


# for i in range(0, len(keypoints)):
#    for j in range(0,len(keypoints1)):
#       ed_here=cv2.norm( descriptors[i], descriptors1[j], cv2.NORM_L2)
#       dictionary_match_1[i][j]=ed_here
      



# for i in range(0, len(keypoints)):
#    ed_values=dictionary_match_1[i].values()
#    ed_values=list(ed_values)
#    min_1=min(ed_values)
#    index_of_min_1=ed_values.index(min_1)
#    dictionary_match[i].append(min_1)
#    dictionary_match_2[i][index_of_min_1]=min_1
#    ed_values.remove(min_1)
#    min_2=min(ed_values)
#    index_of_min_2=ed_values.index(min_2)
#    dictionary_match[i].append(min_2)
#    dictionary_match_2[i][index_of_min_2]=min_2

# final_matches=[]
# final_matches_count=0
# threshold=0.80
# indexes_list=[]
# #print(dictionary_match)
# for i in range(0, len(keypoints)):
#    ratio=dictionary_match[i][0]/dictionary_match[i][1]
#    if ratio<threshold:
#       indexes_list.append(i)
#       final_matches_count+=1
#       dictionary_match_3[i]=min(dictionary_match_2[i], key=dictionary_match_2[i].get)
#       #dictionary_match_3[i]=dictionary_match_2[i]
#       #dictionary_match_2[i]=dictionary_match[i][0]

# #print(len(indexes_list))
# print(dictionary_match_3)
# print(final_matches_count)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(descriptors,descriptors1,k=2)
# print(len(matches))
# #BFMatcher.knnMatch()
# #matches = bf.match(descriptors,descriptors1)
# #print(len(matches))
# # good_matches=[]
# # print(len(indexes_list))
# # for i in range(0,len(indexes_list)):
# #   good_matches.append(matches[indexes_list[i]])
# good = [[m] for m, n in matches if m.distance < 0.75*n.distance]
# print(len(good))
# img3 = cv2.drawMatchesKnn(img,keypoints,img1,keypoints1,good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imwrite("Finalresult4.jpg", img3)
#print(dictionary_match_2[2])
#print(final_matches_count)

# print(descriptors[2])
# print(descriptors1[8])

#descriptor_result=[]
#descriptor_result=np.ndarray(descriptor_result)
# for i in indexes_list:
#    #print(len(descriptors[i]))
#    #np.append(descriptor_result, descriptors[i])
#    descriptor_result.append(descriptors[i])


#descriptor_result=np.ndarray(descriptor_result)
#print(descriptor_result.shape)
#print(len(descriptors[2]))
#print(len(descriptor_result))
#print(dictionary_match_2)
# for i in indexes_list:
#    print(descriptors[i])

# matcher = cv2.BFMatcher()
# matches = matcher.match(descriptors,descriptors1)

#final_img = cv2.drawMatches(img, keypoints, img1, keypoints1, matches,None)
   #print(ratio)
# ed_here_1=cv2.norm( descriptors[0], descriptors1[0], cv2.NORM_L2)
# print(ed_here_1)


# for i in range(0, len(keypoints)):
#    min_ed=100000000
#    min_ed_index=-100
#    for j in range(0,len(keypoints1)):
#       ed_here=cv2.norm( descriptors[i], descriptors1[j], cv2.NORM_L2)
#       if ed_here<min_ed:
#          min_ed=ed_here
#          min_ed_index=j
#    #dictionary_match[i]=min_ed_index
#    dictionary_match[i].append(min_ed_index)

# print(dictionary_match)


# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(descriptors,descriptors1)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv2.drawMatches(img,keypoints,img1,keypoints1,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# # print("D1",dictionary_match)
# # print("D2",dictionary_match_1)
# #print(type(x))
# # for i in range(0, len(keypoints)):
# #    min_ed=100000000
# #    min_ed_index=-100
# #    for j in range(0,len(keypoints1)):
# #       ed_here=cv2.norm( descriptors[i], descriptors[j], cv2.NORM_L2)
# #       if ed_here<min_ed:
# #          min_ed=ed_here
# #          min_ed_index=j
# #    dictionary_match[i]=min_ed_index
# #print(dictionary_match)
#       #print(cv2.norm( descriptors[i], descriptors[j], cv2.NORM_L2))
#       #print("Keypoint #%d: x=%d, y=%d, descriptor=%s, distance between this descriptor and descriptor #0 is %d" % (i, keypoints[i].pt[0], keypoints[i].pt[1], np.array2string(descriptors[i]), cv2.norm( descriptors[0], descriptors[i], cv2.NORM_HAMMING)))


# # matcher = cv2.BFMatcher()
# # matches = matcher.match(descriptors,descriptors1)

# #print(matches)
# #put a little X on each feature
# # for i in range(0, len(keypoints)):
# #    #print(i)
# #    #print("Keypoint #%d: x=%d, y=%d, descriptor=%s, distance between this descriptor and descriptor #0 is %d" % (i, keypoints[i].pt[0], keypoints[i].pt[1], np.array2string(descriptors[i]), cv2.norm( descriptors[0], descriptors[i], cv2.NORM_HAMMING)))
# #    for j in range(-5, 5):
# #       img[int(keypoints[i].pt[1])+j, int(keypoints[i].pt[0])+j] = 0 
# #       img[int(keypoints[i].pt[1])-j, int(keypoints[i].pt[0])+j] = 255 

# # cv2.imwrite("lincoln-orb.jpg", img)
