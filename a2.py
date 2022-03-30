import numpy as np
from PIL import Image, ImageOps
from numpy.linalg import inv
import sys
import cv2
import os
import sys
from sklearn.cluster import AgglomerativeClustering
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import dendrogram


def part1_function():
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

    number_of_matches_matrix = np.zeros(shape=(len(images), len(images)))
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            image1 = sys.argv[3]+"/"+arr[i]
            image2 = sys.argv[3]+"/"+arr[j]
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

    for i in range(0,len(res)):
        for j in range(i+1, len(res)):
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


    print(true_positives)
    print(true_negatives)

    accuracy=(true_positives+true_negatives)/total_pairs
    print(accuracy)
    filename=sys.argv[4]

    list_of_cluster_indexes=list(dictionary_list_1.keys())

    for i in range(0,len(list_of_cluster_indexes)):
        line_here=dictionary_list_1[list_of_cluster_indexes[i]]
        print(line_here)
        with open(filename, 'a') as f:
            for j in range(len(line_here)):
                f.write(line_here[j]+" ")
            f.write("\n")






















# def orb_descriptor(image1,image2):
#     orb = cv2.ORB_create()
#     (keypoints, descriptors) = orb.detectAndCompute(image1, None)
#     (keypoints1, descriptors1) = orb.detectAndCompute(image2, None)
#     dictionary_match={}
#     dictionary_match_1={}
#     dictionary_match_2={}
#     dictionary_match_3={}
#     for i in range(0, len(keypoints)):
#         dictionary_match[i]=[]

#     for i in range(0, len(keypoints)):
#         dictionary_match_1[i]={}

#     for i in range(0, len(keypoints)):
#         dictionary_match_2[i]={}


#     for i in range(0, len(keypoints)):
#         for j in range(0,len(keypoints1)):
#             ed_here=cv2.norm( descriptors[i], descriptors1[j], cv2.NORM_L2)
#             dictionary_match_1[i][j]=ed_here
      



#     for i in range(0, len(keypoints)):
#         ed_values=dictionary_match_1[i].values()
#         ed_values=list(ed_values)
#         min_1=min(ed_values)
#         index_of_min_1=ed_values.index(min_1)
#         dictionary_match[i].append(min_1)
#         dictionary_match_2[i][index_of_min_1]=min_1
#         ed_values.remove(min_1)
#         min_2=min(ed_values)
#         index_of_min_2=ed_values.index(min_2)
#         dictionary_match[i].append(min_2)
#         dictionary_match_2[i][index_of_min_2]=min_2

#     final_matches=[]
#     final_matches_count=0
#     threshold=0.80
#     indexes_list=[]
#     #print(dictionary_match)
#     for i in range(0, len(keypoints)):
#         ratio=dictionary_match[i][0]/dictionary_match[i][1]
#         if ratio<threshold:
#             indexes_list.append(i)
#             final_matches_count+=1
#             dictionary_match_3[i]=min(dictionary_match_2[i], key=dictionary_match_2[i].get)
#       #dictionary_match_3[i]=dictionary_match_2[i]
#       #dictionary_match_2[i]=dictionary_match[i][0]

# #print(len(indexes_list))
#     print(dictionary_match_3)


#     return dictionary_match_3

# def orb_descriptor_1(image1,image2):
#     orb_1 = cv2.ORB_create()
#     kp1, des1 = orb_1.detectAndCompute(image1,None)
#     kp2, des2 = orb_1.detectAndCompute(image2,None)
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1,des2,k=2)
#     good = []
#     for m,n in matches:
#         if m.distance/n.distance<0.80:
#             good.append(m)
#     final_points={}
#     final_descriptors={}
#     for match in good:
#         query_index=match.queryIdx
#         train_index=match.trainIdx
#         final_points[kp1[query_index]]=kp2[train_index]
#         final_descriptors[des1[query_index]]=des2[train_index]
    
#     return final_points,final_descriptors

    #print(good)

def warp(image_array, inverse_tm):
    dest_image = np.zeros(image_array.shape)
    for col in range(len(image_array[0])):
        for row in range(len(image_array)):
            source_pixels = np.dot(inverse_tm, [col, row, 1])
            if (source_pixels[0] > 0 and source_pixels[1] > 0 and source_pixels[2] > 0) or (
                    source_pixels[0] < 0 and source_pixels[1] < 0 and source_pixels[2] < 0):
                source_x = source_pixels[1] / source_pixels[2]
                source_y = source_pixels[0] / source_pixels[2]

                x_ = int(np.floor(source_x))
                y_ = int(np.floor(source_y))
                # print(x_,y_)
                a = source_x - x_
                b = source_y - y_

                rgb = [0, 0, 0]
                for i in range(3):
                    # print(i)
                    if x_ + 1 < len(image_array) and y_ + 1 < len(image_array[0]):
                        value = (1 - a) * (1 - b) * image_array[x_][y_][i] + (1 - a) * (b) * image_array[x_][y_ + 1][
                            i] + (a) * (1 - b) * image_array[x_ + 1][y_][i] + (a) * (b) * image_array[x_ + 1][y_ + 1][i]
                        rgb[i] = int(value)

                dest_image[row][col] = rgb

    dest_image = dest_image.astype(np.uint8)
    return dest_image

def transform(n,image2,image1,output_image,img1_p1,img1_p2,img1_p3,img1_p4,img2_p1,img2_p2,img2_p3 ,img2_p4):
    # n = 1
    if n==1:
        matrix = np.array([[1, 0, abs(img1_p1[0] - img2_p1[0])], [0, 1, abs(img1_p1[1] - img2_p1[1])], [0, 0, 1]])
        print("Transformation matrix\n",matrix)
        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image2, inverse_tm))
        transformed_image.save(output_image)
    if n == 2:
        # j = [img2_p1, img2_p2]
        # k = np.array(img1_p1)
        # # print(j)
        # # print(k)
        # res1 = np.linalg.solve(j, k)
        #
        # j = [img2_p1, img2_p2]
        # k = np.array(img1_p2)
        # res2 = np.linalg.solve(j, k)
        matrix = np.array([
            [img2_p1[0], -img2_p1[1], 1, 0],
            [img2_p1[1], img2_p1[0], 0, 1],
            [img2_p2[0], -img2_p2[1], 1, 0],
            [img2_p2[1], img2_p2[0], 0, 1]
        ])

        mat_b = np.array(
            [[img1_p1[0]], [img1_p1[1]], [img1_p2[0]], [img1_p2[1]]])
        x = np.linalg.solve(matrix, mat_b)
        print("x",x)
        # x = np.append(x, [0, 0, 1])
        matrix = np.array([
            [x[0],-x[1],x[2]],
            [x[1],x[0],x[3]],
            [0,0,1]
        ],dtype=float)
        print("Transformation matrix\n",matrix)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image2, inverse_tm))
        transformed_image.save(output_image)

    if n == 3:

        matrix = np.array([
            [img2_p1[0], img2_p1[1], 1, 0, 0, 0],
            [0, 0, 0, img2_p1[0], img2_p1[1], 1],
            [img2_p2[0], img2_p2[1], 1, 0, 0, 0],
            [0, 0, 0, img2_p2[0], img2_p2[1], 1],
            [img2_p3[0], img2_p3[1], 1, 0, 0, 0],
            [0, 0, 0, img2_p3[0], img2_p3[1], 1]])

        mat_b = np.array(
            [[img1_p1[0]], [img1_p1[1]], [img1_p2[0]], [img1_p2[1]], [img1_p3[0]], [img1_p3[1]]])
        x = np.linalg.solve(matrix, mat_b)
        x = np.append(x, [0,0,1])
        # print(x)
        # print(x.reshape((3, 3)))
        matrix = x.reshape((3, 3))
        # matrix = np.array([res1, res2, [0, 0, 1]])
        print("Transformation matrix\n",matrix)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image2, inverse_tm))
        transformed_image.save(output_image)
    if n == 4:
        matrix = np.array([
            [img2_p1[0], img2_p1[1], 1, 0, 0, 0, -img1_p1[0] * img2_p1[0], -img1_p1[0] * img2_p1[1]],
            [0, 0, 0, img2_p1[0], img2_p1[1], 1, -img1_p1[1] * img2_p1[0], -img1_p1[1] * img2_p1[1]],
            [img2_p2[0], img2_p2[1], 1, 0, 0, 0, -img1_p2[0] * img2_p2[0], -img1_p2[0] * img2_p2[1]],
            [0, 0, 0, img2_p2[0], img2_p2[1], 1, -img1_p2[1] * img2_p2[0], -img1_p2[1] * img2_p2[1]],
            [img2_p3[0], img2_p3[1], 1, 0, 0, 0, -img1_p3[0] * img2_p3[0], -img1_p3[0] * img2_p3[1]],
            [0, 0, 0, img2_p3[0], img2_p3[1], 1, -img1_p3[1] * img2_p3[0], -img1_p3[1] * img2_p3[1]],
            [img2_p4[0], img2_p4[1], 1, 0, 0, 0, -img1_p4[0] * img2_p4[0], -img1_p4[0] * img2_p4[1]],
            [0, 0, 0, img2_p4[0], img2_p4[1], 1, -img1_p4[1] * img2_p4[0], -img1_p4[1] * img2_p4[1]]])

        mat_b = np.array(
            [[img1_p1[0]], [img1_p1[1]], [img1_p2[0]], [img1_p2[1]], [img1_p3[0]], [img1_p3[1]], [img1_p4[0]],
             [img1_p4[1]]])

        x = np.linalg.solve(matrix, mat_b)
        x = np.append(x, [1])
        # print(x.reshape((3, 3)))
        matrix = x.reshape((3, 3))
        print("Transformation matrix\n",matrix)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image2, inverse_tm))
        transformed_image.save(output_image)

if __name__=="__main__":
    part_number = sys.argv[1]
    if part_number == "part1":
        print("Hello")
        part1_function()

    if part_number == "part2":
        n = int(sys.argv[2])
        image_name2 = sys.argv[3]
        image_name1 = sys.argv[4]

        image1 = Image.open(image_name1)
        image1 = np.asarray(image1)

        image2 = Image.open(image_name2)
        image2 = np.asarray(image2)

        output_image = sys.argv[5]
        img2_p1 = np.array([ int(i) for i in sys.argv[6].split(",")])
        img1_p1 = np.array([ int(i) for i in sys.argv[7].split(",")])

        img2_p2 = np.array([ int(i) for i in sys.argv[8].split(",")])
        img1_p2 = np.array([ int(i) for i in sys.argv[9].split(",")])

        img2_p3 = np.array([ int(i) for i in sys.argv[10].split(",")])
        img1_p3 = np.array([ int(i) for i in sys.argv[11].split(",")])

        img2_p4 = np.array([ int(i) for i in sys.argv[12].split(",")])
        img1_p4 = np.array([ int(i) for i in sys.argv[13].split(",")])

        transform(n,image2,image1,output_image,img1_p1,img1_p2,img1_p3,img1_p4,img2_p1,img2_p2,img2_p3 ,img2_p4)









