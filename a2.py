import numpy as np
from PIL import Image, ImageOps
from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt
from copy import deepcopy as deepcopy
import cv2
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import os
import glob

def part1_function(images,arr,k):
    points_dictionary={}
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            image1 = list(images.keys())[i]
            image2 = list(images.keys())[j]
            points_dictionary[(image1,image2)]=None

    number_of_matches_matrix = np.zeros(shape=(len(images), len(images)))
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
                if first_closest.distance/second_closest.distance < 0.9 :
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
                if first_closest.distance/second_closest.distance < 0.9 :
                    good1.append(first_closest)
            number_of_matches_matrix[j][i] = len(good1)
            points_dictionary[(image1,image2)]=good1


    clustering = AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='complete').fit(number_of_matches_matrix).labels_
    z= zip(arr,clustering)
    new_list=list(z)


    clustering = AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='complete').fit(number_of_matches_matrix).labels_

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



def orb_descriptor(image1,image2):
    #https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    orb_1 = cv2.ORB_create()
    kp1, des1 = orb_1.detectAndCompute(image1,None)
    kp2, des2 = orb_1.detectAndCompute(image2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1,des2,k=2)
    print("---------------------------------------------",len(matches))
    good = []
    for first_closest,second_closest in matches:
        #print(first_closest.distance,first_closest)
        #print(second_closest.distance,second_closest)
        if first_closest.distance/second_closest.distance < 0.9 :
            good.append(first_closest)
    img3 = cv2.drawMatches(image1,kp1,image2,kp2,good,None)
    cv2.imwrite('image.jpg',img3)

    image1_points = []
    image2_points = []
    for match in good:
        query_index=match.queryIdx
        train_index=match.trainIdx
        #image_index = match.imgIdx
        #final_points[kp1[query_index].pt]=kp2[train_index].pt
        image1_points.append(kp1[query_index].pt)
        image2_points.append(kp2[train_index].pt)
    return image1_points, image2_points

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

def transform(n,img1_p1,img1_p2,img1_p3,img1_p4,img2_p1,img2_p2,img2_p3 ,img2_p4):
    # n = 1
    if n==1:
        matrix = np.array([[1, 0, abs(img1_p1[0] - img2_p1[0])], [0, 1, abs(img1_p1[1] - img2_p1[1])], [0, 0, 1]])
        print("Transformation matrix\n",matrix)
        return matrix
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

        return matrix

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

        return matrix

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

        return matrix



if __name__=="__main__":
    part_number = sys.argv[1]
    if part_number == "part1":
        print("starting Part 1:")
        images ={}
        arr=[]
        for file in glob.glob(sys.argv[3]):
            images[os.path.basename(file)]=cv2.imread(file)
            arr.append(os.path.basename(file))
        k = int(sys.argv[2])
        part1_function(images,arr,k)


    if part_number == "part2":
        #part2 4 part2-images\book2.jpg part2-images\book1.jpg try.jpg 141,131 318,256 480,159 534,372 493,630 316,670 64,601 73,473
        print("Starting Part 2:")
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

        matrix = transform(n,img1_p1,img1_p2,img1_p3,img1_p4,img2_p1,img2_p2,img2_p3 ,img2_p4)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image2, inverse_tm))
        transformed_image.save(output_image)


    if part_number == "part3":
        print("Starting Part 3:")

        # part3 scene1.jpg scene2.jpg
        image_name1 = sys.argv[2]
        image_name2 = sys.argv[3]

        image1 = Image.open(image_name1)
        image1 = np.asarray(image1)

        image2 = Image.open(image_name2)
        image2 = np.asarray(image2)
        # call the orb descriptor function to get the orb descriptors and point correspodences.

        points_image1, points_image2 = orb_descriptor(image1,image2)
        #print("final_points",points_image1,points_image2)
        print("length of final_points",len(points_image2))
        s = 4 # number of samples or point correspondences for homography.
        max_inliers = 0
        best_points = []
        best_transformation = None
        for i in range(2000):
            inliers = 0
            sample_indices = list(np.random.randint(0,len(points_image1),s))
            #print(sample_indices)
            #print(points_image1[sample_indices[0]])
            img1_p1,img1_p2,img1_p3,img1_p4 = [points_image1[i] for i in sample_indices]
            img2_p1,img2_p2,img2_p3 ,img2_p4 = [points_image2[i] for i in sample_indices]

            try:
            #find the projective trasformation
                matrix = transform(1,img1_p1,img1_p2,img1_p3,img1_p4,img2_p1,img2_p2,img2_p3 ,img2_p4)
            #print(matrix)
            except:
                print("singular matrix. \n------ skipping-------\n")
                continue

            #finding the number of inliers for the above projective transformation
            for i in range(len(points_image1)):
                pt1  = np.array([points_image2[i][1],points_image2[i][0],1])
                pt_ = np.dot(matrix,pt1)
                pt_ = pt_/pt_[2]

                pt2 = np.array([points_image1[i][1],points_image1[i][0],1])
                #print("pt_",pt_)
                #print("pt2",pt2)
                #find euclidean distance between the transformed point and the actual point from the descriptor
                if np.sqrt(np.sum((pt_-pt2)**2)) < 15:
                    inliers += 1
            print(inliers)
            if inliers > max_inliers:
                print("------here------!!!!!!!!!!!!!!!")
                best_transformation = deepcopy(matrix)
                max_inliers = inliers



        print(max_inliers)
        print(best_transformation)
        inverse_tm = inv(best_transformation)
        transformed_image = Image.fromarray(warp(image1, inverse_tm))
        transformed_image.save("lets-see.jpg")