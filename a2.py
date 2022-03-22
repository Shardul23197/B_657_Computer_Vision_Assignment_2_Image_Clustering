import numpy as np
from PIL import Image, ImageOps
from numpy.linalg import inv
import sys

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
        j = [img2_p1, img2_p2]
        # k = np.array(img1_p1)
        k = [img1_p1,img1_p2]
        # print(j)
        # print(k)
        res1 = np.linalg.solve(j, k)

        print(res1)
        j = [img2_p1, img2_p2]
        k = np.array(img1_p2)
        res2 = np.linalg.solve(j, k)

        # matrix = np.array([[res1[0], res1[1], 0], [res2[0], res2[1], 0], [0, 0, 1]])
        matrix = np.array([[res1[0][0],res1[0][1],0],[res1[1][0],res1[1][1],0],[0, 0, 1]])
        print("Transformation matrix\n",matrix)

        inverse_tm = inv(matrix)
        transformed_image = Image.fromarray(warp(image2, inverse_tm))
        transformed_image.save(output_image)
    if n == 3:
        j = [img2_p1[0], img2_p1[1], 1]
        k = [img2_p2[0], img2_p2[1], 1]
        l = [img2_p3[0], img2_p3[1], 1]
        c = np.array([img1_p1[0], img1_p1[1], 1])
        # print(j)
        clubed = np.array((j, k, l))
        res1 = np.linalg.solve(clubed, c)

        j = [img2_p1[0], img2_p1[1], 1]
        k = [img2_p2[0], img2_p2[1], 1]
        l = [img2_p3[0], img2_p3[1], 1]
        c = np.array([img1_p2[0], img1_p2[1], 1])
        # print(j)
        clubed = np.array((j, k, l))
        res2 = np.linalg.solve(clubed, c)
        # print(res1)
        # print(res2)
        matrix = np.array([res1, res2, [0, 0, 1]])
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









