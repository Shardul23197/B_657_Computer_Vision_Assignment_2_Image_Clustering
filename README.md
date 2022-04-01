# Part 1: Image matching and clustering

## Limitations

# Part 2: Image transformations

### Image Warping:
Image warping means that the image is transformed using a 3x3 Transformation matrix.
We have written a function warp() which takes an image which is to be transformed and the transformation matrix as its arguments.
We have used bilinear interpolation along with inverse warping as specified in the assignemnt. Using inverse warping aviods holes in the warped image and bilinear transformation helps smoothen the image.
To write the code for bilinear transformation, I watched the following video https://youtu.be/UhGEtSdBwIQ  and followed its steps.
To test this function we tested it on the given "lincoln.jpg" image to confirm if warp() worked correctly.

<br> <br>
### Finding the transformation matrix from given correspondenses:

The results for transformation on the book image from the assignemnt are:<br>

**1) Translation n=1:** <br>
![Alt text](https://media.github.iu.edu/user/18152/files/ad69aaa2-04ff-4ad2-b01c-614485789c21)


**2) Euclidean n=2:**<br>
![Alt text](https://media.github.iu.edu/user/18152/files/a2dc1e72-ccb2-4f40-b8c0-78a64123c766)

**1) Affine n=3:**<br>
![Alt text](https://media.github.iu.edu/user/18152/files/dd09c93b-79ab-458e-84f5-a94946e1c17d)

**1) Projective n=4:**<br>
![Alt text](https://media.github.iu.edu/user/18152/files/9088ca4c-7bf0-4d6e-82fb-bbc046a0bfcf)


We also tested the transformations on the building image which was in the assignment named as scene1 and scene2. <br>
![Alt text](https://media.github.iu.edu/user/18152/files/ebd243e5-da8f-4852-9a7a-111bd5aef0ce)

The results from those images are as follows.<br>
We tested both types of transformations which are as follows:
The corresponding point matches of the above images were manually found out using Paint.<br>
1) Transforming scene2 as per scene1:<br>
python a2.py part2 4 scene2.jpg scene1.jpg scene_output1.jpg 476,243 220,328 449,246 192,332 700,73 442,158 671,363 414,449
![Alt text](https://media.github.iu.edu/user/18152/files/c21a4a3c-67ed-43a7-898f-04e5e0652c52)

2) Transforming scene1 as per scene2:<br>
python a2.py part2 4 scene1.jpg scene2.jpg scene_output2.jpg 220,328 476,243 192,332 449,246 442,158 700,73 414,449 671,363
![Alt text](https://media.github.iu.edu/user/18152/files/6091d3f2-6fa9-4b54-a69f-0b63ce857193)



## Limitations:
The code fails to tranform the image correctly if the corresponding points have errors.
We tried to find corresponding points manually on a high resolution image but there are some manual errors which are introduced. This results in a poorly transformed image.
The following is one such example.
python a2.py part2 4 src.jpg dest.jpg bhutan.jpg 167,801 617,1133 723,1693 1057,1930 725,2075 1019,2391 335,141 843,437
![Alt text](https://media.github.iu.edu/user/18152/files/cb85d8e9-6060-4452-8901-d6d2b07aead4)



# Part 3: Automatic image matching and transformations

## Limitations
