# Part 1: Image matching and clustering

## Limitations

# Part 2: Image transformations

### Image Warping:
Image warping means that the image is transformed using a 3x3 Transformation matrix.
We have written a function warp() which takes an image which is to be transformed and the transformation matrix as its arguments.
We have used bilinear interpolation along with inverse warping as specified in the assignemnt. Using inverse warping aviods holes in the warped image and bilinear transformation helps smoothen the image.
To test this function we tested it on the given "lincoln.jpg" image to confirm if warp() worked correctly.
![Alt text](https://media.github.iu.edu/user/18152/files/b02fb0d6-c001-4948-97ed-53514b6f6a03)

### Finding the transformation matrix from given correspondenses:


Runing on book images:
n=1
![Alt text](https://media.github.iu.edu/user/18152/files/ad69aaa2-04ff-4ad2-b01c-614485789c21)

n=2
![Alt text](https://media.github.iu.edu/user/18152/files/a2dc1e72-ccb2-4f40-b8c0-78a64123c766)

n=3
![Alt text](https://media.github.iu.edu/user/18152/files/dd09c93b-79ab-458e-84f5-a94946e1c17d)

n=4
![Alt text](https://media.github.iu.edu/user/18152/files/9088ca4c-7bf0-4d6e-82fb-bbc046a0bfcf)

scene1.jpg
![Alt text](https://media.github.iu.edu/user/18152/files/95536bf5-f23c-47d5-9572-31b5f421daa1)

scene2.jpg
![Alt text](https://media.github.iu.edu/user/18152/files/e8f74ec1-531b-4de8-8a44-d7e2576c684e)

python a2.py part2 4 scene2.jpg scene1.jpg scene_output_projective2.jpg 476,243 220,328 449,246 192,332 700,73 442,158 671,363 414,449
![Alt text](https://media.github.iu.edu/user/18152/files/c21a4a3c-67ed-43a7-898f-04e5e0652c52)

python a2.py part2 4 scene1.jpg scene2.jpg scene_output_projective3.jpg 220,328 476,243 192,332 449,246 442,158 700,73 414,449 671,363
![Alt text](https://media.github.iu.edu/user/18152/files/6091d3f2-6fa9-4b54-a69f-0b63ce857193)



## Limitations


# Part 3: Automatic image matching and transformations

## Limitations
