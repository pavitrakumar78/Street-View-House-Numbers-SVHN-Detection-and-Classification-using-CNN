# Street View House Numbers (SVHN) Detection and Classification using CNN

This is my (not very successful) attempt to do both detection and classification of numbers in [SVHN dataset](http://ufldl.stanford.edu/housenumbers/) using 2 CNNs.
  
This project contains 2 parts:  
1. Using CNN to do bounding box regression to find the top, left, width and height of the bounding box
which contains all the digits in a given image  
2. Use the bounding box from step 1 to extract a part of the image with only digits and use a 
another multi-output CNN to classify the digits of the cut image.  
  
My original intension was that this would improve the accuracy compared to the case where we just
feed the entire svhn image into the CNN and let the CNN predict all the digits in the image. But the 
entire pipeline gave me only 51% accuracy where all the digits match exactly and individual 
digit accuracies of 71%, 65%, 84% and 98% for the 1st, 2nd, 3rd and 4th digit respectively (we only consider max of 4-digit prediction).  

## The detection and classification pipeline:   
1. Get input image (so far, this has only been tested on test dataset images of SVHN dataset)  
2. Resize to 64x64, convert to greyscale and normalize the image  
3. Feed processed image into detection CNN to get bounding box  
4. Re-scale bounding box to image's original size  
5. Cut the bounding box alone and resize to 64x64  
6. Feed the image we just cut and resized to the classification CNN to get digits  
7. Convert CNN predictions into an understandable format  
8. Output digits  

## Examples where the detection and classification pipeline worked well:  
The bounding boxes in the images below are coordinates predicted by the detection CNN and the number prediction is done by the classification CNN.  
  
| Image  | Predicted value | Actual value |
| ------------- | ------------- | ------------- |
| ![working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_10045.png)   | 1522  | 1502  |
| ![working_img2](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_1648.png)   | 135  | 135 |
| ![working_img3](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_2458.png)   | 861 | 861 |
| ![working_img4](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_2604.png)  | 348 | 348 |
| ![working_img5](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_7141.png)   | 114 | 114 |
| ![working_img6](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_7638.png)   | 23 | 23 |
  
  
## Examples where the detection and classification pipeline did not work well:  
The bounding boxes in the images below are coordinates predicted by the detection CNN and the number prediction is done by the classification CNN.  

| Image  | Predicted value | Actual value |
| ------------- | ------------- | ------------- |
| ![not_working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_1017.png)  | 32 | 863 |
| ![not_working_img2](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_10271.png)   | 6 | 7 |
| ![not_working_img3](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_12285.png)   | 8 | 26 |
| ![not_working_img4](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_2532.png)  | 1 | 184 |
| ![not_working_img5](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_4350.png)   | 1410 | 44 |
| ![not_working_img6](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_5772.png)   | 27 | 6 |

## Improvements that can be made:  
* I did not want to use YOLO for such a simple task, but detection CNN could be improved  
* Augmenting the dataset by shifting the actual bounding boxes for training the detection CNN slighlty improved the accuracy (+5%) 
more augmentation can be exlored  
* Same can be done for classification CNN - but it was not done in this project  

## Project files:
`construct_datasets.py`  
Uses the images downloaded from [SVHN dataset website](http://ufldl.stanford.edu/housenumbers/)  website along with the .mat files describing the bounding box to build a single table for each test and train for easy use in other files. If you don't want to run this file, download it .h5 files from the google drive link below.
  
`train_digit_classification.py`  
Uses the processed .h5 files in data folder to train a classification CNN.  
  
`train_digit_detection.py`  
Uses the processed .h5 files in data folder to train a detection CNN.  
  
`combi_models.py`  
After training both networks, this file uses both networks to implement all the steps described in the pipeline section above.  
  
## Download weights and processed datasets from here:
Weights for both CNNs and .h5 files for train and test datasets are available in the link below:   
  
CNN Weights: https://drive.google.com/open?id=1vv7vzqzGjjUqjcCZYeX_NaGrqSU1Ami2  
Dataset: https://drive.google.com/open?id=1KfVqQHjimQnXdzsCtQurwmTSpMe2mmA7  
   
## Environment
Python 3.5  
All code was run on Amazon EC2 Deep Learning AMI version 7 (ami-139a476c)  
I also tested this on my local Windows 10 PC with the following libraries:  
* Numpy 1.13.1  
* Keras 2.0.5  
* Pandas 0.20.3  
* OpenCV 3.2.0  
* TensorFlow 1.2.1 (with GPU support)  
