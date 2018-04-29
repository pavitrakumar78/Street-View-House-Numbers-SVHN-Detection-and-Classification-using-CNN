# Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN

This is my (not very successful) attempt to do both detection and classification of numbers in SVHN
dataset using 2 CNNs.
  
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
  
| Image  | Predicted and Acutal value |
| ------------- | ------------- |
| ![working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_10045.png)   | Predicted: 1522  
Actual: 1502    |
| Content Cell  | Content Cell  |
  
![working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_10045.png)  
Predicted: 1522  
Actual: 1502  
![working_img2](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_1648.png)  
Predicted: 135  
Actual: 135  
![working_img3](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_2458.png)  
Predicted: 861  
Actual: 861  
![working_img4](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_2604.png)  
Predicted: 348  
Actual:  348  
![working_img5](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_7141.png)  
Predicted: 114  
Actual: 23  
![working_img6](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_7638.png)  
Predicted:  
Actual:  

  
## Examples where the detection and classification pipeline did not work well:  
  
![not_working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_1017.png)  
Predicted: 32  
Actual: 863  
![not_working_img2](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_10271.png)  
Predicted: 6  
Actual: 7  
![not_working_img3](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_12285.png)  
Predicted: 8  
Actual: 26  
![not_working_img4](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_2532.png)  
Predicted: 1  
Actual: 184  
![not_working_img5](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_4350.png)  
Predicted: 1410  
Actual:44  
![not_working_img6](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_5772.png)  
Predicted: 27  
Actual: 6  


## Improvements that can be made:  
* I did not want to use YOLO for such a simple task, but detection CNN could be improved  
* Augmenting the dataset by shifting the actual bounding boxes for training the detection CNN slighlty improved the accuracy (+5%) 
more augmentation can be exlored  
* Same can be done for classification CNN - but it was not done in this project  

## Download weights and processed datasets from here:
  
  
## Environment
Python 3.5  
All code was run on Amazon EC2 Deep Learning AMI version 7 (ami-139a476c)  
I also tested this on my local Windows 10 PC with the following libraries:  
Numpy 1.13.1  
Keras 2.0.5  
Pandas 0.20.3  
OpenCV 3.2.0  
TensorFlow 1.2.1 (with GPU support)  
