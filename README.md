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
  
![working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_10045.png)
![working_img2](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_1648.png)
![working_img3](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_2458.png)
![working_img4](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_2604.png)
![working_img5](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_7141.png)
![working_img6](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/correctly%20classified%20examples/TEST_ID_7638.png)


  
## Examples where the detection and classification pipeline did not work well:  
  
![not_working_img1](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_1017.png)
![not_working_img2](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_10271.png)
![not_working_img3](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_12285.png)
![not_working_img4](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_2532.png)
![not_working_img5](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_4350.png)
![not_working_img6](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN/blob/master/wrongly%20classified%20examples/TEST_ID_5772.png)



## Improvements that can be made:  
* I did not want to use YOLO for such a simple task, but detection CNN could be improved  
* Augmenting the dataset by shifting the actual bounding boxes for training the detection CNN slighlty improved the accuracy (+5%) 
more augmentation can be exlored  
* Same can be done for classification CNN - but it was not done in this project  

## Download weights and processed datasets from here:
