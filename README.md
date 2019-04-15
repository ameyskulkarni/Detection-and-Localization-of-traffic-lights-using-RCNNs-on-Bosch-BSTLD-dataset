# Detection-and-Localization-of-traffic-lights-using-RCNNs-on-Bosch-BSTLD-dataset
This repository presents a code to detect the traffic lights using RCNNs. The dataset consists of road images with traffic lights showing different signals. The labels are given in a .YAML file. A separate script is written to convert this .yaml file into a desired .csv format. Each row of the new .csv file consists of name of the image, details about coordinates of the bounding box(x_min, x_max, y_min and y_max), and the label itself. The labels here are RED, GREEN, YELLOW etc light colors.

Details are extracted from the csv file and stored in a dataframe. 

Object detection: There are two parts to object detection-

Object classification
Object localization

Bounding boxes are used usually for the localization purpose and the labels are used for classification. The two major techniques used in the industry for object detection are RCNNs and YOLO. I have dedicated the time spent on these assignments to learn about one of these techniques: RCNNs.

Region Based Convolutional Neural Networks

The Architecture of RCNN is very extensive as it has different blocks of layes for the above mentioned purposes: classification and localization.

The code I have used takes VGG-16 as the first block of layers which take in the images as 3D tensors and and give out feature maps. To understand the importance of Transfer learning, I have used pre-trained weights of this model. This is the base network.

The next network block is the Region Proposal Network. This is a Fully Convolutional Network. This network uses a concept of Anchors. It is a very interesting concept. This solves the problem of using exactly what length of bounding boxes. The image is scaled down and now each pixel woks as an anchor.

Each anchor defines a certain number of bounding box primitives. The RPN is used to predict the score of object being inside each of this bounding box primitive. A Region of INterest pooling layer appears next. This is a layer which takes in ROIs of the feature map to compare and classify each bounding box.

A post processing technique of Non-maximal supression is used to select the bounding box with the highest probability of the object being there. The image is scaled back up and this box is displayed.

Hyperparameters used- Number of samples for training- 1017, Number of samples for testing- 199 ROIs- 4 epoch length- 500 Epochs- 25 Anchors-9

All results are visible in the ipynb files of training and testing. I am planning to make the code more modular so that I can allocate resources to different modules separately and this issue is overcome. The accuarcy can further be improved by training over a larger dataset and running for more epochs. I will try to do this and improve the accuracy.

train.ipynb- used to train
test.ipynb- for generating loss graphs and finding mAP
yamltocsv.ipynb- is used ot convert the yaml file given with the datset to csv file for easy processing

References -

1. https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras by Rockyxu66- excellent tutorial for using RCNN code on google colab
2. https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a for understanding the RCNN flow
3. https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/ for understanding the RCNN architecture
4. https://colab.research.google.com/notebooks/welcome.ipynb#recent=true- Google colab for the resources
