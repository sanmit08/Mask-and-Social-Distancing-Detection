# Mask-and-Social-Distancing-Detection

During the pandemic it was mandatory for everyone to wear a face mask and maintain social distancing to prevent the rapid spread of the corona virus. These norms had to be strictly followed and monitored in public places due to large number of people outside. 

Keeping this in mind, a web application was created which would detect whether a person was wearing a mask or not and whether social distancing was being maintained or not in public places. This has been implemented with the help of deep learning and computer vision techniques.

## Face Mask Detection:
* The dataset from https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset was used as the primary dataset along with some manually added images. In total the dataset contained around 12 thousand images belonging to two classes: ‘WithMask’ and ‘WithoutMask’.
* This dataset was then trained on the CNN model ‘VGG16’, which would then classify an image of a face into ‘WithMask’ or ‘WithoutMask’.
* But before detecting the presence of a mask, the faces from an image had to be detected.
* For this the pretrained CNN model ‘MTCNN’ was used. This model would give the coordinates of the faces in the image as output. 
* Once the faces were detected, they were given sequentially to the earlier created model to obtain the final predictions.
* Results obtained are as shown in the image below:
![result1](https://user-images.githubusercontent.com/83156232/165504267-f295a5bd-25e6-41e3-b214-ca20c2f660fe.png)

## Social Distancing Detection:
* Our idea to implement social distancing was to ensure that for a given amount of time there shouldn’t be more then ‘x’ number of people in a given area.
* To implement this idea the main thing to do was count the number of people present in an area.
* This was done with the help of the ‘YOLO’ algorithm which was created using neural networks as well.
* If the number of people detected is greater than ‘x’ (predefined threshold for max people allowed) for more than ‘y’ secs (predefined threshold for max time allowed) an alert message would be printed on the screen asking everyone to spread out.
* Results obtained are as shown below:

![result2](https://user-images.githubusercontent.com/83156232/165504695-3a568535-c981-4395-8095-91972676e670.gif)
