# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center_2017_10_22_12_46_55_588.jpg "Center image"
[image2]: ./center_2017_10_22_12_49_21_958.jpg "Left recovery"
[image3]: ./center_2017_10_22_12_49_25_180.jpg "Right recovery"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* models folder with several different working models

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used is copied from Nvidia. It consists of five convolutional layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 94-106).
These are followed by a flatten layer and 4 dense layers.

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is cropped (code line 95) and normalized in the model using a Keras lambda layer (code line 96). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 90-92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since I copied the Nvidia architecture, nothing needed to be changed there besides the input shape.

In order to gauge how well the modeles were working, I split my image and steering angle data into a training and validation set. I found that most of my models
got low training and validation error, but some of them didn't learn to turn according to the track.

To combat this, I modified the generator so that it drops some of the data belonging to straight driving.

Then I tweaked the correction factor used for right and left images to imitate sharper turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture didn't change at all, since it was copied and behaved correctly.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would be a cheap way to add extra data without having to actually drive counterclockwise.

I finally randomly shuffled the data set and put 500 entries of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by overfitting showed
when increased above 7 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
