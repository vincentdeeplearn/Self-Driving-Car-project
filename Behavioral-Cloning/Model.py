# coding: utf-8
import csv
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
import math
#read the train images' path  to the list:lines
lines=[]
with open(r'G:\PYthon data\SelfDrive\Term1\P3\p3--data\data\driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print('before lines length is :{}'.format(len(lines)))
#use sklearn Api split the dataset:lines to training data(80%) and validation data(20%)
train_samples,validation_samples=train_test_split(lines,test_size=0.2)

#brightness function randomly
def augment_brightness_camera_images(image):
    image1=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1=np.array(image1,dtype=np.float64)
    random_bright=.5+np.random.uniform()
    image1[:,:,2]=image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]=255
    image1=np.array(image1,dtype=np.uint8)
    image1=cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
#Horizontal and vertical shift the image.
def trans_image(image,steer,trans_range):
    rows,cols,channels=image.shape
    tr_x=trans_range*np.random.uniform()-trans_range/2
    steer_ang=steer+tr_x/trans_range*2*.2
    tr_y=40*np.random.uniform()-40/2
    trans_m=np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr=cv2.warpAffine(image,trans_m,(cols,rows))
    return image_tr,steer_ang
## Crop the image ,then resize image to 64*64
new_size_col,new_size_row = 64, 64
def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[50:shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
    return image
###  Produce a Generator 
def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1:  ### loop forever so the generator never terminates
        samples=shuffle(samples)
        ## split the input data to batchs
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
            images=[]
            angles=[]
            ## use for loop to traverse each sample in every batch_samples 
            for batch_sample in batch_samples:
                steering_center=float(batch_sample[3])
                #For augment data i use both the left and right camera's images ,set adjust correction:0.2
                correction=0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                ## loop over all the left/center/right camera images to augment the data for each sample
                for i in range(3):
                    # here use the order in the file to judge which is center(0) image or left(1)/right(2) image,
                    if i==0:
                        y_steer=steering_center
                    elif i==1:
                        y_steer=steering_left
                    else:
                        y_steer=steering_right
                    
                    source_path=batch_sample[i]
                    filename=source_path.split('/')[-1]
                    current_path='G:\\PYthon data\\SelfDrive\\Term1\\P3\\p3--data\\data\\IMG\\'+filename
                    image=cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2RGB)
                    image,y_steer = trans_image(image,y_steer,100)
                    image = augment_brightness_camera_images(image)
                    image = preprocessImage(image)
                    ## ====randomly flip the image with probability 0.5
                    ind_flip = np.random.randint(2)
                    if ind_flip==0:
                        image = cv2.flip(image,1)
                        y_steer = -y_steer
                    images.append(image)
                    angles.append(y_steer)
                X_train=np.array(images)
                y_train=np.array(angles)
                yield sklearn.utils.shuffle(X_train,y_train)

#Using the generator function generator train_generator and valid_generator
train_generator=generator(train_samples)
validation_generator=generator(validation_samples)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Activation
from keras.layers import Lambda,Cropping2D
import keras
from keras.layers import Convolution2D

# Now use the keras build the model
model=Sequential()
# use the lambda method to normalize the input data,since the origin data scale from 0 to 255.
model.add(Lambda(lambda x:x/127.5-1.0,input_shape=(64,64,3)))
#add Conv layers,strides=(5,5),use relu activation
model.add(Convolution2D(24,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
#Flattern the conv layers
model.add(Flatten())
#add full connect layers,and use relu activation
model.add(Dense(100,activation='relu'))
#add Dropout layers,drop rate=20%,to prevent overfitting
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='relu'))
#the final output do not use activation function
model.add(Dense(1))
#use the Adam to be the optiomizer and set learning rate=0.0008
opt=keras.optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer=opt)
# add EarlyStop to stop training if the loss do not decrease automatically
earlystop=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.002,patience=3,verbose=2,mode='auto')
##  use fit_generator method then can output a history object that contains the training 
## and validation loss for each epoch 
history_object=model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*3,
                                   validation_data=validation_generator,nb_val_samples=len(validation_samples)*3,
                                   epochs=10,callbacks=[earlystop])
model.save('model.h5')

#Visualizing the Loss
import matplotlib.pyplot as plt
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['training set','Validation set'],loc='upper right')
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='model.png')