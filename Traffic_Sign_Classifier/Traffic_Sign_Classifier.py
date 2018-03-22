
# coding: utf-8

# ##  Build a Traffic Sign Recognition Classifier
# 

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle

# Fill this in based on where you saved the training and testing data
from sklearn.model_selection import StratifiedShuffleSplit
training_file = r'C:\Users\Administrator\Self driving\Term2\CarND-Traffic-Sign-Classifier-Project\traffic-signs-data\train.p'
testing_file = r'C:\Users\Administrator\Self driving\Term2\CarND-Traffic-Sign-Classifier-Project\traffic-signs-data\test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_all, y_train_all = train['features'], train['labels']
sss=StratifiedShuffleSplit(n_splits=10,test_size=0.2)
for train_index,valid_index in sss.split(X_train_all, y_train_all):
    X_train,y_train=X_train_all[train_index], y_train_all[train_index]
    X_valid,y_valid=X_train_all[valid_index], y_train_all[valid_index]
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 

# ### Provide a Basic Summary of the Data Set

# In[2]:



import pandas as pd

n_train = X_train.shape[0]


n_validation = X_valid.shape[0]
print('Number of valid examples:',n_validation )

n_test = X_test.shape[0]


image_shape = X_train.shape[1:]


n_classes = len(pd.DataFrame(y_train)[0].unique())

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s).

# In[3]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')
index = random.randint(0, len(X_train))
image = X_train[index]

plt.figure(figsize=(2,2))
plt.imshow(image, cmap="gray")
print(y_train[index])
print(image.shape)


# In[4]:


count_train=pd.DataFrame(y_train)[0].value_counts()
count_valid=pd.DataFrame(y_valid)[0].value_counts()
count_test=pd.DataFrame(y_test)[0].value_counts()
count_train.hist(bins=30)


# In[6]:


count_valid.hist(bins=30)


# In[7]:


count_test.hist(bins=30)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. 

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# In[4]:



import cv2
import numpy as np
def gray(img):
    #transform single image to gray pics
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def scale(input):
    #input:the gray tensor,scale the input to (-1,1) 
    return (input*1.0-128)/128


def preprocess_gray(input):
    #input:the origin features,like:X_train/X_valid
    X_new=[]
    for img in input:
        temp=gray(img)
        X_new.append(temp)
    X_new=np.array(X_new)
    print(type(X_new))
    print('Before scale:',X_new.shape,'the mini value:',X_new.min(),'the maxi value is:',X_new.max())
    X_new=X_new.reshape([input.shape[0],input.shape[1],input.shape[2],1])
    X_new=scale(X_new)
    print('After scale:',X_new.shape,'the mini value:',X_new.min(),'the maxi value is:',X_new.max())
    return X_new
X_train_gray=preprocess_gray(X_train)
X_valid_gray=preprocess_gray(X_valid)
X_test_gray=preprocess_gray(X_test)


# In[55]:


temp=gray(X_train[234])
plt.imshow(temp,cmap='gray')


# In[8]:


#shuffle the training set

from sklearn.utils import shuffle

X_train_gray, y_train = shuffle(X_train_gray, y_train)


# ### Model Architecture

# In[5]:



from tensorflow.contrib.layers import flatten
import tensorflow as tf
def mynet(x):
    kernel_size=(5,5)
    #32*32*1 conv2d to 28*28*6
    conv1=tf.layers.conv2d(x,6,kernel_size,strides=(1,1),padding='valid',activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    #28*28*6 to 14*14*6
    pool1=tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #14*14*6 conv2d to 10*10*16
    conv2=tf.layers.conv2d(pool1,16,kernel_size,strides=(1,1),padding='valid',activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    #10*10*16 maxpool to 5*5*16
    pool2=tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #flatten 5*5*16 to 400
    fc0=flatten(pool2)
    #fullconnect 400 to 120
    fc1=tf.layers.dense(fc0,120,activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    #fullconnect 120 to 43(logits)
    logits=tf.layers.dense(fc0,43,activation=None,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    return logits


# ### Train, Validate and Test the Model

# In[6]:


### Train model 
#1 input
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
#Training Pipeline
rate = 0.001
logits = mynet(x)
output=tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
train_opt = tf.train.AdamOptimizer(learning_rate = rate).minimize(loss_operation)
#Model evaluation use Valid dataset
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



# In[9]:


### Train  model.and save the trained model
EPOCHS=30
BATCH_SIZE=128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_gray)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_gray, y_train = shuffle(X_train_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray[offset:end], y_train[offset:end]
            sess.run((loss_operation,train_opt), feed_dict={x: batch_x, y: batch_y})
            #print("train cost = {:.3f}".format(loss_operation))
        train_accuracy=evaluate(X_train_gray, y_train)
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        validation_accuracy = evaluate(X_valid_gray, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
    saver.save(sess, './mynet')
    print("Model saved")


# In[10]:


BATCH_SIZE=128
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gray, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# 

# ### Load and Output the Images

# In[11]:



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# In[12]:


#read all real german signs image
images=[]
for index in range(1,6):
    index=str(index)
    prefix='./real_german_traffic_signs/'
    img=mpimg.imread(prefix+index+'.jpg')
    images.append(img)


# In[13]:


len(images)


# In[14]:



#printing out some stats and plotting
#reading in an image
def blurZoomImg(image):
    #params:image is a single origin image 
    img=np.copy(image)
    img=cv2.GaussianBlur(img,(9,9),0)
    print('This Blur_image is:', type(img), 'with dimensions:', img.shape)
    #zoom in to shape 32*32*3
    img_zo=cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
    print('This Zoom_image is:', type(img_zo), 'with dimensions:', img_zo.shape)
    #plt.imshow(img_zo)
    return img_zo
def gray1(img):
    #transform single image to gray pics
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def scale1(input):
    #input:the gray tensor,scale the input to (-1,1) 
    return (input*1.0-128)/128
#preprocess image to the type need to feed into X placeholder.
def preprocess_gray1(input):
    #input:single origin pic for input
    X_new=blurZoomImg(input)
    X_new=gray1(X_new)
    X_new=np.array(X_new)
    print(type(X_new))
    #print('Before scale:',X_new.shape,'the mini value:',X_new.min(),'the maxi value is:',X_new.max())
    X_new=X_new.reshape([32,32,1])
    X_new=scale1(X_new)
    #print('After scale:',X_new.shape,'the mini value:',X_new.min(),'the maxi value is:',X_new.max())
    return X_new

real_x=[]
for each_img in images:
    real_x.append(preprocess_gray1(each_img))
real_x=np.array(real_x)
print('The real_x array shape is:{}'.format(real_x.shape))
## Feel free to use as many code cells as needed.


# ### Predict the Sign Type for Each Image

# In[15]:



import pandas as pd
#read the signsname.csv
df=pd.read_csv(r'C:\Users\Administrator\Self driving\Term2\CarND-Traffic-Sign-Classifier-Project\signnames.csv')
name_list=np.array(df['SignName'])
#predict the real image
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    result = sess.run(output,feed_dict={x:real_x})
print(np.array(result).shape)


# In[16]:


result=np.array(result)
prediction_y=[]
for row in range(len(images)):
    index=str(row+1)
    prefix='./real_german_traffic_signs/'
    img=mpimg.imread(prefix+index+'.jpg')
    plt.imshow(img)
    plt.show()
    prediction_y.append(np.argmax(result[row]))
    print('the NO.{} real sign image:class is:{},prediction is:{}'.format(row,np.argmax(result[row]),name_list[np.argmax(result[row])]))


# ### Analyze Performance

# In[17]:



real_y=np.array([25,14,3,17,13])
prediction_y=np.array(prediction_y)
cnt=0
for index in range(len(real_y)):
    if real_y[index]==prediction_y[index]:
        cnt+=1
accurate=cnt*1.0/len(real_y)
print('The accurate on these new total six images is:{}'.format(accurate))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# In[21]:


### Print out the top five softmax probabilities for the predictions 
with tf.Session() as sess:
    for row in range(len(images)):
        probabilities,indices=sess.run(tf.nn.top_k(result[row], k=5))
        #print('the NO. {} real traffic sign images top five probabilities is:{}'.format(row,probabilities))
        print('the NO. {} real traffic sign images top five probabilities is:'.format(row))
        for pro,index in zip(probabilities,indices):
            print('probabilities is:{:.5f},the class name is:{}'.format(pro,name_list[index]))


# In[ ]:




def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

