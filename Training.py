#!/usr/bin/env python
# coding: utf-8

# In[39]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[40]:


IMAGE_SIZE = 256 
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50


# In[41]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage", #Data directory inside training folder
    shuffle = True,  #Randomly shuffle the images and load them
    image_size = (IMAGE_SIZE, IMAGE_SIZE),#Our image sizes are 256, 256
    batch_size = BATCH_SIZE #32 batch size is standard size
    
)


# In[42]:


class_names = dataset.class_names
class_names #Our folder names are our class names


# In[43]:


print(len(dataset)) #Every element in the dataset is actually a batch of 32 images

print(68*32, "This is the total number of images not accurate though")


# In[44]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

#When we do this one batch(dataset.take(1)), it gives us one batch & one batch is 32 images
#(32, 256, 256, 3): this means, there are 32 images, each image is 256 by 256, and 3 channels i.e., RGB channels

#[1 1 0 0 0 1 1 0 0 1 2 2 1 0 0 1 1 0 0 0 1 1 2 0 0 0 0 1 0 1 0 0] means:
# 0 is for Potato___Early_blight
#1 is for Potato___Late_blight
#2 is for Potato___healthy


# In[45]:


#Lets print first image only for now:
for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].shape, "is the first image")
    print(image_batch[0].numpy()) #Those numbers are in 3d array, every number is bewteen 0 to 255(the color)


# In[46]:


plt.figure(figsize = (10,10)) #Increase the size of the area
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8")) #Converted float to int using uint8
        plt.axis("off") #To hide the number
        plt.title(class_names[label_batch[i]]) #Display the lable


# In[47]:


# 80% ==> training #Used for training
# 20% ==> 10% validation, 10% test


# In[48]:


len(dataset)


# In[49]:


train_size = 0.8
len(dataset)*train_size


# In[50]:


train_ds = dataset.take(54) # it'll take first 54 batches/ arr[:54]
len(train_ds)


# In[51]:


# Gives reamining 20% and then we split into test and validation
test_ds = dataset.skip(54) # arr[54:] #skip first 54 dataset and get the remaining ones
len(test_ds)


# In[52]:


val_size = 0.1
len(dataset)*val_size


# In[53]:


val_ds = test_ds.take(6)
len(val_ds)


# In[54]:


test_ds = test_ds.skip(6)
len(test_ds)


# In[58]:


def get_dataset_partitions_tf(ds, train_split = 0.8, val_split= 0.1, test_split = 0.1, shuffle =True, shuffle_size = 100000):
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
        
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[59]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[60]:


len(train_ds)


# In[62]:


len(val_ds)


# In[64]:


len(test_ds)


# In[67]:


# TO OPTIMIZE TRAINING TIME

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE) #It'll read the image from disk and for next iteration it'll keep that image in the memory
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE) 
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
# prefetch will load next set of batch from our disk which will improve our perforance
# cache will save time reading the images


# In[ ]:





# In[69]:


resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[71]:


#Data Augmentation - to make our model robust
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])


# In[74]:


#Building Our Model Now:

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

#Based on Trial and Error
model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'), 
    layers.Dense(n_classes, activation='softmax'), #softmax, it'll normalize the probability of our classes
])

model.build(input_shape=input_shape)


# In[76]:


model.summary()


# In[78]:


#Compling the Model now:
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[79]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,# Help us track the accuracy
    verbose=1,  #Prints lots of output
    epochs=50,
)


# In[81]:


scores = model.evaluate(test_ds)


# In[83]:


scores


# In[84]:


history


# In[85]:


history.params


# In[86]:


history.history.keys()


# In[87]:


type(history.history['loss'])


# In[88]:


len(history.history['loss'])


# In[89]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[91]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[92]:


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[94]:


# Running prediction on a sample image
import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# In[96]:


#Writing a function for Inference:
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[98]:


#Now runnig inference on few sample images:
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[99]:


#Saving the model: We append the model to the list of models as a new version:


# In[103]:


import os
model_version=max([int(i) for i in os.listdir("../models") + [0]])+1 #Automatically icrement the folders
model.save(f"../models/{model_version}")


# In[104]:


model.save("../potatoes.h5")


# In[ ]:




