#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[3]:


IMAGE_SIZE = 256 
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50


# In[4]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage", #Data directory inside training folder
    shuffle = True,  #Randomly shuffle the images and load them
    image_size = (IMAGE_SIZE, IMAGE_SIZE),#Our image sizes are 256, 256
    batch_size = BATCH_SIZE #32 batch size is standard size
    
)


# In[5]:


class_names = dataset.class_names
class_names #Our folder names are our class names


# In[6]:


print(len(dataset)) #Every element in the dataset is actually a batch of 32 images

print(68*32, "This is the total number of images not accurate though")


# In[7]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

#When we do this one batch(dataset.take(1)), it gives us one batch & one batch is 32 images
#(32, 256, 256, 3): this means, there are 32 images, each image is 256 by 256, and 3 channels i.e., RGB channels

#[1 1 0 0 0 1 1 0 0 1 2 2 1 0 0 1 1 0 0 0 1 1 2 0 0 0 0 1 0 1 0 0] means:
# 0 is for Potato___Early_blight
#1 is for Potato___Late_blight
#2 is for Potato___healthy


# In[8]:


#Lets print first image only for now:
for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].shape, "is the first image")
    print(image_batch[0].numpy()) #Those numbers are in 3d array, every number is bewteen 0 to 255(the color)


# In[27]:


plt.figure(figsize = (10,10)) #Increase the size of the area
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8")) #Converted float to int using uint8
        plt.axis("off") #To hide the number
        plt.title(class_names[label_batch[i]]) #Display the lable


# In[28]:


# 80% ==> training #Used for training
# 20% ==> 10% validation, 10% test


# In[29]:


len(dataset)


# In[30]:


train_size = 0.8
len(dataset)*train_size


# In[31]:


train_ds = dataset.take(54) # it'll take first 54 batches/ arr[:54]
len(train_ds)


# In[32]:


# Gives reamining 20% and then we split into test and validation
test_ds = dataset.skip(54) # arr[54:] #skip first 54 dataset and get the remaining ones
len(test_ds)


# In[33]:


val_size = 0.1
len(dataset)*val_size


# In[34]:


val_ds = test_ds.take(6)
len(val_ds)


# In[35]:


test_ds = test_ds.skip(6)
len(test_ds)


# In[36]:


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


# In[37]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[38]:


len(train_ds)


# In[39]:


len(val_ds)


# In[40]:


len(test_ds)


# In[41]:


# TO OPTIMIZE TRAINING TIME

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE) #It'll read the image from disk and for next iteration it'll keep that image in the memory
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE) 
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
# prefetch will load next set of batch from our disk which will improve our perforance
# cache will save time reading the images


# In[48]:


import tensorflow as tf
from tensorflow.keras import layers


# In[51]:


resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])


# In[54]:


#Data Augmentation - to make our model robust
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


# In[55]:


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


# In[98]:


layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
layers.Dropout(0.5), # Example dropout layer with 50% dropout rate


# In[56]:


model.summary()


# In[57]:


#Compling the Model now:
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[90]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,# Help us track the accuracy
    verbose=1,  #Prints lots of output
    epochs=50,
)


# In[91]:


#Final Accuracy and Loss

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

final_accuracy = acc[-1]
final_loss = loss[-1]
print(f"Final Training Accuracy: {final_accuracy}")
print(f"Final Training Loss: {final_loss}")


# In[92]:


#Validation Metrics

final_val_accuracy = val_acc[-1]
final_val_loss = val_loss[-1]
print(f"Final Validation Accuracy: {final_val_accuracy}")
print(f"Final Validation Loss: {final_val_loss}")


# In[93]:


scores = model.evaluate(test_ds)


# In[94]:


#Test Set Performance

test_accuracy = scores[1]
test_loss = scores[0]
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")


# In[96]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get true labels and predicted labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[104]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import numpy as np

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)

# Now, we calculate precision, recall, and accuracy
precision = Precision()
recall = Recall()
accuracy = CategoricalAccuracy()

precision.update_state(y_true, y_pred)
recall.update_state(y_true, y_pred)
accuracy.update_state(y_true, y_pred)

# To compute F1 score, we can use the formula F1 = 2 * (precision * recall) / (precision + recall)
# We need to compute the result for precision and recall first
precision_result = precision.result().numpy()
recall_result = recall.result().numpy()
f1_score = 2 * (precision_result * recall_result) / (precision_result + recall_result)

print(f"Precision: {precision_result}")
print(f"Recall: {recall_result}")
print(f"F1 Score: {f1_score}")
print(f"Categorical Accuracy: {accuracy.result().numpy()}")


# In[68]:


scores


# In[69]:


history


# In[70]:


history.params


# In[71]:


history.history.keys()


# In[72]:


type(history.history['loss'])


# In[73]:


len(history.history['loss'])


# In[74]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[110]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[111]:


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


# In[113]:


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




