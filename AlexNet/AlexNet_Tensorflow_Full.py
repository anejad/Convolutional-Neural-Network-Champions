import datetime
import os,glob
import shutil 
import zipfile,tarfile
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.metrics import confusion_matrix

#-- tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical,multi_gpu_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.client import  device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#--
from utils import *


"""
Settings
"""
SEED=42
#---  Plotting Settings:
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (8, 6)
DPI=300
save_fig=True
save_path='./results/'
#---  Data  Location Settings:
# path to image folders
path_images='../ILSVRC/Data/CLS-LOC/'
# path to image class labels
path_labels='../'
# locations of the  images
train_directory=os.path.join(path_images,'train')
validation_directory=os.path.join(path_images,'val')
#--- Data Pipeline Settings:
image_height = 227
image_width = 227
EPOCHS = 90
BATCH_SIZE = 32
PATIENCE_FRACTION=1

def Pipelines(train_directory,validation_directory,BATCH_SIZE,SEED,image_height, image_width):
    train_datagen = ImageDataGenerator(
                  rescale=1./255,)
                  #rotation_range=10,
                  #width_shift_range=0.1,
                  #height_shift_range=0.1,
                  #shear_range=0.1,
                  #zoom_range=0.1)

    train_generator = train_datagen.flow_from_directory(train_directory,\
                                                    target_size=(image_height, image_width),\
                                                    color_mode="rgb",\
                                                    batch_size=BATCH_SIZE,\
                                                    seed=SEED,\
                                                    shuffle=True,\
                                                    class_mode="categorical")
    valid_datagen=ImageDataGenerator(rescale=1./255)

    valid_generator = valid_datagen.flow_from_directory(validation_directory,
                                                            target_size=(image_height, image_width),
                                                            color_mode="rgb",
                                                            batch_size=BATCH_SIZE,
                                                            seed=SEED,
                                                            shuffle=True,
                                                            class_mode="categorical",
                                                            )

    return train_generator,valid_generator

def AlexNet( input_shape, num_classes):
    initializer = tf.keras.initializers.GlorotNormal()
    model = Sequential(name='AlexNet')
    model.add(Conv2D(96, kernel_size=(11,11), strides= 4,
                    padding= 'valid', activation= 'relu',
                    input_shape= input_shape, kernel_initializer= initializer))
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None))

    model.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= initializer))
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)) 

    model.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= initializer))

    model.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= initializer))

    model.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= initializer))

    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None))

    model.add(Flatten())
    model.add(Dense(4096, activation= 'relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(4096, activation= 'relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1000, activation= 'relu'))
    model.add(Dense(num_classes, activation= 'softmax'))

    model.compile(optimizer= tfa.optimizers.SGDW(learning_rate=1e-5,momentum=0.9,weight_decay=0.0005,nesterov=True),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def TrainModel(model,valid_generator,train_generator,EPOCHS,BATCH_SIZE,PATIENCE_FRACTION):   
    early_stopping=EarlyStopping(monitor='val_loss',min_delta=0.0005,patience=int(EPOCHS*PATIENCE_FRACTION))
    checkpoint=ModelCheckpoint(filepath='./models/checkpoint_saves/AlexNet_checkpoint_save.h5',\
                               monitor='val_loss',save_weights_only=False)
    callback_list = [early_stopping,checkpoint]
    history=model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    callbacks=callback_list,
                    steps_per_epoch=train_generator.samples//BATCH_SIZE,
                    validation_steps= valid_generator.samples//BATCH_SIZE,
                    verbose=1)
    # laoding the best model 
    checkpoint_model=tf.keras.models.load_model('./models/checkpoint_saves/AlexNet_checkpoint_save.h5')
    return checkpoint_model,history

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create data pipelines
    train_generator,valid_generator,test_generator=Pipelines(train_directory,validation_directory,BATCH_SIZE,SEED,image_height, image_width)
    # Create model
    num_classes= len(np.unique(train_generator.classes))
    model = AlexNet((image_height, image_width, 3), num_classes)


# Train model
model,history=TrainModel(model,valid_generator,train_generator,EPOCHS,BATCH_SIZE*strategy.num_replicas_in_sync,PATIENCE_FRACTION)
