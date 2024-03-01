import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

filea="/home/spydie/Desktop/deeplearning/food/dataset"

#part 1

#ImageDataGenerator
trainaug=ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

testaug=ImageDataGenerator(rescale=1./255)

print("[INFO] data augmentation done")

train_generator=trainaug.flow_from_directory(os.path.join(filea,"training"), target_size=(256,256),batch_size=32,class_mode="categorical")

val_generator=testaug.flow_from_directory(os.path.join(filea,"validation"), target_size=(256,256),batch_size=32,class_mode="categorical")

test_generator=testaug.flow_from_directory(os.path.join(filea,"evaluation"), target_size=(256,256),batch_size=32,class_mode="categorical")

print("[INFO] generator done")

#loading base model
basemodel=InceptionResNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(256,256,3)))

#print(basemodel.summary())

print("[INFO] basemodel loading done")

basemodel.trainable=False

headmodel=basemodel.output
headmodel=GlobalAveragePooling2D(name="global_average_pool")(headmodel)
headmodel=Flatten(name="flatten")(headmodel)
headmodel=Dense(256,activation="relu",name="dense_1")(headmodel)
headmodel=Dropout(0.3)(headmodel)
headmodel=Dense(128,activation="relu",name="dense_2")(headmodel)
headmodel=Dropout(0.3)(headmodel)
headmodel=Dense(11,activation="softmax",name='dense_3')(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)

print(model.summary())

print("[INFO] added head to pre-trained network")

model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])

print("[INFO] model compiled")

#using earlystopping to exit training if validation is not decreasing even after after certain epochs(patience)
earlystopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=20)

#for saving the best model
checkpointer=ModelCheckpoint(filepath="output/primaryweights.hdf5",verbose=1,save_best_only=True)

#print("[INFO] added callbacks and saving best model")

print("[INFO] training primary model")
history=model.fit(train_generator,steps_per_epoch=train_generator.n//32,epochs=25,validation_data=val_generator,validation_steps=val_generator.n//32,callbacks=[checkpointer,earlystopping])

print("[INFO primary model trained")

#part 2

#part 3
