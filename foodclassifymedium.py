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
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.losses import SparseCategoricalCrossentropy

filea="/home/spydie/Desktop/deeplearning/food/data"

#part 1

#ImageDataGenerator
trainaug=ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

testaug=ImageDataGenerator(rescale=1./255)

print("[INFO] data augmentation done")

train_generator=trainaug.flow_from_directory(os.path.join(filea,"training"), target_size=(256,256),batch_size=32,class_mode="categorical")

#val_generator=testaug.flow_from_directory(os.path.join(filea,"validation"), target_size=(256,256),batch_size=32,class_mode="categorical")

test_generator=testaug.flow_from_directory(os.path.join(filea,"evaluation"), target_size=(256,256),batch_size=32,class_mode="categorical")

print("[INFO] generator done")

#loading base model
basemodel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(256,256,3)))

#print(basemodel.summary())

print("[INFO] basemodel loading done")

basemodel.trainable=False

headmodel=basemodel.output
headmodel=GlobalAveragePooling2D(name="global_average_pool")(headmodel)
headmodel=Dropout(0.2)(headmodel)
headmodel=Dense(20,activation="softmax",name='dense_1')(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)

print(model.summary())

print("[INFO] added head to pre-trained network")

model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.0001),metrics=['accuracy'])

print("[INFO] model compiled")

#using earlystopping to exit training if validation is not decreasing even after after certain epochs(patience)
#earlystopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=20)

#for saving the best model
checkpointer=ModelCheckpoint(filepath="output/Pmobilenetmedium.hdf5",monitor="val_loss",verbose=1,save_best_only=True)

#print("[INFO] added callbacks and saving best model")

print("[INFO] training primary model")
history=model.fit(train_generator,epochs=200,validation_data=test_generator,callbacks=[checkpointer])

print("[INFO primary model trained")
'''
print("[INFO] evaluating after initialization")
model.load_weights("output/Pmobilenetmedium.hdf5")

evaluate=model.evaluate_generator(test_generator,steps=test_generator.n//32,verbose=1)

print("Accuracy Test: {}".format(evaluate[1]))

labels={0:'biryani',1:'bisibelebath',2:'butternaan',3:'chaat',4:'chappati',5:'dhokla',6:'dosa',7:'gulabjamun',8:'halwa',9:'idly',10:'kathi roll',11:'mendu vada',12:'noodles',13:'paniyaram',14:'poori',15:'samosa',16:'tandoori chicken',17:'upma',18:'vada pav',19:'ven pongal'}
#labels = {0: 'Bread', 1: 'Dairy product', 2: 'Dessert', 3:'Egg', 4: 'Fried food', 5:'Meat',6:'Noodles-Pasta',7:'Rice', 8:'Seafood',9:'Soup',10: 'Vegetable-Fruit'}

#loading images and predictions+

prediction=[]
original=[]
image=[]
count=0

for i in os.listdir(os.path.join(filea,"evaluation")):
	for item in os.listdir(os.path.join(filea,"evaluation",i)):
		img=PIL.Image.open(os.path.join(filea,"evaluation",i,item))
	
		img=img.resize((256,256))
		image.append(img)
		img=np.asarray(img,dtype=np.float32)
		img=img/255.0
		img=img.reshape(-1,256,256,3)
		predict=model.predict(img)
		predict=np.argmax(predict)
		prediction.append(labels[predict])
		original.append(i)
	
print(classification_report(np.asarray(original),np.array(prediction)))


#finetuning


#part 2
#part 3

#multiple dishes in one plate(pic)
#
