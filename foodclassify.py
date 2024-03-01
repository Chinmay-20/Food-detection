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

'''
list all images under training
print(os.listdir(os.path.join(filea,"training")))
print()
print()
print()
print()
list all images under validation
print(os.listdir(os.path.join(filea,"validation")))
print()
print()
print()
print()
list all images under testing
print(os.listdir(os.path.join(filea,"evaluation")))
print()
print()
print()
	train.extend(os.listdir(os.path.join(i,'training')))
	
	valid.extend(os.listdir(os.path.join(filea,i)))
	test.extend(os.listdir(os.path.join(filea,i)))
for i in os.listdir(filea):
	if i=="training":
		train.extend(os.listdir(os.path.join(filea,i)))
	elif i=="validation":
		valid.extend(os.listdir(os.path.join(filea,i)))
	elif i=="evaluation":
		test.extend(os.listdir(os.path.join(filea,i)))
'''

train=[]
valid=[]
test=[]

for i in os.listdir(filea):
	if i=="training":
		filet=os.path.join(filea,i)
		for j in os.listdir(filet):
			train.extend(os.listdir(os.path.join(filet,j)))
	elif i=="validation":
		filev=os.path.join(filea,i)
		for j in os.listdir(filev):
			valid.extend(os.listdir(os.path.join(filev,j)))
	elif i=="evaluation":
		filee=os.path.join(filea,i)
		for j in os.listdir(filee):
			test.extend(os.listdir(os.path.join(filee,j)))


print("[INFO] Number of train images: {} \nNumber of validation images: {} \nNumber of test images: {}".format(len(train),len(valid),len(test)))

#visualization not done 

no_images_per_train_class=[]
train_class_name=[]

print("\n\nTraining")
for i in os.listdir(os.path.join(filea,"training")):
	train_class_name.append(i)
	train_class=os.listdir(os.path.join(filea,"training",i))
	print("[INFO] Number of images in {}={}".format(i,len(train_class)))
	no_images_per_train_class.append(len(train_class))
	
print("[INFO] classes names are {}".format(train_class_name))

no_images_per_val_class=[]
val_class_name=[]

print("\n\nValidation")
for i in os.listdir(os.path.join(filea,"validation")):
	val_class_name.append(i)
	val_class=os.listdir(os.path.join(filea,"validation",i))
	print("[INFO] Number of images in {}={}".format(i,len(val_class)))
	no_images_per_val_class.append(len(val_class))
	
print("[INFO] classes names are {}".format(val_class_name))

no_images_per_test_class=[]
test_class_name=[]

print("\n\nTesting")
for i in os.listdir(os.path.join(filea,"evaluation")):
	test_class_name.append(i)
	test_class=os.listdir(os.path.join(filea,"evaluation",i))
	print("[INFO] Number of images in {}={}".format(i,len(test_class)))
	no_images_per_test_class.append(len(test_class))
	
print("[INFO] classes names are {}".format(test_class_name))

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

print(basemodel.summary())

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

print("[INFO] added head to pre-trained network")

model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])

print("[INFO] model compiled")

#using earlystopping to exit training if validation is not decreasing even after after certain epochs(patience)
earlystopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=20)

#for saving the best model
checkpointer=ModelCheckpoint(filepath="output/primaryweights.hdf5",verbose=1,save_best_only=True)

print("[INFO] added callbacks and saving best model")

print("[INFO] training primary model")
history=model.fit(train_generator,steps_per_epoch=train_generator.n//32,epochs=25,validation_data=val_generator,validation_steps=val_generator.n//32,callbacks=[checkpointer,earlystopping])

print("[INFO primary model trained")

#finetuning model checking primary and finetuned model accuracy at same time
basemodel.trainable=True
print(model.summary())

#using earlystopping to exit training if validation is not decreasing even after after certain epochs(patience)
earlystopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=20)

#for saving the best model
checkpointer=ModelCheckpoint(filepath="output/finetuneweights.hdf5",verbose=1,save_best_only=True)

model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001,momentum=0.9),metrics=["accuracy"])
print("[INFO] compiled finetuned model")

print("[INFO] training finetuned model")
history=model.fit(train_generator,steps_per_epoch=train_generator.n//32, epochs=10, validation_data=val_generator,validation_steps=val_generator.n//32, callbacks=[earlystopping,checkpointer])

print("[INFO] trained finetuned model")

#loading pre-trained model
model.load_weights("output/finetunewieghts.hdf5")

evaluate=model.evaluate_generator(test_generator,steps=test_generator.n//32,verbose=1)

print("Accuracy Test: {}".format(evaluate[1]))

labels = {0: 'Bread', 1: 'Dairy product', 2: 'Dessert', 3:'Egg', 4: 'Fried food', 5:'Meat',6:'Noodles-Pasta',7:'Rice', 8:'Seafood',9:'Soup',10: 'Vegetable-Fruit'}

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
	img=img/255
	img=img.reshape(-1,256,256,3)
	predict=model.predict(img)
	predict=np.argmax(predict)
	prediction.append(labels[predict])
	original.append(i)
	
print(classification_report(np.asarray(original),np.array(prediction)))

