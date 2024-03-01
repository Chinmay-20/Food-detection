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
from keras.models import load_model

filea="/home/spydie/Desktop/deep learning/food/data"

model=load_model("output/Pmobilenetmedium.hdf5")

print("[INFO] model loaded...")

labels={0:'biryani',1:'bisibelebath',2:'butternaan',3:'chaat',4:'chappati',5:'dhokla',6:'dosa',7:'gulabjamun',8:'halwa',9:'idly',10:'kathi roll',11:'mendu vada',12:'noodles',13:'paniyaram',14:'poori',15:'samosa',16:'tandoori chicken',17:'upma',18:'vada pav',19:'ven pongal'}

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
