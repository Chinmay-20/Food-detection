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

filea="/home/spydie/Desktop/deep learning/food/dataset"

prediction=[]
original=[]
image=[]
count=0

for i in os.listdir(os.path.join(filea,"evaluation")):
	for item in os.listdir(os.path.join(filea,"evaluation",i)):
		img=PIL.Image.open(os.path.join(filea,"evaluation",i,item))
		count=count+1
		
print("{} number of images loaded successfully ".format(count))
